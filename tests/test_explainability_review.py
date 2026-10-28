"""Exercise the explainability review helpers and a reduced real review run.

Pure helper tests use small hand-made arrays with known answers. The module-scoped
review fixture runs real CPU SHAP/LIME work with reduced budgets (one case per
stratum, two LIME seeds, two backgrounds, 500 LIME samples) when the optional
scientific distributions are installed; otherwise those tests skip. Passing these
tests checks the review machinery, not the quality of either explanation method."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from lab.engineering_explainability import explainability_status, validate_invoice
from lab.explainability_review import (
    FEATURE_KEYS, IMMUTABLE_FIELDS, REFERENCE_INVOICE, THRESHOLDS, _enumerated_shap, _jaccard,
    _pairwise_stability, _spearman, _top_k, bounded_counterfactual, calibration_table,
    review_parameters, run_explainability_review, select_review_cases,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "evidence" / "ai-engineering-review-20261028" / "explainability" / "protocol.json"
needs_extras = pytest.mark.skipif(not explainability_status()["available"], reason="Optional SHAP/LIME environment is not installed")


def test_review_module_import_is_lazy():
    """Import the review module in an isolated subprocess without scientific packages.

    The subprocess must not load numpy, shap, lime or sklearn at import time. A nonzero
    exit raises CalledProcessError and fails the test."""
    script = ("import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); import lab.explainability_review; "
              "assert not {'numpy', 'shap', 'lime', 'sklearn'} & set(sys.modules)")
    subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True)


@needs_extras
def test_calibration_table_matches_a_hand_computed_example():
    """Check bin assignment and both ECE values on four rows with known gaps.

    Probabilities 0.1, 0.9, 0.8 and 0.3 fall into bins 1, 9, 8 and 3 with gaps 0.1,
    0.1, 0.2 and 0.3, so the equal-width ECE is 0.175. With two equal-mass groups the
    sorted pairs (0.1, 0.3) and (0.8, 0.9) give gaps 0.2 and 0.15, so ECE is 0.175."""
    table = calibration_table([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], bins=10, equal_mass_groups=2)
    assert math.isclose(table["ece_equal_width"], 0.175, abs_tol=1e-12)
    assert math.isclose(table["ece_equal_mass"], 0.175, abs_tol=1e-12)
    assert [row["count"] for row in table["bins"]] == [0, 1, 0, 1, 0, 0, 0, 0, 1, 1]
    assert table["scale"] == "probability"
    assert calibration_table([1], [1.0])["bins"][-1]["count"] == 1


@needs_extras
@pytest.mark.parametrize("labels,probabilities", [([0, 2], [0.1, 0.2]), ([0, 1], [0.1, 1.2]), ([0, 1], [0.1]), ([], [])])
def test_calibration_table_rejects_invalid_inputs(labels, probabilities):
    """Reject non-binary labels, out-of-range probabilities and mismatched or empty inputs.

    Args:
        labels (list): Candidate labels from the parametrization.
        probabilities (list): Candidate probabilities from the parametrization."""
    with pytest.raises(ValueError):
        calibration_table(labels, probabilities)


@needs_extras
def test_case_selection_is_stratified_deterministic_and_reports_shortfall():
    """Select from a toy split where stratum D has a single row.

    The selection must be identical for the same seed, sorted, drawn from the right
    strata, and report D's shortfall without substituting another row."""
    y = [1, 1, 1, 0, 0, 0, 1, 1, 0]
    linear = [1, 1, 1, 0, 0, 0, 0, 0, 1]
    tree = [1, 1, 0, 0, 0, 0, 0, 1, 0]
    first = select_review_cases(y, linear, tree, seed=7, per_stratum=2)
    assert first == select_review_cases(y, linear, tree, seed=7, per_stratum=2)
    strata = {item["code"]: item for item in first["strata"]}
    assert set(strata["A"]["positions"]) <= {0, 1} and strata["A"]["selected"] == 2
    assert set(strata["B"]["positions"]) <= {3, 4, 5} and strata["B"]["selected"] == 2
    assert set(strata["C"]["positions"]) <= {2, 6, 7} and strata["C"]["selected"] == 2
    assert strata["D"]["positions"] == [8] and strata["D"]["shortfall"] == 1
    assert first["positions"] == sorted(first["positions"]) and len(first["positions"]) == 7
    with pytest.raises(ValueError):
        select_review_cases([0, 1], [0, 1], [0], seed=7, per_stratum=1)


@needs_extras
def test_rank_helpers_handle_ties_and_constant_vectors():
    """Check top-k tie order, Jaccard, Spearman and the undefined-correlation rule.

    Equal absolute values keep schema order; a constant vector has no defined rank
    correlation and must return None, which the stability summary counts as not passed."""
    assert _top_k([0.5, -0.5, 0.1, 0.5, 0.0, 0.0]) == frozenset({0, 1, 3})
    assert _jaccard(frozenset({0, 1, 2}), frozenset({0, 1, 3})) == 0.5
    assert math.isclose(_spearman([1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]), 1.0)
    assert _spearman([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]) is None
    stable = _pairwise_stability([[3, 2, 1, 0.5, 0.2, 0.1]] * 3, 0.8, 0.8)
    assert stable["passed"] and stable["mean_top3_jaccard"] == 1.0 and stable["pairs"] == 3
    constant = _pairwise_stability([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], 0.8, 0.8)
    assert constant["undefined_spearman_pairs"] == 1 and constant["passed"] is False


@needs_extras
def test_enumerated_shap_matches_the_closed_form_of_a_linear_function():
    """Compare direct enumeration with w_i (x_i - mean_i) for a linear function.

    For f(x) = w.x + b with an independent background, interventional Shapley values
    are known exactly. Enumeration must match them, add up to f(x) - mean f(background),
    and return f(x) as its full-coalition value."""
    import numpy as np
    weights = np.array([0.5, -1.0, 2.0, 0.0, 1.5, -0.25])

    def linear(rows):
        """Return a known linear function of the rows.

        Args:
            rows (numpy.ndarray): Inputs of shape (n, 6).

        Returns:
            numpy.ndarray: rows @ weights + 0.3."""
        return rows @ weights + 0.3

    background = np.random.default_rng(3).normal(size=(40, 6))
    row = np.array([1.0, 0.0, -2.0, 4.0, 0.5, 1.0])
    values, base, full = _enumerated_shap(np, linear, background, row)
    assert np.allclose(values, weights * (row - background.mean(axis=0)), atol=1e-12)
    assert math.isclose(base + values.sum(), full, abs_tol=1e-12)
    assert math.isclose(full, float(linear(row.reshape(1, -1))[0]), abs_tol=1e-12)


def test_review_parameters_reject_out_of_bounds_budgets():
    """Reject each budget outside its public bound before any computation.

    One value at a time exceeds lime_seeds, background_rows, cases_per_stratum or
    lime_samples bounds, or is a boolean; each must raise ValueError."""
    for options in ({"lime_seeds": 1}, {"background_rows": 151}, {"cases_per_stratum": 0}, {"lime_samples": 499}, {"seed": True}):
        with pytest.raises(ValueError):
            review_parameters(**options)


def test_protocol_registers_exactly_the_applied_parameters_and_thresholds():
    """Load the written protocol and compare it with what the code applies.

    The script's loader must accept the protocol, its thresholds must equal THRESHOLDS,
    and its parameters must equal review_parameters for the registered values."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from review_explainability_engineering import load_protocol
    protocol, arguments = load_protocol(PROTOCOL)
    assert protocol["thresholds"] == THRESHOLDS
    assert review_parameters(**arguments) == protocol["parameters"]
    assert protocol["written_before_any_run"] == "2026-10-28"


def test_script_refuses_to_overwrite_existing_evidence(tmp_path):
    """Return exit status 2 without running when the output file already exists.

    Args:
        tmp_path (Path): Pytest temporary directory holding a pre-existing file.

    The existing file's content must be unchanged."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from review_explainability_engineering import main
    existing = tmp_path / "report.json"
    existing.write_text("kept", encoding="utf-8")
    assert main(["--output", str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "kept"


@pytest.fixture(scope="module")
def review():
    """Provide one reduced real review run shared by the dependent tests.

    Returns:
        dict: run_explainability_review with one case per stratum, two LIME seeds, two
        SHAP backgrounds, 500 LIME samples and small bootstrap budgets.

    Skips when the optional scientific distributions are missing; scientific errors
    propagate as test errors."""
    if not explainability_status()["available"]:
        pytest.skip("Optional SHAP/LIME environment is not installed")
    return run_explainability_review(cases_per_stratum=1, lime_seeds=2, shap_backgrounds=2, lime_samples=500,
                                     bootstrap_resamples=20, coefficient_bootstrap_resamples=10)


def test_review_explains_the_same_cases_on_both_scales_with_shap_additivity(review):
    """Check shared cases, stated scales, SHAP additivity and the Exact cross-check.

    Args:
        review (dict): Reduced real review run.

    Both models must explain the same four stratified cases plus the reference case on
    both scales; SHAP must add up within the protocol tolerance and shap's Exact explainer
    must match the independent enumeration."""
    assert sorted(case["stratum"] for case in review["case_selection"]["selected_cases"]) == ["A", "B", "C", "D"]
    for model in review["explanations"].values():
        for scale in ("log_odds", "probability"):
            block = model["scales"][scale]
            assert len(block["cases"]) == 4 and block["reference_case"]["scale"] == scale
            summary = block["summary"]
            assert summary["C1_additivity_shap"]["verdict"] == "pass"
            assert summary["C1_additivity_shap"]["max_abs_error"] <= THRESHOLDS["additivity_abs_tolerance"]
            assert summary["C1b_implementation_crosscheck_shap"]["exact_verdict"] == "pass"
            assert all(case["shap"]["scale"] == case["lime"]["scale"] == scale for case in block["cases"])
    assert review["performance_and_calibration"]["logistic_regression"]["matches_base_helper_metrics"] is True
    assert review["performance_and_calibration"]["hist_gradient_boosting"]["matches_base_helper_metrics"] is True


def test_lime_reconstruction_matches_the_library(review):
    """Require the review's LIME design-space rebuild to reproduce LIME's own numbers.

    Args:
        review (dict): Reduced real review run.

    The recomputed surrogate value at the case and the recomputed fitting R2 must equal
    LIME's values, otherwise held-out fidelity and attributions would be invalid."""
    for model in review["explanations"].values():
        for scale in ("log_odds", "probability"):
            summary = model["scales"][scale]["summary"]["C2_C3_local_fidelity_lime"]
            assert summary["self_check_max_local_prediction_difference"] <= 1e-9
            assert summary["self_check_max_fit_r2_difference"] <= 1e-9
    assert review["assumptions_and_data"]["lime_neighbourhood"]["same_perturbations_for_both_models_and_scales"] is True


def test_review_report_is_finite_json_with_explicit_limits(review):
    """Check serialisation, verdict vocabulary and the not-measured statements.

    Args:
        review (dict): Reduced real review run.

    Every check row needs a verdict from the protocol vocabulary and a scale; reviewer
    usefulness and causality must be marked as not measured; production approval stays
    false."""
    json.dumps(review, allow_nan=False)
    assert all(row["verdict"] in ("pass", "fail", "not applicable") and row["scale"] in ("log_odds", "probability") for row in review["check_table"])
    assert review["not_measured"]["reviewer_usefulness"]["measured"] is False
    assert review["not_measured"]["causality"]["measured"] is False
    assert review["production_approval"] is False
    assert review["interactions"]["logistic_regression"]["log_odds"]["no_interaction"] is True


@needs_extras
def test_counterfactual_flips_the_decision_within_bounds_and_keeps_immutable_fields():
    """Validate a counterfactual for the fictional high-score input on both models.

    The returned input must re-predict to the other class, pass validate_invoice and keep
    supplier history, duplicate reference and bank change unchanged."""
    import numpy as np
    from lab.engineering_explainability import _fit_synthetic_model, _fit_synthetic_tree_model
    bundle = _fit_synthetic_model(1729, 800)
    tree = _fit_synthetic_tree_model(1729, 800)[0]
    row = np.asarray(validate_invoice(REFERENCE_INVOICE))
    amount_scale = float(bundle["x_train"][:, 0].std())
    for model in (bundle["model"], tree):
        result = bounded_counterfactual(model, row, amount_scale=amount_scale)
        if not result["found"]:
            assert result["reason"]
            continue
        assert result["validated"] and result["decision_flipped"] and result["schema_valid"]
        candidate = np.array([result["counterfactual"][key] for key in FEATURE_KEYS]).reshape(1, -1)
        assert int(model.predict(candidate)[0]) != result["original_class"]
        assert all(result["counterfactual"][key] == REFERENCE_INVOICE[key] for key in IMMUTABLE_FIELDS)
    with pytest.raises(ValueError):
        bounded_counterfactual(bundle["model"], row, amount_scale=0.0)
