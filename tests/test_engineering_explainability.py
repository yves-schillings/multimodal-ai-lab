"""Exercise input boundaries and real synthetic explanation numerics.

Schema/import tests run without the scientific extras. Module-scoped result and
comparison fixtures run CPU SHAP/LIME experiments when their distribution metadata
is available; otherwise dependent tests skip. Numerical assertions concern one
fixed synthetic dataset/case, including an intentionally failed high-score LIME
surrogate. Passing these tests is not real-invoice model or release acceptance."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from lab.engineering_explainability import explainability_status, invoice_schema, run_explainability_comparison, run_explainability_lab, validate_invoice


@pytest.fixture(scope="module")
def result():
    """Provide one module-scoped real default explanation report to tests.

    Returns:
        dict: run_explainability_lab's default held-out-case report, including the
        fitted synthetic model, SHAP/LIME diagnostics and reproducibility metadata.

    Pytest skips the fixture when optional distribution metadata is missing. Otherwise
    it performs bounded CPU fitting/explanation and can populate the model cache;
    scientific/import errors propagate as test errors rather than a false pass."""
    if not explainability_status()["available"]:
        pytest.skip("Optional SHAP/LIME environment is not installed")
    return run_explainability_lab()


def valid_invoice():
    """Return a fresh six-field fictional positive-score input dictionary.

    Values use the declared ratio, days and binary domains. Each call returns an
    independent dict so boundary tests can mutate it without affecting other tests.
    The helper does not train a model, read an invoice file or imply a real label."""
    return dict(amount_ratio=2.4, purchase_order_match=0, supplier_history_days=90, duplicate_reference=1, delivery_confirmed=0, recent_bank_change=1)


def test_module_import_does_not_load_optional_scientific_packages():
    """Verify lazy imports in a fresh isolated Python subprocess.

    The subprocess adds the repository root, imports engineering_explainability, and
    asserts that shap, lime and numpy are absent from sys.modules. A nonzero subprocess
    exit raises CalledProcessError and fails the test. This checks import isolation,
    not whether the optional packages can execute an experiment."""
    root = str(Path(__file__).resolve().parents[1])
    script = "import sys; sys.path.insert(0, " + repr(root) + "); import lab.engineering_explainability; assert 'shap' not in sys.modules; assert 'lime' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("field,value", [("amount_ratio", float("nan")), ("amount_ratio", 20), ("purchase_order_match", 0.5), ("supplier_history_days", -1), ("recent_bank_change", "1")])
def test_invalid_inputs_are_rejected_before_computation(field, value):
    """Reject each parametrized schema violation without fitting a model.

    Args:
        field (str): One valid feature key to overwrite in a fresh valid invoice.
        value: A nonfinite number, out-of-range ratio/day count, fractional binary
            flag or numeric string selected by pytest parametrization.

    validate_invoice must raise ValueError. The test concerns the six-field teaching
    schema, not business-data validation or model quality."""
    invoice = valid_invoice()
    invoice[field] = value
    with pytest.raises(ValueError):
        validate_invoice(invoice)


def test_missing_and_unexpected_fields_are_rejected():
    """Reject a missing feature combined with an unrecognized file_path key.

    A fresh input loses amount_ratio and gains a path-like string. Validation must
    raise ValueError without interpreting or opening that path; this test never asks
    the model to process an arbitrary serialized object."""
    invoice = valid_invoice()
    invoice.pop("amount_ratio")
    invoice["file_path"] = "arbitrary-model.pkl"
    with pytest.raises(ValueError):
        validate_invoice(invoice)


def test_shap_reconstructs_the_model_in_the_declared_output_space(result):
    """Check default Linear SHAP addition and its probability conversion.

    Args:
        result (dict): Real module-scoped default lab report from the result fixture.

    The baseline plus six shares must match the class-1 margin within 1e-8 log-odds,
    and its sigmoid must match model probability within 1e-10. The output-space label
    must say log-odds. These assertions test numerical consistency, not causation."""
    shap = result["shap"]
    reconstructed = shap["base_value_log_odds"] + sum(item["contribution_log_odds"] for item in shap["contributions"])
    assert math.isclose(reconstructed, shap["model_margin_log_odds"], rel_tol=0, abs_tol=1e-8)
    probability = 1 / (1 + math.exp(-reconstructed))
    assert math.isclose(probability, result["case"]["probability_review_suggested"], rel_tol=0, abs_tol=1e-10)
    assert "log-odds" in shap["output_space"]


def test_both_explanations_target_the_same_score_and_class(result):
    """Check class alignment, held-out separation and the default LIME diagnostic.

    Args:
        result (dict): Real default held-out-case explanation report.

    SHAP and LIME must explain class 1, LIME's reference must equal the classifier
    probability, and the 0.5 threshold must agree with the predicted class. The fixed
    800-row setup must split into 600/200 disjoint rows. Execution/reconstruction and
    the illustrative LIME range/error checks must pass for this default case; this
    does not assert that every input yields a reliable LIME explanation."""
    case = result["case"]
    assert result["shap"]["explained_class"] == result["lime"]["explained_class"] == case["positive_class"] == 1
    assert math.isclose(result["lime"]["model_probability"], case["probability_review_suggested"], abs_tol=1e-12)
    assert case["predicted_class"] == int(case["probability_review_suggested"] >= case["decision_threshold"])
    assert result["training"]["train_rows"] == 600
    assert result["training"]["test_rows"] == 200
    assert result["training"]["train_test_index_overlap"] == 0
    assessment = result["quality_assessment"]
    assert assessment["execution_ok"] is True
    assert assessment["shap_numerical_reconstruction_passed"] is True
    assert assessment["illustrative_only"] is True
    assert assessment["lime_case_in_probability_range"] is True
    assert assessment["lime_case_error"] <= assessment["lime_illustrative_error_threshold"] == 0.05
    assert assessment["lime_illustrative_gate_passed"] is True


def test_local_lime_fidelity_and_explanation_are_explicit(result):
    """Require explicit finite LIME outputs and independent-neighbourhood metadata.

    Args:
        result (dict): Real default explanation report.

    The six feature conditions/weights, fitting R2 and 256 separately seeded
    neighbourhood diagnostics must be present and finite. The report must retain its
    fitting-versus-validation and causation limitations and serialize as finite JSON.
    The artificial neighbourhood is not a representative real-data validation set."""
    lime = result["lime"]
    assert len(lime["contributions"]) == len(invoice_schema()) == 6
    assert all(item["condition"] and math.isfinite(item["weight"]) for item in lime["contributions"])
    assert math.isfinite(lime["fit_neighbourhood_weighted_r2"])
    independent = lime["independent_neighbourhood"]
    assert independent["rows"] == 256
    assert independent["seed"] != result["parameters"]["seed"]
    assert math.isfinite(independent["weighted_r2"])
    assert 0 <= independent["weighted_mae_probability"] < 1
    assert "not independent" in lime["fit_metric_note"]
    assert any("causation" in limitation for limitation in result["limitations"])
    json.dumps(result, allow_nan=False)


def test_same_seed_reproduces_the_explanations(result):
    """Repeat the default run and compare deterministic scientific outputs.

    Args:
        result (dict): Earlier default run used as the reference.

    The dataset fingerprint, selected case and SHAP/LIME contribution lists must be
    identical in this environment. Timing fields are deliberately excluded. Cached
    fitted models can be reused; the test does not promise identical floating-point
    results across package versions or hardware platforms."""
    repeated = run_explainability_lab()
    assert repeated["training"]["dataset_sha256"] == result["training"]["dataset_sha256"]
    assert repeated["case"] == result["case"]
    assert repeated["shap"]["contributions"] == result["shap"]["contributions"]
    assert repeated["lime"]["contributions"] == result["lime"]["contributions"]


def test_invalid_case_or_cost_parameters_are_bounded():
    """Reject sample, perturbation, seed and index values outside public limits.

    Each supplied option set violates one explicit bound before expensive experiment
    work: n_samples=5001, lime_samples=20001, seed=-1 or case_index=-1. Every call must
    raise ValueError rather than quietly clamp or run the requested oversized job."""
    for parameters in ({"n_samples": 5001}, {"lime_samples": 20001}, {"seed": -1}, {"case_index": -1}):
        with pytest.raises(ValueError):
            run_explainability_lab(**parameters)


@pytest.fixture(scope="module")
def comparison(result):
    """Provide a module-scoped real linear/tree comparison for one fictional input.

    Args:
        result (dict): Dependency fixture that first ensures scientific extras exist
            and establishes the compatible default synthetic dataset/model cache.

    Returns:
        dict: run_explainability_comparison's fixed high-score invoice report.

    The displayed input has no observed ground-truth label. Fitting/explanation runs
    on CPU and preserves the deliberately unreliable linear-model LIME diagnostic."""
    return run_explainability_comparison()


def test_tree_shap_reconstructs_a_real_positive_margin_on_the_same_split(comparison, result):
    """Check a fitted tree explanation and preserve the failed LIME counterexample.

    Args:
        comparison (dict): Real fixed high-score linear/tree comparison.
        result (dict): Default baseline report used to check the shared dataset/split.

    Tree SHAP must reconstruct the positive-class margin and sigmoid probability at
    1e-8/1e-10 tolerance, with a 600/200 split matching the baseline fingerprint. All
    numerical checks pass while the comparison still denies production approval and
    retains no ground-truth label. The high-score linear LIME surrogate must exceed 1
    and fail its illustrative gate; a successful test therefore preserves a known
    explanation failure instead of hiding or clipping it."""
    tree = comparison["tree_model"]
    shap = tree["shap"]
    reconstructed = shap["base_value_log_odds"] + sum(item["contribution_log_odds"] for item in shap["contributions"])
    assert math.isclose(reconstructed, shap["model_margin_log_odds"], rel_tol=0, abs_tol=1e-8)
    assert math.isclose(1 / (1 + math.exp(-reconstructed)), tree["case"]["probability_review_suggested"], abs_tol=1e-10)
    assert tree["case"]["predicted_class"] == 1
    assert tree["shap"]["contributors_increasing_score"]
    assert tree["training"]["dataset_sha256"] == result["training"]["dataset_sha256"]
    assert tree["training"]["train_rows"] == 600 and tree["training"]["test_rows"] == 200
    assert all(comparison["checks"].values())
    assert comparison["production_approval"] is False
    assert comparison["case"]["ground_truth_label"] is None
    # The deliberately difficult positive case must retain the failed LIME surrogate.
    assert comparison["linear_baseline"]["lime"]["surrogate_probability_at_case"] > 1
    assert comparison["quality_assessment"]["lime_assessment"]["lime_illustrative_gate_passed"] is False
    json.dumps(comparison, allow_nan=False)


def test_linear_coefficients_recover_raw_unit_predictions(comparison):
    """Verify coefficient conversion from standardized features to original units.

    Args:
        comparison (dict): Real comparison containing original-unit coefficients,
            training scales, explicit case features and the linear model's margin.

    Raw intercept plus coefficient times feature values must reproduce log-odds within
    1e-8. Multiplying a raw coefficient by its training scale must recover the
    standardized coefficient within 1e-10, and exponentiating the declared increment
    must recover the stated odds multiplier. These are algebraic, not causal checks."""
    coefficients = comparison["linear_coefficients"]
    features = comparison["case"]["features"]
    reconstructed = coefficients["intercept_original_units"] + sum(
        item["coefficient_per_original_unit"] * features[item["feature_key"]]
        for item in coefficients["features"]
    )
    assert math.isclose(reconstructed, comparison["linear_baseline"]["shap"]["model_margin_log_odds"], abs_tol=1e-8)
    for item in coefficients["features"]:
        assert math.isclose(
            item["coefficient_per_original_unit"] * item["training_standard_deviation"],
            item["coefficient_per_training_standard_deviation"], abs_tol=1e-10,
        )
        assert math.isclose(math.exp(item["log_odds_change_for_increment"]), item["odds_multiplier_for_increment"], abs_tol=1e-10)
