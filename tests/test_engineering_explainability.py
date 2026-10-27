"""Meaningful numerical and boundary checks for the optional synthetic XAI helper."""
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
    if not explainability_status()["available"]:
        pytest.skip("Optional SHAP/LIME environment is not installed")
    return run_explainability_lab()


def valid_invoice():
    return dict(amount_ratio=2.4, purchase_order_match=0, supplier_history_days=90, duplicate_reference=1, delivery_confirmed=0, recent_bank_change=1)


def test_module_import_does_not_load_optional_scientific_packages():
    root = str(Path(__file__).resolve().parents[1])
    script = "import sys; sys.path.insert(0, " + repr(root) + "); import lab.engineering_explainability; assert 'shap' not in sys.modules; assert 'lime' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("field,value", [("amount_ratio", float("nan")), ("amount_ratio", 20), ("purchase_order_match", 0.5), ("supplier_history_days", -1), ("recent_bank_change", "1")])
def test_invalid_inputs_are_rejected_before_computation(field, value):
    invoice = valid_invoice()
    invoice[field] = value
    with pytest.raises(ValueError):
        validate_invoice(invoice)


def test_missing_and_unexpected_fields_are_rejected():
    invoice = valid_invoice()
    invoice.pop("amount_ratio")
    invoice["file_path"] = "arbitrary-model.pkl"
    with pytest.raises(ValueError):
        validate_invoice(invoice)


def test_shap_reconstructs_the_model_in_the_declared_output_space(result):
    shap = result["shap"]
    reconstructed = shap["base_value_log_odds"] + sum(item["contribution_log_odds"] for item in shap["contributions"])
    assert math.isclose(reconstructed, shap["model_margin_log_odds"], rel_tol=0, abs_tol=1e-8)
    probability = 1 / (1 + math.exp(-reconstructed))
    assert math.isclose(probability, result["case"]["probability_review_suggested"], rel_tol=0, abs_tol=1e-10)
    assert "log-odds" in shap["output_space"]


def test_both_explanations_target_the_same_score_and_class(result):
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
    repeated = run_explainability_lab()
    assert repeated["training"]["dataset_sha256"] == result["training"]["dataset_sha256"]
    assert repeated["case"] == result["case"]
    assert repeated["shap"]["contributions"] == result["shap"]["contributions"]
    assert repeated["lime"]["contributions"] == result["lime"]["contributions"]


def test_invalid_case_or_cost_parameters_are_bounded():
    for parameters in ({"n_samples": 5001}, {"lime_samples": 20001}, {"seed": -1}, {"case_index": -1}):
        with pytest.raises(ValueError):
            run_explainability_lab(**parameters)


@pytest.fixture(scope="module")
def comparison(result):
    return run_explainability_comparison()


def test_tree_shap_reconstructs_a_real_positive_margin_on_the_same_split(comparison, result):
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
