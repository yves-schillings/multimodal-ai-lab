"""Focused acceptance-gate behavior using explicitly artificial report fixtures."""
from copy import deepcopy
import json

import pytest

from lab.engineering_acceptance import evaluate_candidate_evidence


def fixture_manifest():
    """A minimal unit fixture, not a claim about an executed lab measurement."""
    source = {"case_id": "synthetic-unit-case", "text": "The synthetic bicycle was beside the library."}
    trials = []
    for expectation in ("supported", "refinement", "abstain", "multi_step"):
        count = 2 if expectation in ("refinement", "multi_step") else 1
        unsupported = expectation == "abstain"
        result = {"case_id": "synthetic-unit-case", "max_attempts": 3, "attempt_count": count,
                  "attempts": [{"query": "query " + str(i)} for i in range(count)],
                  "sources": [] if unsupported else [source],
                  "claims": [] if unsupported else [{"source": 1, "quote": "beside the library"}],
                  "abstained": unsupported, "stop_reason": "model_abstained" if unsupported else "answered"}
        trials.append({"expectation": expectation, "result": result, "passed": True,
                       "checks": {"same_case_revision": True, "case_specific_expected_behavior": True}})
    return {
        "rag": {"trials": trials, "passed_trials": 999},
        "vision": {"benchmarks": [{"model_call": {"samples": 10, "p50_ms": 1.0, "p95_ms": 2.0},
                                     "full_pipeline": {"samples": 10, "p50_ms": 2.0, "p95_ms": 4.0}}],
                   "training": {"epochs": 2},
                   "dataset": {"training_count": 20, "held_out_count": 10, "image_hash_overlap": 0,
                               "synthetic_only": True},
                   "model": {"weights_sha256": "a" * 64, "quality_threshold": 0.9},
                   "quality": {"cpu": {"samples": 10, "correct": 10, "accuracy": 1.0}}},
        "explainability": {"ok": True,
                           "quality_assessment": {"execution_ok": True,
                                                  "lime_illustrative_error_threshold": 0.05,
                                                  "lime_illustrative_gate_passed": True},
                           "shap": {"reconstruction_absolute_error": 1e-15,
                                    "reconstruction_tolerance": 1e-8, "reconstruction_passed": True},
                           "lime": {"surrogate_probability_at_case": 0.42, "model_probability": 0.4}},
    }


def statuses(result):
    """Map each check id of a gate result to its status ("pass", "fail" or "review_pending")."""
    return {check["id"]: check["status"] for check in result["checks"]}


def test_passing_technical_evidence_still_requires_separate_human_approval():
    """Passing evidence gives only "technical_candidate"; human approval stays pending and production is not approved."""
    result = evaluate_candidate_evidence(fixture_manifest())
    assert result["decision"] == "technical_candidate"
    assert result["manual_review_required"] is True and result["production_approved"] is False
    assert statuses(result)["governance.human_approval"] == "review_pending"
    assert all(len(digest) == 64 for digest in result["evidence_hashes"].values())


def test_fast_random_weight_benchmark_does_not_pass_quality_or_release():
    """Fast timings measured with random weights pass the timing check but block quality and release."""
    manifest = fixture_manifest()
    manifest["vision"]["model"]["weights_origin"] = "random weights; timing probe only"
    manifest["vision"].pop("training")
    manifest["vision"].pop("quality")
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["vision.performance_measured"] == "pass"
    assert statuses(result)["vision.trained_weights"] == "fail"
    assert statuses(result)["vision.held_out_quality"] == "fail"
    assert result["decision"] == "blocked" and not result["production_approved"]


def test_open_loop_probe_accounting_and_latency_do_not_release_random_weights():
    """Complete request accounting passes, random weights still block, and a broken accounting fails."""
    manifest = fixture_manifest()
    manifest["vision"] = {
        "schema_version": "vision-open-loop-v1", "release_eligible": False,
        "model": {"random_weights": True, "trained": False, "quality_evaluated": False},
        "cases": [{"status": "completed", "load": {
            "offered_requests": 10, "submitted_requests": 9, "completed_requests": 9,
            "successful_requests": 8, "error_requests": 1, "client_safety_drops": 1,
            "accounting_complete": True,
            "http_end_to_end_latency": {"samples": 8, "p50_ms": 10, "p95_ms": 20, "p99_ms": 25, "max_ms": 30},
        }}],
    }
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["vision.performance_measured"] == "pass"
    assert statuses(result)["vision.open_loop_accounting"] == "pass"
    assert statuses(result)["vision.trained_weights"] == "fail"
    assert statuses(result)["vision.held_out_quality"] == "fail"
    manifest["vision"]["cases"][0]["load"]["error_requests"] = 0
    assert statuses(evaluate_candidate_evidence(manifest))["vision.open_loop_accounting"] == "fail"


def test_quality_failure_is_not_hidden_by_fast_timing_or_global_ok():
    """A held-out quality below the threshold blocks, even with good timings and a global "ok" flag."""
    manifest = fixture_manifest()
    manifest["vision"]["ok"] = True
    manifest["vision"]["quality"]["cpu"].update(correct=5, accuracy=0.5)
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["vision.performance_measured"] == "pass"
    assert statuses(result)["vision.held_out_quality"] == "fail"
    assert result["decision"] == "blocked"


def test_failed_lime_case_blocks_even_if_report_claims_success():
    """A LIME output outside the probability range blocks, even when the report claims success."""
    manifest = fixture_manifest()
    manifest["explainability"]["lime"]["surrogate_probability_at_case"] = 1.425
    # Both ok and illustrative_gate remain true to prove raw numeric checks matter.
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["explainability.case_1.execution"] == "pass"
    assert statuses(result)["explainability.case_1.lime_case_fidelity"] == "fail"
    assert result["decision"] == "blocked"


def test_all_supplied_explanation_cases_are_evaluated_without_hiding_failure():
    """Every supplied explanation case is checked, so one failing case blocks beside a passing one."""
    manifest = fixture_manifest()
    first = manifest["explainability"]
    second = deepcopy(first)
    second["lime"].update(surrogate_probability_at_case=0.8, model_probability=0.4)
    manifest["explainability"] = [first, second]
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["explainability.case_1.lime_case_fidelity"] == "pass"
    assert statuses(result)["explainability.case_2.lime_case_fidelity"] == "fail"
    assert result["decision"] == "blocked"


def test_tree_comparison_checks_tree_shap_reconstruction_numerically():
    """The Tree SHAP sum is recomputed from the numbers, so a self-declared pass cannot hide an error."""
    manifest = fixture_manifest()
    manifest["explainability"] = {"linear_baseline": manifest["explainability"],
                                  "tree_model": {"shap": {"reconstruction_absolute_error": 0.1,
                                                           "reconstruction_tolerance": 1e-8,
                                                           "reconstruction_passed": True}}}
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["explainability.case_1.tree_shap_reconstruction"] == "fail"
    assert result["decision"] == "blocked"


@pytest.mark.parametrize("change", ["criterion", "quote", "cross_case", "attempts"])
def test_rag_counter_or_self_rating_does_not_override_failed_evidence(change):
    """A failed criterion, an invented quote, a source from another case or four searches block the gate.

    Args:
        change: Parametrized mutation of the synthetic RAG report intended to bypass its real checks.
    """
    manifest = fixture_manifest()
    trial = manifest["rag"]["trials"][0]
    if change == "criterion":
        trial["passed"] = False
    elif change == "quote":
        trial["result"]["claims"][0]["quote"] = "Invented supporting text"
    elif change == "cross_case":
        trial["result"]["sources"][0]["case_id"] = "another-case"
    else:
        trial["result"]["attempt_count"] = 4
    result = evaluate_candidate_evidence(manifest)
    assert result["decision"] == "blocked"


def test_missing_evidence_blocks_and_approval_fields_or_paths_are_rejected():
    """Missing evidence blocks; approval fields and file paths are refused as input."""
    assert evaluate_candidate_evidence({})["decision"] == "blocked"
    with pytest.raises(ValueError):
        evaluate_candidate_evidence({**fixture_manifest(), "approved_by": "invented"})
    with pytest.raises(ValueError):
        evaluate_candidate_evidence({"rag": "C:/private/arbitrary-report.json"})


def test_hashes_are_stable_for_key_order_and_change_with_evidence_without_file_io(monkeypatch):
    """Hashes ignore key order, change only with the evidence they cover, and no file is opened.

    Args:
        monkeypatch: pytest fixture replacing open to prove that gate evaluation performs no file I/O.
    """
    import builtins

    manifest = fixture_manifest()
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: pytest.fail("The pure evaluator opened a file"))
    original = evaluate_candidate_evidence(manifest)
    reordered = json.loads(json.dumps(manifest, sort_keys=True))
    assert original["evidence_hashes"] == evaluate_candidate_evidence(reordered)["evidence_hashes"]
    manifest["vision"]["quality"]["cpu"].update(correct=9, accuracy=0.9)
    changed = evaluate_candidate_evidence(manifest)
    assert original["evidence_hashes"]["vision"] != changed["evidence_hashes"]["vision"]
    assert original["evidence_hashes"]["rag"] == changed["evidence_hashes"]["rag"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "x" * (513 * 1024)],
                         ids=["nan", "infinity", "oversized"])
def test_nonfinite_or_oversized_manifest_is_rejected(value):
    """NaN, infinity and a manifest over 512 KiB are refused.

    Args:
        value: Parametrized non-finite value or oversized report fixture that canonicalization must reject.
    """
    manifest = fixture_manifest()
    manifest["rag"]["unexpected_value"] = value
    with pytest.raises(ValueError):
        evaluate_candidate_evidence(manifest)


def two_step_trial(manifest):
    """Return the two-step ("multi_step") trial of a fixture manifest."""
    return next(t for t in manifest["rag"]["trials"] if t["expectation"] == "multi_step")


def test_missing_two_step_case_blocks_even_when_the_three_basic_cases_pass():
    """Without a two-step case, the three basic questions still pass, but the gate stays blocked."""
    manifest = fixture_manifest()
    manifest["rag"]["trials"].remove(two_step_trial(manifest))
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["rag.expected_outcomes"] == "pass"
    assert statuses(result)["rag.multi_step_case"] == "fail"
    assert result["decision"] == "blocked"


def test_wrong_two_step_answer_blocks_although_its_quote_is_exact():
    """A wrong two-step answer with an exact quote keeps the quote check green and still blocks the gate."""
    manifest = fixture_manifest()
    trial = two_step_trial(manifest)
    trial["passed"] = False
    trial["checks"]["case_specific_expected_behavior"] = False
    result = evaluate_candidate_evidence(manifest)
    assert statuses(result)["rag.supporting_quotes"] == "pass"
    assert statuses(result)["rag.expected_outcomes"] == "pass"
    assert statuses(result)["rag.multi_step_case"] == "fail"
    assert result["decision"] == "blocked"


def test_two_step_answer_after_a_single_search_does_not_pass():
    """A two-step answer found after a single search does not pass the two-step check."""
    manifest = fixture_manifest()
    trial = two_step_trial(manifest)
    trial["result"].update(attempt_count=1, attempts=trial["result"]["attempts"][:1])
    assert statuses(evaluate_candidate_evidence(manifest))["rag.multi_step_case"] == "fail"


def test_paraphrase_rule_accepts_one_search_but_historical_refinement_rule_does_not():
    """One search fails the historical "refinement" rule and passes the "paraphrase" answer rule."""
    manifest = fixture_manifest()
    trial = next(t for t in manifest["rag"]["trials"] if t["expectation"] == "refinement")
    trial["result"].update(attempt_count=1, attempts=trial["result"]["attempts"][:1])
    assert statuses(evaluate_candidate_evidence(manifest))["rag.expected_outcomes"] == "fail"
    trial["expectation"] = "paraphrase"
    assert statuses(evaluate_candidate_evidence(manifest))["rag.expected_outcomes"] == "pass"
