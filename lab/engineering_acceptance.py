"""Pure evidence gate for a laboratory candidate, never production approval.

The caller supplies trusted measurement reports in memory. No file, URL, model
rating, upload, registry transition or human approval is read or written here.
"""
import hashlib
import json
import math
import re


_SHA256 = re.compile(r"[a-f0-9]{64}")
_MAX_BYTES = 512 * 1024


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _object(value):
    return value if isinstance(value, dict) else {}


def _canonical(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Evidence must be finite JSON data, not paths or executable objects.") from exc
    if len(data) > _MAX_BYTES:
        raise ValueError("Evidence manifest exceeds the 512 KiB laboratory limit.")
    return data


def evaluate_candidate_evidence(manifest):
    """Return a technical gate, checks and content hashes for supplied reports.

    Manifest keys: rag, vision, explainability. Explainability may contain one
    report, up to eight reports, or a linear/tree comparison report. Include all
    relevant cases; this helper cannot detect omitted or selectively chosen runs.
    Missing evidence blocks the candidate. Approval fields are deliberately not
    accepted; a separate authorized human decision is still required afterwards.
    """
    if not isinstance(manifest, dict) or set(manifest) - {"rag", "vision", "explainability"}:
        raise ValueError("Only rag, vision and explainability measurement reports are accepted.")
    _canonical(manifest)
    if any(not isinstance(value, (dict, list)) for value in manifest.values()):
        raise ValueError("Supply measurement values in memory; paths are not accepted.")
    checks = []

    def check(identifier, passed, detail):
        checks.append({"id": identifier, "status": "pass" if passed else "fail", "detail": detail})

    def pending(identifier, detail):
        checks.append({"id": identifier, "status": "review_pending", "detail": detail})

    hashes = {key: hashlib.sha256(_canonical(value)).hexdigest() for key, value in manifest.items()}
    for key in ("rag", "vision", "explainability"):
        check(key + ".evidence_present", bool(manifest.get(key)), "A trusted measurement report is required.")

    # RAG: recompute boundedness/citations; do not accept model self-ratings or
    # an aggregate passed counter as substitutes for per-trial observations.
    rag = _object(manifest.get("rag"))
    trials = rag.get("trials", [])
    valid_trials = isinstance(trials, list) and 1 <= len(trials) <= 16
    bounds_ok, citations_ok, outcomes_ok = bool(valid_trials), bool(valid_trials), bool(valid_trials)
    expectations = set()
    if valid_trials:
        for trial_value in trials:
            trial = _object(trial_value)
            result = _object(trial.get("result"))
            expectation = trial.get("expectation")
            if isinstance(expectation, str):
                expectations.add(expectation)
            attempts = result.get("attempts", [])
            sources, claims = result.get("sources", []), result.get("claims", [])
            limit, count = result.get("max_attempts"), result.get("attempt_count")
            bounded = (type(limit) is int and 1 <= limit <= 3 and type(count) is int
                       and isinstance(attempts, list) and 1 <= count == len(attempts) <= limit
                       and isinstance(sources, list) and len(sources) <= 3
                       and isinstance(claims, list) and len(claims) <= 3)
            if bounded:
                queries = [a.get("query") for a in attempts if isinstance(a, dict)]
                bounded = (len(queries) == count and all(isinstance(q, str) and 1 <= len(q) <= 2000 for q in queries)
                           and len({" ".join(q.casefold().split()) for q in queries}) == count
                           and all(isinstance(s, dict) and isinstance(s.get("text"), str)
                                   and s.get("case_id") == result.get("case_id") for s in sources)
                           and sum(len(s["text"]) for s in sources) <= 4500)
            bounds_ok = bounds_ok and bounded
            cited = bounded
            if cited:
                for claim in claims:
                    claim = _object(claim)
                    number, quote = claim.get("source"), claim.get("quote")
                    if (type(number) is not int or not 1 <= number <= len(sources)
                            or not isinstance(quote, str) or len(quote) < 8
                            or quote not in sources[number-1]["text"]):
                        cited = False
            citations_ok = citations_ok and cited
            if expectation == "abstain":
                observed = result.get("abstained") is True and not claims
            elif expectation in ("supported", "refinement"):
                observed = (result.get("abstained") is False and bool(claims)
                            and result.get("stop_reason") == "answered"
                            and (expectation != "refinement" or (type(count) is int and count > 1)))
            else:
                observed = False
            observations = _object(trial.get("checks"))
            outcomes_ok = (outcomes_ok and observed and trial.get("passed") is True
                           and bool(observations) and all(value is True for value in observations.values()))
    check("rag.bounded_case_evidence", bounds_ok, "Every trial must remain within three searches and three case-scoped sources.")
    check("rag.supporting_quotes", citations_ok, "Every returned claim requires an exact quote from its numbered source.")
    check("rag.expected_outcomes", outcomes_ok and {"supported", "refinement", "abstain"} <= expectations,
          "Supported, refined and unsupported questions must meet the predefined measurement criteria; counters alone do not pass.")
    pending("rag.semantic_review", "Human review of entailment, relevance and representative questions remains required.")

    # Vision: execution time and classifier quality are independent checks.
    vision = _object(manifest.get("vision"))
    benchmarks = vision.get("benchmarks", [])
    timings_ok = isinstance(benchmarks, list) and 1 <= len(benchmarks) <= 16
    load_probe = vision.get("schema_version") == "vision-open-loop-v1"
    if load_probe:
        cases = vision.get("cases", [])
        timings_ok = isinstance(cases, list) and 1 <= len(cases) <= 16
        accounting_ok = bool(timings_ok)
        if timings_ok:
            for value in cases:
                case = _object(value)
                load = _object(case.get("load"))
                stats = _object(load.get("http_end_to_end_latency"))
                percentiles = [stats.get(key) for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms")]
                timings_ok = (timings_ok and case.get("status") == "completed"
                              and type(stats.get("samples")) is int and stats["samples"] >= 2
                              and all(_number(p) and p >= 0 for p in percentiles)
                              and percentiles == sorted(percentiles))
                counts = [load.get(key) for key in ("offered_requests", "submitted_requests", "completed_requests",
                                                   "successful_requests", "error_requests", "client_safety_drops")]
                complete = all(type(n) is int and n >= 0 for n in counts)
                if complete:
                    offered, submitted, completed, successful, errors, dropped = counts
                    complete = (offered == submitted + dropped and submitted == completed
                                and completed == successful + errors and stats.get("samples") == successful
                                and load.get("accounting_complete") is True)
                accounting_ok = accounting_ok and complete
        check("vision.open_loop_accounting", accounting_ok,
              "All offered, submitted, successful, failed and dropped requests must be accounted for; rejections are not hidden.")
    elif timings_ok:
        for benchmark in benchmarks:
            for name in ("model_call", "full_pipeline"):
                stats = _object(_object(benchmark).get(name))
                p50, p95, samples = stats.get("p50_ms"), stats.get("p95_ms"), stats.get("samples")
                timings_ok = (timings_ok and type(samples) is int and samples >= 2
                              and _number(p50) and _number(p95) and 0 <= p50 <= p95)
    check("vision.performance_measured", bool(timings_ok), "Measured latency scope is checked separately from model quality and service acceptance.")
    training, dataset, model = (_object(vision.get(name)) for name in ("training", "dataset", "model"))
    weights = model.get("weights_sha256")
    trained = (type(training.get("epochs")) is int and training["epochs"] > 0
               and type(dataset.get("training_count")) is int and dataset["training_count"] > 0
               and isinstance(weights, str) and bool(_SHA256.fullmatch(weights)))
    random_weights = (model.get("random_weights") is True or model.get("trained") is False
                      or any("random" in str(model.get(name, "")).casefold()
                             for name in ("weights_origin", "weights_status", "weights")))
    check("vision.trained_weights", trained and not random_weights,
          "Random-weight timing benchmarks cannot qualify a model candidate for release.")
    threshold = model.get("quality_threshold")
    quality = _object(vision.get("quality"))
    quality_ok = (bool(quality) and _number(threshold) and 0.5 <= threshold <= 1
                  and type(dataset.get("image_hash_overlap")) is int and dataset["image_hash_overlap"] == 0
                  and type(dataset.get("held_out_count")) is int and dataset["held_out_count"] > 0)
    if quality_ok:
        for device_metrics in quality.values():
            metrics = _object(device_metrics)
            samples, correct, accuracy = metrics.get("samples"), metrics.get("correct"), metrics.get("accuracy")
            quality_ok = (quality_ok and type(samples) is int and samples == dataset["held_out_count"]
                          and type(correct) is int and 0 <= correct <= samples and _number(accuracy)
                          and abs(accuracy - correct / samples) <= 1e-9 and threshold <= accuracy <= 1)
    check("vision.held_out_quality", bool(quality_ok), "Observed held-out quality must meet the recorded laboratory threshold without split overlap.")
    pending("vision.operational_acceptance", "Synthetic classification and sequential timings do not accept real object detection, peak load or a service latency target.")

    # Explainability: every supplied case is checked, including known failures.
    supplied = manifest.get("explainability", [])
    cases = [supplied] if isinstance(supplied, dict) else supplied
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        check("explainability.case_count", False, "Supply one to eight explanation reports.")
        cases = []
    for index, value in enumerate(cases, 1):
        report = _object(value)
        comparison = "linear_baseline" in report
        baseline = _object(report.get("linear_baseline")) if comparison else report
        assessment = _object(baseline.get("quality_assessment"))
        prefix = f"explainability.case_{index}"
        check(prefix + ".execution", assessment.get("execution_ok") is True,
              "Successful execution is required but does not prove a useful explanation.")

        def shap_check(explanation, suffix):
            error = explanation.get("reconstruction_absolute_error")
            tolerance = explanation.get("reconstruction_tolerance")
            passed = (_number(error) and _number(tolerance) and 0 <= error <= tolerance <= 1e-8
                      and explanation.get("reconstruction_passed") is True)
            check(prefix + suffix, passed, "SHAP numerical reconstruction is checked; additivity does not prove causality or suitability.")

        shap_check(_object(baseline.get("shap")), ".shap_reconstruction")
        lime = _object(baseline.get("lime"))
        predicted, reference = lime.get("surrogate_probability_at_case"), lime.get("model_probability")
        threshold = assessment.get("lime_illustrative_error_threshold")
        lime_ok = (_number(predicted) and _number(reference) and 0 <= predicted <= 1 and 0 <= reference <= 1
                   and _number(threshold) and 0 <= threshold <= 0.05
                   and abs(predicted - reference) <= threshold
                   and assessment.get("lime_illustrative_gate_passed") is True)
        check(prefix + ".lime_case_fidelity", lime_ok,
              "Out-of-range or inaccurate local surrogate output blocks a reliable-explanation claim; no clipping is accepted.")
        if comparison:
            shap_check(_object(_object(report.get("tree_model")).get("shap")), ".tree_shap_reconstruction")
    pending("explainability.domain_review", "Domain experts must review explanation usefulness, stability, perturbations and limitations.")
    pending("governance.human_approval", "A separate authorized human approval has not been recorded by this evaluator.")
    failed = any(item["status"] == "fail" for item in checks)
    return {"schema_version": 1, "decision": "blocked" if failed else "technical_candidate",
            "manual_review_required": True, "production_approved": False,
            "checks": checks, "evidence_hashes": hashes,
            "limitations": [
                "The caller must supply trusted measurement reports; this is not a signature or source-authenticity verifier.",
                "Completeness, independence and representativeness of the supplied evidence are not established here.",
                "Technical-candidate status concerns this laboratory evidence only and is not permission to deploy.",
                "No human approval, model registry release, production capacity, compliance or business acceptance is performed.",
            ]}
