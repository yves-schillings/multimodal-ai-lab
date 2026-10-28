"""Check a held-out RAG run against its freeze, then apply the analysis plan written before its results.

Steps, in the order of ``test-analysis-plan.json``:

1. Integrity: the run record carries the hashes of the freeze, and every strategy, repeat and question
   is present exactly once.
2. Per strategy: the frozen summary of ``lab.rag_benchmark`` and the strict success of the plan, with
   failures told apart (withheld output, unavailable service, deadline, invalid plan, wrong object).
3. Paired, case-clustered comparisons: the primary contrast, the secondary ones and all six pairs,
   with Holm-adjusted sign-flip values.
4. A blinded sample for the independent review of judgements, drawn with the seed of the plan.

Usage::

    python scripts/compare_rag_strategies.py --run test-run-cpu.json --freeze test-freeze.json \\
        --plan test-analysis-plan.json --out test-comparison.json --review-out test-review-sheet.json

It reads the run record and never changes it. Output files must not exist yet.
"""
import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.rag_benchmark import percentile, summarize_variant  # noqa: E402
from lab.rag_paired_analysis import FAILURE_STOPS, compare, holm, strict_success  # noqa: E402

STRATEGIES = ("fixed", "fixed_reranked", "agentic", "identifier_linked")
PRIMARY = ("agentic", "fixed_reranked")
SECONDARY = (("identifier_linked", "fixed_reranked"), ("fixed_reranked", "fixed"))
SEED = 20261028


def sha256(path):
    """Return the SHA-256 hex digest of the file at ``path``, read as raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_integrity(report, freeze, eval_set):
    """Compare the run record with its freeze and with the evaluation set.

    Args:
        report (dict): The run record.
        freeze (dict): The freeze record.
        eval_set (dict): The evaluation set.
    Returns:
        dict: ``hashes_match``, ``expected_trials``, ``trials``, ``missing`` and ``duplicates``, plus ``ok``.
    """
    environment = report["environment"]
    hashes_match = (environment["source_sha256"] == freeze["source_sha256"]
                    and environment["input_sha256"] == freeze["input_sha256"])
    questions = [q["id"] for case in eval_set["cases"] if case["split"] == report["split"] for q in case["questions"]]
    expected = {(s, r, q) for s in report["variants"] for r in range(1, report["repeats"] + 1) for q in questions}
    seen = [(t["variant"], t["repeat"], t["question_id"]) for t in report["trials"]]
    duplicates = sorted({key for key in seen if seen.count(key) > 1})
    missing = sorted(expected - set(seen))
    return {"hashes_match": hashes_match, "expected_trials": len(expected), "trials": len(seen),
            "missing": [list(key) for key in missing], "duplicates": [list(key) for key in duplicates],
            "finished": "finished_at" in report,
            "ok": hashes_match and not missing and not duplicates and "finished_at" in report}


def failure_category(trial):
    """Name why a trial is not a strict success, or return None when it is one.

    Args:
        trial (dict): A scored trial.
    Returns:
        str | None: ``withheld_output`` (a stored raw output failed a check), ``service_unavailable``,
        ``deadline``, ``invalid_plan``, ``wrong_object``, ``wrong_or_unsupported``, ``false_abstention``,
        ``answered_unanswerable`` or ``error``; None for a success.
    """
    if strict_success(trial):
        return None
    stop, verdict = trial.get("stop_reason"), trial["verdict"]
    if verdict.get("kind") == "error":
        return "error"
    if stop == "model_unavailable":
        return "withheld_output" if trial.get("raw_outputs") else "service_unavailable"
    if stop == "deadline":
        return "deadline"
    if stop == "invalid_model_plan":
        return "invalid_plan"
    if verdict.get("abstained"):
        return "false_abstention"
    if trial["expect"] == "abstain":
        return "answered_unanswerable"
    if verdict.get("entity_link_ok") is False:
        return "wrong_object"
    return "wrong_or_unsupported"


def describe(trials):
    """Summarise one strategy with the frozen summary and the strict success of the plan.

    Args:
        trials (list[dict]): Every trial of one strategy, access probes included.
    Returns:
        dict: ``frozen_summary`` (from ``summarize_variant``), ``strict_success`` (share of scored trials),
        ``failure_categories``, strict success per question type, latency over all scored trials and the
        number of deadlines.
    """
    scored = [t for t in trials if t["expect"] != "refused"]
    categories, per_type = {}, {}
    for trial in scored:
        category = failure_category(trial) or "success"
        categories[category] = categories.get(category, 0) + 1
        entry = per_type.setdefault(trial["category"], {"success": 0, "trials": 0})
        entry["success"] += category == "success"
        entry["trials"] += 1
    latency = [t["elapsed_ms"] for t in scored if t.get("elapsed_ms") is not None]
    return {
        "frozen_summary": summarize_variant(trials),
        "strict_success": round(categories.get("success", 0) / len(scored), 3),
        "failure_categories": dict(sorted(categories.items())),
        "strict_success_per_type": {k: {**v, "share": round(v["success"] / v["trials"], 3)}
                                    for k, v in sorted(per_type.items())},
        "latency_all_scored_ms": {"observations": len(latency), "p50": percentile(latency, 50),
                                  "p95": percentile(latency, 95), "max": max(latency) if latency else None},
        "deadlines_at_90_s": sum(1 for t in scored if t.get("stop_reason") == "deadline"),
    }


def review_sample(by_strategy, per_side=3, seed=SEED):
    """Draw the blinded review sample of the plan from repeat 1.

    Args:
        by_strategy (dict[str, list[dict]]): Trials per strategy.
        per_side (int): Successes and failures drawn per strategy.
        seed (int): Seed of the draw, fixed in the plan.
    Returns:
        tuple[list[dict], dict]: The blinded sheet (shuffled, without strategy names or verdicts) and the
        key mapping each sheet item to its strategy, question, repeat and automatic outcome.
    """
    generator, picked = random.Random(seed), []
    for name in STRATEGIES:
        first = [t for t in by_strategy.get(name, []) if t["repeat"] == 1 and t["expect"] != "refused"]
        good = [t for t in first if strict_success(t)]
        bad = [t for t in first if not strict_success(t)]
        picked += generator.sample(good, min(per_side, len(good))) + generator.sample(bad, min(per_side, len(bad)))
    queued = {(t["variant"], t["question_id"]) for trials in by_strategy.values() for t in trials
              if t["repeat"] == 1 and t["verdict"].get("review")}
    picked += [t for trials in by_strategy.values() for t in trials
               if t["repeat"] == 1 and (t["variant"], t["question_id"]) in queued and t not in picked]
    generator.shuffle(picked)
    sheet, key = [], {}
    for number, trial in enumerate(picked, 1):
        item = f"R{number:02d}"
        sheet.append({"item": item, "question_id": trial["question_id"], "question": trial["question"],
                      "expected_outcome": trial["expect"], "answer": trial.get("answer", ""),
                      "cited_quotes": [c.get("quote", "") for c in trial.get("claims", [])],
                      "reviewer_outcome": "", "reviewer_note": ""})
        key[item] = {"strategy": trial["variant"], "question_id": trial["question_id"], "repeat": trial["repeat"],
                     "automatic_strict_success": strict_success(trial), "automatic_kind": trial["verdict"]["kind"],
                     "in_review_queue": bool(trial["verdict"].get("review"))}
    return sheet, key


def main(run_path, freeze_path, plan_path, eval_set_path, out, review_out):
    """Write the comparison record and, when asked, the blinded review sheet and its key.

    Args:
        run_path, freeze_path, plan_path, eval_set_path (str): Input files.
        out (str): New JSON file for the comparison.
        review_out (str | None): New JSON file for the blinded sheet; its key goes beside it with ``-key``.
    Returns:
        dict: The comparison record.
    Raises:
        ValueError: An output file exists already.
    """
    targets = [Path(out)] + ([Path(review_out), Path(review_out).with_name(Path(review_out).stem + "-key.json")]
                             if review_out else [])
    if any(path.exists() for path in targets):
        raise ValueError("Choose new output filenames; existing results are preserved.")
    report = json.loads(Path(run_path).read_text(encoding="utf-8"))
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    by_strategy = {}
    for trial in report["trials"]:
        by_strategy.setdefault(trial["variant"], []).append(trial)
    names = [s for s in STRATEGIES if s in by_strategy]
    pairs = {f"{a} minus {b}": compare(by_strategy[a], by_strategy[b], seed=SEED)
             for a, b in combinations(names, 2)}
    adjusted = holm({name: item["sign_flip_check"] for name, item in pairs.items()})
    for name, item in pairs.items():
        item["sign_flip_holm"] = adjusted[name]

    def contrast(a, b):
        """Return the comparison ``a minus b``: the stored pair when it exists, otherwise a new computation."""
        if f"{a} minus {b}" in pairs:
            return pairs[f"{a} minus {b}"]
        return compare(by_strategy[a], by_strategy[b], seed=SEED)

    record = {
        "run_sha256": sha256(run_path), "freeze_sha256": sha256(freeze_path), "plan_sha256": sha256(plan_path),
        "integrity": check_integrity(report, freeze, eval_set),
        "failure_stops": list(FAILURE_STOPS),
        "strategies": {name: describe(by_strategy[name]) for name in names},
        "access_probes": {name: {"refused": sum(t["verdict"]["correct"] for t in by_strategy[name]
                                                if t["expect"] == "refused"),
                                 "total": sum(t["expect"] == "refused" for t in by_strategy[name])} for name in names},
        "primary": {"contrast": f"{PRIMARY[0]} minus {PRIMARY[1]}", **contrast(*PRIMARY)}
        if all(s in by_strategy for s in PRIMARY) else None,
        "secondary": {f"{a} minus {b}": contrast(a, b) for a, b in SECONDARY if a in by_strategy and b in by_strategy},
        "all_pairs_exploratory": pairs,
    }
    Path(out).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    if review_out:
        sheet, key = review_sample(by_strategy)
        targets[1].write_text(json.dumps(sheet, indent=1, ensure_ascii=False), encoding="utf-8")
        targets[2].write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"integrity_ok": record["integrity"]["ok"],
                      "strict_success": {n: s["strict_success"] for n, s in record["strategies"].items()},
                      "primary": record["primary"] and {k: record["primary"][k] for k in
                                                        ("difference", "case_bootstrap_95", "sign_flip_check")}},
                     ensure_ascii=False), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--review-out")
    args = parser.parse_args()
    main(args.run, args.freeze, args.plan, args.eval_set, args.out, args.review_out)
