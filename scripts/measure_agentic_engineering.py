"""Run the historical three-question CPU/Ollama agent demonstration.

Uses reviewed synthetic case fixtures, simulated actors and the explicit
ReviewedLexicalRetriever. It never exercises BGE-M3 or pgvector despite retaining
an embedding-model pin in the configured inference client. The real Qwen3:4b
planner/generator is called with at most three searches and a deadline per
question. Local inference must already be running with the pinned model.

Writes a new JSON record incrementally and prints compact progress; existing
outputs are never overwritten. The historical paraphrase rule requires more
than one search, and the answer oracle includes a simple library substring
check. Passes are narrow fixture checks, not general factual/semantic accuracy.
The CLI exits 0 only if all three trial checks pass; otherwise it exits 1.
No pulls, provisioning, cluster acceptance or server-computation cancellation
is performed. The code requests keep_alive=0 for its inference calls.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.agentic_search import AgenticSearch, ReviewedLexicalRetriever
from lab.local_inference import InferenceUnavailable, LocalInference
from lab.store import StatementStore


GENERATION_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
EMBEDDING_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
TRIALS = (
    ("supported_location", "Where was the blue bicycle left?", "supported"),
    ("paraphrase_needs_retrieval", "At what location was the cycle parked?", "refinement"),
    ("unsupported_vehicle_registration", "What is the registration number of the vehicle?", "abstain"),
)


def measure(out, max_seconds=90):
    """Execute and record the three historical reviewed-only lexical trials.

    Args:
        out (str | Path): New JSON evidence filename outside the product source
            tree; its parent is created. Existing output is refused.
        max_seconds (float): Per-question elapsed budget in seconds, default 90;
            AgenticSearch validates > 0 and <= 300. At most three searches occur.
    Returns:
        dict: Synthetic scope, fixed model/options, per-trial result/checks,
        completed count and number meeting the historical fixture rules.
    Raises:
        ValueError: Output already exists or the controller budget is invalid.
        OSError: Parent/temp/store/output operations fail.
    Side effects:
        Creates/deletes a temporary StatementStore under the output parent, marks
        its seeded passages reviewed, calls local Ollama on CPU, writes the JSON
        after each trial and prints progress/final location. Expected inference,
        input, permission and missing-case errors become failed trial records.
    The quote check is verbatim membership; the location check is a substring.
    Neither checks entailment or the correct owner/object, and the simulated actor
    is not authentication. A deadline withholds late output rather than stopping
    server work. Returned failures must remain visible in any published summary.
    """
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new evidence filename; existing results are preserved.")
    out.parent.mkdir(parents=True, exist_ok=True)
    inference = LocalInference("http://127.0.0.1:11434", "bge-m3:latest", EMBEDDING_DIGEST,
                               "qwen3:4b", GENERATION_DIGEST)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Actual bounded search-controller/CPU-LLM trial on synthetic reviewed case evidence.",
        "retrieval": "reviewed-only lexical; no pgvector or embedding call exercised",
        "identity": "simulated case actors; existing local-session controls have separate behavior tests",
        "generation_model": "qwen3:4b", "generation_digest": GENERATION_DIGEST,
        "options": {"num_gpu": 0, "num_ctx_max": 4096, "num_predict_max": 384, "keep_alive": 0,
                    "max_attempts": 3, "max_seconds_per_question": max_seconds},
        "limitations": ["Three synthetic questions cannot establish general answer quality.",
                        "Exact source quotes do not establish entailment or the truth of a statement.",
                        "The deadline rejects late output; phase timeouts do not cancel server computation.",
                        "Cluster serving, pgvector, GPU latency, real user identity and deployment are not accepted here."],
        "trials": [],
    }
    with TemporaryDirectory(prefix="agentic-synthetic-fixture-", dir=out.parent) as temporary:
        store = StatementStore(Path(temporary))
        case = store.get_case("officer-a", "demo-case-a")
        for segment in list(case["segments"]):
            case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
        agent = AgenticSearch(ReviewedLexicalRetriever(), inference, max_attempts=3,
                             max_seconds=max_seconds, cpu_only=True)
        for name, question, expectation in TRIALS:
            trial = {"id": name, "question": question, "expectation": expectation}
            try:
                result = agent.answer(store, "officer-a", case["id"], question)
                sources, claims = result["sources"], result["claims"]
                current = store.get_case("officer-a", case["id"])
                quote_checks = all(
                    type(claim["source"]) is int and 1 <= claim["source"] <= len(sources)
                    and claim["quote"] in sources[claim["source"]-1]["text"]
                    for claim in claims)
                location = "library" in result["answer"].casefold()
                expected = (result["abstained"] and result["stop_reason"] in
                            ("model_abstained", "answer_abstained", "no_reviewed_evidence")
                            if expectation == "abstain" else
                            not result["abstained"] and location and
                            (expectation != "refinement" or result["attempt_count"] > 1))
                checks = {"same_case_revision": result["case_revision"] == current["revision"],
                          "supporting_quotes_checked": quote_checks,
                          "bounded_attempts": result["attempt_count"] <= 3,
                          "case_specific_expected_behavior": bool(expected)}
                trial.update(result=result, checks=checks, passed=all(checks.values()))
            except (InferenceUnavailable, ValueError, PermissionError, KeyError) as exc:
                # Record the type only: never provider bodies, credentials or private text.
                trial.update(passed=False, error_type=type(exc).__name__)
            report["trials"].append(trial)
            print(json.dumps({"trial": name, "passed": trial["passed"],
                              "stop_reason": trial.get("result", {}).get("stop_reason"),
                              "attempts": trial.get("result", {}).get("attempt_count")}), flush=True)
            # Save incrementally so failures/timeouts remain reviewable.
            report["passed_trials"] = sum(item["passed"] for item in report["trials"])
            report["completed_trials"] = len(report["trials"])
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"evidence": str(out), "passed": report["passed_trials"],
                      "total": report["completed_trials"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new local evidence JSON file outside the source repo")
    parser.add_argument("--max-seconds", type=float, default=90)
    args = parser.parse_args()
    result = measure(args.out, args.max_seconds)
    raise SystemExit(0 if result["passed_trials"] == result["completed_trials"] else 1)
