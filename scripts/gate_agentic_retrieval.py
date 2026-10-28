"""Evaluate one recorded comparison run with the laboratory evidence gate; never production approval.

The search trials come from a compare_agentic_retrieval.py run record. Each trial is judged by the rule written in
the protocol before the run: the answer rule for the three basic questions, the loop rule for the two-step question.
The vision and explanation reports are the existing records, supplied unchanged. A failed or missing two-step case
keeps the gate blocked until a corrected run passes it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.engineering_acceptance import evaluate_candidate_evidence  # noqa: E402


# Protocol expectation -> gate expectation. "refinement" is judged with the answer rule written before the run,
# so it becomes "paraphrase" (one to three searches); the historical verdict stays in the run record.
EXPECTATIONS = {"supported": "supported", "refinement": "paraphrase", "abstain": "abstain", "loop": "multi_step"}
RULES = {"supported": "answer_rule", "refinement": "answer_rule", "abstain": "answer_rule", "loop": "loop_rule"}


def gate_trial(trial):
    verdicts = trial["verdicts"]
    rule = RULES[trial["expectation"]]
    checks = {rule: verdicts[rule] is True}
    if "answer_named_expected_place" in verdicts:
        checks["answer_named_expected_place"] = verdicts["answer_named_expected_place"] is True
    return {"id": trial["id"], "expectation": EXPECTATIONS[trial["expectation"]], "result": trial["result"],
            "passed": checks[rule], "checks": checks}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(run, vision, explainability, out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new evidence filename; existing results are preserved.")
    record = json.loads(Path(run).read_text(encoding="utf-8"))
    manifest = {"rag": {"trials": [gate_trial(t) for t in record["trials"]]},
                "vision": json.loads(Path(vision).read_text(encoding="utf-8")),
                "explainability": json.loads(Path(explainability).read_text(encoding="utf-8"))}
    result = evaluate_candidate_evidence(manifest)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "purpose": "Laboratory evidence gate for one recorded search backend; not a human approval.",
              "backend": record["backend"],
              "inputs": {Path(p).name: sha256(p) for p in (run, vision, explainability)},
              "expectation_mapping": EXPECTATIONS, "rules_applied": RULES,
              "rag_trials": [{"id": t["id"], "expectation": t["expectation"], "passed": t["passed"],
                              "checks": t["checks"]} for t in manifest["rag"]["trials"]],
              "gate": result}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run record from compare_agentic_retrieval.py")
    parser.add_argument("--vision", required=True, help="Existing vision measurement record")
    parser.add_argument("--explainability", required=True, help="Existing explanation comparison record")
    parser.add_argument("--out", required=True, help="A new evidence JSON file")
    args = parser.parse_args()
    report = evaluate(args.run, args.vision, args.explainability, args.out)
    gate = report["gate"]
    print(gate["decision"], {c["id"]: c["status"] for c in gate["checks"] if c["id"].startswith("rag.")})
