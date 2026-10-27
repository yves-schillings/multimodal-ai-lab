"""Run the optional synthetic explainability lab and emit a reproducible JSON report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.engineering_explainability import ExplainabilityUnavailable, run_explainability_comparison, run_explainability_lab


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--lime-samples", type=int, default=4000)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="Compare LogisticRegression and HistGradientBoosting on the same split, with a fixed fictional positive-score default")
    parser.add_argument("--invoice-json", help="Optional JSON object containing only the six bounded numeric demonstration inputs")
    parser.add_argument("--output", type=Path, help="Optional destination for the same JSON printed to stdout")
    args = parser.parse_args()
    try:
        invoice = json.loads(args.invoice_json) if args.invoice_json else None
        run = run_explainability_comparison if args.compare else run_explainability_lab
        result = run(invoice, seed=args.seed, n_samples=args.n_samples, lime_samples=args.lime_samples, case_index=args.case_index)
        exit_code = 0 if all(result["checks"].values()) else 1
    except (ExplainabilityUnavailable, ValueError, json.JSONDecodeError) as error:
        result = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
        exit_code = 2
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
