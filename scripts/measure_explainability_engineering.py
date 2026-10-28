"""Run an optional synthetic explanation experiment from the command line.

The script chooses the single-model lab or same-split model comparison, accepts
only a six-field JSON numeric demonstration input, and prints finite UTF-8 JSON.
An optional output path receives the same text, creating its parent directories
and replacing that file if it already exists. This is a local CPU experiment,
not a model upload, external inference service or production release action.
See main for exit-code meaning; a zero exit code is not business acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.engineering_explainability import ExplainabilityUnavailable, run_explainability_comparison, run_explainability_lab


def main() -> int:
    """Parse CLI options, execute the selected experiment and emit its report.

    Arguments come from sys.argv: --seed, --n-samples, --lime-samples and --case-index
    feed bounded public helpers; --compare selects run_explainability_comparison;
    --invoice-json is an optional JSON object with the six numeric fields; --output
    is an optional filesystem Path. No function parameters are accepted.

    Returns:
        int: 0 when every result[checks] value is true, 1 when any is false, or 2 for
        caught dependency/input/JSON errors. The __main__ block uses it as exit status.

    Caught ExplainabilityUnavailable/ValueError/JSONDecodeError become an ok=False
    error object. Argparse can raise SystemExit for CLI usage errors; uncaught import,
    scientific, serialization and output-I/O errors propagate. stdout always receives
    one formatted report after normal execution/caught errors; --output creates its
    parents and writes identical text, overwriting an existing file. Numerical-check
    success may coexist with a failed LIME diagnostic or absent real-world model
    validation, so exit 0 never authorizes production."""
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
