"""Run the pre-registered explainability review and write its JSON report.

The script reads the review protocol, checks that its parameters and thresholds are
exactly those the code will apply, runs lab.explainability_review on the CPU and
writes one finite UTF-8 JSON report. The report adds SHA-256 hashes of the protocol
and of the source files, the command line, UTC start/end times and the process
duration. The output file must not exist: evidence is never overwritten. This is a
local synthetic experiment, not a model upload, inference service or release action.
See main for exit-code meaning; exit 0 is not business or production acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.engineering_explainability import ExplainabilityUnavailable
from lab.explainability_review import THRESHOLDS, review_parameters, run_explainability_review

DEFAULT_PROTOCOL = ROOT / "docs" / "evidence" / "ai-engineering-review-20261028" / "explainability" / "protocol.json"
TUNABLE = ("seed", "n_samples", "lime_samples", "cases_per_stratum", "lime_seeds", "shap_backgrounds",
           "background_rows", "bootstrap_resamples", "coefficient_bootstrap_resamples")
SOURCES = ("lab/engineering_explainability.py", "lab/explainability_review.py", "scripts/review_explainability_engineering.py")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of one file's bytes.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lower-case hexadecimal digest.

    Raises:
        OSError: The file cannot be read."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path) -> tuple[dict, dict]:
    """Read the protocol and return it with the keyword arguments it registers.

    Args:
        path (Path): UTF-8 JSON protocol written before the run.

    Returns:
        tuple: (protocol dict, keyword arguments for run_explainability_review).

    Raises:
        OSError: The protocol cannot be read.
        json.JSONDecodeError: The protocol is not valid JSON.
        ValueError: A tunable parameter is missing or out of bounds, the complete
            parameter record differs from review_parameters for those values, or the
            thresholds differ from the module's THRESHOLDS.

    The check guarantees that the run applies exactly what was registered."""
    protocol = json.loads(path.read_text(encoding="utf-8"))
    registered = protocol.get("parameters", {})
    missing = [name for name in TUNABLE if name not in registered]
    if missing:
        raise ValueError(f"protocol parameters are missing: {missing}")
    arguments = {name: registered[name] for name in TUNABLE}
    if review_parameters(**arguments) != registered:
        raise ValueError("protocol parameters differ from the parameters the review code applies")
    if protocol.get("thresholds") != THRESHOLDS:
        raise ValueError("protocol thresholds differ from lab.explainability_review.THRESHOLDS")
    return protocol, arguments


def main(argv: list[str] | None = None) -> int:
    """Parse options, verify the protocol, run the review and write the report.

    Args:
        argv (list[str] | None): Command-line arguments without the program name;
            None reads sys.argv. Options: --protocol (default: the 28 October review
            protocol) and the required --output path of a file that must not exist.

    Returns:
        int: 0 when every recorded check verdict passed, 1 when the report was written
        with at least one failed check, 2 for an existing output file, a missing
        dependency, or an invalid or mismatched protocol (nothing is written).

    Before the run, CUDA_VISIBLE_DEVICES is set to -1 in this process so that no GPU is
    visible. The parent directory of --output is created when needed. The report is printed as a
    one-line summary on stdout; the full JSON goes only to the file. Scientific and
    unexpected I/O errors propagate. Exit 1 is the expected outcome when a check fails:
    failures are findings and are kept in the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; an existing file is refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        sys.stderr.write(f"Refusing to overwrite existing evidence: {args.output}\n")
        return 2
    try:
        protocol, arguments = load_protocol(args.protocol)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"Invalid protocol: {type(error).__name__}: {error}\n")
        return 2
    # Hide any GPU before the lazily imported scientific stack loads; this review is CPU only.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    try:
        result = run_explainability_review(**arguments)
    except ExplainabilityUnavailable as error:
        sys.stderr.write(f"{error}\n")
        return 2
    result["provenance"] = {
        "protocol_path": args.protocol.resolve().relative_to(ROOT).as_posix() if args.protocol.resolve().is_relative_to(ROOT) else args.protocol.name,
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_title": protocol.get("title"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "command": "python -B scripts/review_explainability_engineering.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "started_utc": started_at, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "process_seconds": time.perf_counter() - started,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    failed = [f"{row['check']} {row['method']} {row['model_id']} {row['scale']}" for row in result["check_table"] if row["verdict"] == "fail"]
    sys.stdout.write(json.dumps({"output": str(args.output), "all_checks_passed": result["all_checks_passed"],
                                 "failed_checks": failed, "seconds": round(result["timing_seconds"]["total"], 2)}) + "\n")
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
