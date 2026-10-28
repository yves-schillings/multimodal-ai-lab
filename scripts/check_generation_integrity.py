"""Check whether Qwen3:4b answers stay intact on a given Ollama device setting, before a benchmark relies on it.

The development run of 28 October showed stray colons inside generated claims ("Ms Dub: The parcel",
"TRK-771:04") and quotes that no longer matched their passage, so correct answers were withheld. This
script replays the same (question, passages) pairs through ``LocalInference.generate`` on each endpoint
and device setting, and counts invalid quotes and stray colons. It changes no model and no server setting;
a second Ollama server with other settings must be started separately and named with ``--endpoint``.

Usage::

    python scripts/check_generation_integrity.py --run dev-run-1.json --pairs 20 --repeats 2 \\
        --setting gpu=http://127.0.0.1:11434 --setting cpu=http://127.0.0.1:11434:cpu --out integrity.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from time import monotonic, perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.local_inference import InferenceUnavailable, LocalInference  # noqa: E402

GENERATION_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
EMBEDDING_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
# A colon glued inside a word or a number, or a short truncated word followed by a colon ("Ms Dub: The").
STRAY = re.compile(r"[a-z]:[a-z]|[A-Z]:[a-z]|\d:\d{2}\d|\b[A-Z][a-z]{1,3}: [A-Z]")


def pairs_from_run(path, count):
    """Return distinct (question, passage texts) pairs that a run gave to the generator.

    Args:
        path (str): A run record of ``benchmark_rag_variants.py``.
        count (int): Maximum number of pairs.
    Returns:
        list[tuple[str, list[str]]]: Questions with their recorded passages, in run order.
    """
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs, seen = [], set()
    for trial in run["trials"]:
        texts = [s["text"] for s in trial.get("sources") or []]
        key = (trial["question"], tuple(texts))
        if texts and key not in seen:
            seen.add(key)
            pairs.append((trial["question"], texts))
    return pairs[:count]


def replay(endpoint, cpu_only, pairs, repeats):
    """Generate an answer for every pair ``repeats`` times and classify each output.

    Args:
        endpoint (str): Loopback Ollama URL.
        cpu_only (bool): Force ``num_gpu`` 0 through the request budget.
        pairs (list): Output of ``pairs_from_run``.
        repeats (int): Generations per pair.
    Returns:
        dict: Counts of valid answers, abstentions, invalid quotes and stray colons, and the mean seconds.
    """
    client = LocalInference(endpoint, "bge-m3:latest", EMBEDDING_DIGEST, "qwen3:4b", GENERATION_DIGEST)
    counts = {"calls": 0, "valid": 0, "abstained": 0, "invalid_quote_or_output": 0, "stray_colon_in_valid": 0}
    seconds, examples = [], []
    for question, texts in pairs:
        for _ in range(repeats):
            sources = [{"text": t} for t in texts]
            started = perf_counter()
            with client.request_budget(monotonic() + 300, cpu_only=cpu_only, max_predict=384):
                try:
                    claims = client.generate(question, sources)
                    counts["valid" if claims else "abstained"] += 1
                    stray = [c["text"] for c in claims if STRAY.search(c["text"]) or STRAY.search(c["quote"])]
                    if stray:
                        counts["stray_colon_in_valid"] += 1
                        examples.append(stray[0][:120])
                except InferenceUnavailable:
                    counts["invalid_quote_or_output"] += 1
            counts["calls"] += 1
            seconds.append(perf_counter() - started)
    counts["mean_seconds"] = round(sum(seconds) / len(seconds), 2) if seconds else None
    counts["examples"] = examples[:5]
    return counts


def main():
    """Parse the options, replay the pairs on every setting and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run record that holds the generator inputs")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--setting", action="append", required=True,
                        help="name=URL, with ':cpu' appended to force num_gpu 0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("Choose a new output file; existing results are preserved.")
    pairs = pairs_from_run(args.run, args.pairs)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "pairs": len(pairs), "repeats": args.repeats,
              "settings": {}}
    for setting in args.setting:
        name, url = setting.split("=", 1)
        cpu = url.endswith(":cpu")
        url = url[:-4] if cpu else url
        report["settings"][name] = {"endpoint": url, "cpu_only": cpu, **replay(url, cpu, pairs, args.repeats)}
        print(name, json.dumps(report["settings"][name]), flush=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
