"""Compare fixed RAG, fixed RAG with a reranker, agentic RAG and the identifier-linked correction.

All four variants answer the same evaluation-set questions on the same synthetic cases, with the same
simulated rights, BGE-M3 embeddings, Qwen3:4b generator and prompt, limits and GPU. Only the search
strategy differs:

- ``fixed``: one search with the question (three nearest passages at or above 0.45), one answer;
- ``fixed_reranked``: ten nearest passages reordered by BAAI/bge-reranker-v2-m3, three kept, one answer;
- ``agentic``: the unchanged ``AgenticSearch`` with the ``fixed`` retriever, up to three searches;
- ``identifier_linked``: ``fixed_reranked`` plus the identifier link of ``lab.rag_variants``.

Rules come from the evaluation set and the protocol, written before the runs. Development cases may
be used to fix bugs and choose settings; test cases run only after a freeze record hashes the code,
the evaluation set and the settings. A failure is recorded as a failure; there is no fallback.

Usage::

    python scripts/benchmark_rag_variants.py run --eval-set eval_set.json --protocol protocol.json \\
        --split dev --repeats 1 --out dev-run.json --gpu-lock <file>
    python scripts/benchmark_rag_variants.py summarize --run test-run.json --out test-summary.json

The reranker service (``scripts/serve_reranker.py``) must already listen on the loopback interface.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.agentic_search import AgenticSearch  # noqa: E402
from lab.local_inference import ANSWER_SCHEMA, SEARCH_PLAN_SCHEMA, InferenceUnavailable  # noqa: E402
from lab.rag_benchmark import judge_trial, paired_comparison, summarize_variant  # noqa: E402
from lab.rag_variants import (LINK_SYSTEM, ExactVectorRetriever, FixedSearch, IdentifierLinkedSearch,  # noqa: E402
                              LinkingInference, RerankedRetriever, RerankerClient)
from lab.store import ConflictError, StatementStore  # noqa: E402
from lab.vector_retrieval import reviewed_sources  # noqa: E402

# Pinned Ollama model digests, the same as the 28 October comparison.
GENERATION_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
EMBEDDING_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
VARIANTS = ("fixed", "fixed_reranked", "agentic", "identifier_linked")
# Source files whose SHA-256 is written into every run record.
SOURCES = ("lab/rag_variants.py", "lab/rag_benchmark.py", "lab/agentic_search.py", "lab/local_inference.py",
           "lab/vector_retrieval.py", "lab/store.py", "scripts/benchmark_rag_variants.py", "scripts/serve_reranker.py")
# Errors recorded as a failed trial (by type only) instead of stopping the run.
ERRORS = (PermissionError, KeyError, ValueError, ConflictError, InferenceUnavailable)


def sha256(path):
    """Return the SHA-256 hex digest of the file at ``path``, read as raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Meter:
    """Record every Ollama call of a client: phase, wall time, token counts and Ollama's own durations.

    It replaces the client's ``_request`` on the instance only, calls the original unchanged, and never
    alters a request or a response.
    """

    def __init__(self, client):
        """Wrap ``client._request`` and start an empty call list.

        Args:
            client (LocalInference): The pinned client whose calls are recorded.
        """
        self.calls, self._original = [], client._request
        client._request = self._request

    @staticmethod
    def _phase(path, payload):
        """Name the purpose of one request from its path and payload.

        Args:
            path (str): Ollama API path.
            payload (dict | None): Request body.
        Returns:
            str: ``tags``, ``embed``, ``plan``, ``generate``, ``link`` or ``other``.
        """
        if path == "/api/tags":
            return "tags"
        if path == "/api/embed":
            return "embed"
        if payload and payload.get("format") == SEARCH_PLAN_SCHEMA:
            return "plan"
        if payload and payload.get("format") == ANSWER_SCHEMA:
            return "generate"
        if payload and payload.get("system") == LINK_SYSTEM:
            return "link"
        return "other"

    def _request(self, method, path, payload=None):
        """Call the original ``_request`` and record the call, also when it fails.

        Args:
            method, path, payload: Passed unchanged to the original method.
        Returns:
            The original response.
        """
        started, response = perf_counter(), None
        try:
            response = self._original(method, path, payload)
            return response
        finally:
            record = {"phase": self._phase(path, payload), "ms": round((perf_counter() - started) * 1000, 1)}
            if isinstance(response, dict):
                record.update(prompt_tokens=response.get("prompt_eval_count", 0),
                              generated_tokens=response.get("eval_count", 0),
                              ollama_ms=round(response.get("total_duration", 0) / 1e6, 1),
                              load_ms=round(response.get("load_duration", 0) / 1e6, 1))
                if record["phase"] in ("plan", "link", "generate"):
                    # Kept for diagnosis only: run_trial stores it when an answer was withheld by a check.
                    record["raw_output"] = str(response.get("response", ""))[:1500]
            self.calls.append(record)

    def raw_outputs(self):
        """Return the raw language-model outputs of the trial, in call order, for diagnosing withheld answers.

        Returns:
            list[dict]: Phase and raw output text (at most 1,500 characters) of each plan, link or generate call.
        """
        return [{"phase": c["phase"], "output": c["raw_output"]} for c in self.calls if "raw_output" in c]

    def summary(self, rerank_calls, searches):
        """Summarise the recorded calls of one trial.

        Args:
            rerank_calls (list[dict]): The reranker client's calls during the trial.
            searches (int): Number of searches the controller reports.
        Returns:
            dict: Counts per phase, tokens, Ollama milliseconds and reranker milliseconds.
        """
        model = [c for c in self.calls if c["phase"] != "tags"]

        def count(phase):
            """Return how many recorded model calls of the trial had this phase."""
            return sum(1 for c in model if c["phase"] == phase)
        return {"searches": searches, "embed": count("embed"), "plan": count("plan"), "link": count("link"),
                "generate": count("generate"), "language_model_calls": count("plan") + count("link") + count("generate"),
                "rerank": len(rerank_calls), "rerank_ms": round(sum(c["ms"] for c in rerank_calls), 1),
                "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in model),
                "generated_tokens": sum(c.get("generated_tokens", 0) for c in model),
                "ollama_ms": round(sum(c.get("ollama_ms", 0) for c in model), 1),
                "load_ms": round(sum(c.get("load_ms", 0) for c in model), 1)}


def environment(paths, reranker):
    """Describe the code, data, versions, hardware and models of the run.

    Args:
        paths (list[Path]): Evaluation-set and protocol files whose SHA-256 is recorded.
        reranker (RerankerClient): The client whose service identity is recorded.
    Returns:
        dict: Commit, uncommitted files, file hashes, versions, CPU, GPU, Ollama and reranker identity.
    """
    import httpx

    def run(*command):
        """Run a local command and return its trimmed output, or None when it cannot start."""
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=ROOT).stdout.strip()
        except OSError:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"), "uncommitted": (run("git", "status", "--porcelain") or "").splitlines(),
            "source_sha256": {p: sha256(ROOT / p) for p in SOURCES},
            "input_sha256": {Path(p).name: sha256(p) for p in paths},
            "python": platform.python_version(), "os": platform.platform(), "logical_cpus": os.cpu_count(),
            "cpu": run("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name.Trim()")
            if os.name == "nt" else platform.processor(),
            "gpu": run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
            "ollama": httpx.get("http://127.0.0.1:11434/api/version", timeout=10, trust_env=False).json().get("version"),
            "httpx": httpx.__version__, "reranker": reranker.info()}


def build_store(root, eval_set, split):
    """Create the synthetic cases of one split, plus the other officer's cases, in a new store.

    Args:
        root (Path): Empty directory for the SQLite store.
        eval_set (dict): The evaluation set.
        split (str): ``dev`` or ``test``.
    Returns:
        tuple[StatementStore, dict]: The store and, per evaluation case ID, the store case ID, owner,
        ``positions`` (segment ID to 0-based position) and ``reviewed_ids``.
    """
    store, cases = StatementStore(root), {}
    selected = [c for c in eval_set["cases"] if c["split"] == split] + eval_set["foreign_cases"]
    for spec in selected:
        owner = spec["owner"]
        segments = [s if isinstance(s, dict) else {"text": s, "reviewed": True} for s in spec["segments"]]
        case = store.create_case(owner, spec["title"])
        case = store.save_transcript(owner, case["id"], {"engine": "synthetic-fixture", "language": "en", "segments": [
            {"start": 5.0 * i, "end": 5.0 * i + 5, "speaker": "Unverified speaker", "text": s["text"]}
            for i, s in enumerate(segments)]})
        for segment, wanted in zip(list(case["segments"]), segments):
            if wanted["reviewed"]:
                case = store.update_segment(owner, case["id"], segment["id"], segment["text"], case["revision"])
        case = store.get_case(owner, case["id"])
        cases[spec["id"]] = {"id": case["id"], "owner": owner, "revision": case["revision"],
                             "positions": {s["id"]: i for i, s in enumerate(case["segments"])},
                             "reviewed_ids": {s["id"] for s in reviewed_sources(case)}}
    return store, cases


def acquire(lock):
    """Wait until the shared GPU lock file is free, then create it for this benchmark.

    Args:
        lock (Path | None): Lock file shared with the other GPU measurements; None disables locking.
    """
    if lock is None:
        return
    while True:
        try:
            with open(lock, "x", encoding="utf-8") as handle:
                handle.write("rag")
            return
        except FileExistsError:
            sleep(15)


def release(lock):
    """Delete the GPU lock file if this benchmark holds it.

    Args:
        lock (Path | None): The lock file, or None when locking is disabled.
    """
    if lock is not None and lock.exists() and lock.read_text(encoding="utf-8") == "rag":
        lock.unlink()


def run_trial(name, controller, store, info, question, meter, reranker, repeat, split):
    """Ask one question through one controller and record everything needed to judge it.

    Args:
        name (str): Variant name.
        controller: ``FixedSearch``, ``IdentifierLinkedSearch`` or ``AgenticSearch``.
        store (StatementStore): The store that holds the case.
        info (dict): The case entry from ``build_store``.
        question (dict): The evaluation-set question.
        meter (Meter): Records the Ollama calls.
        reranker (RerankerClient): Records the reranker calls.
        repeat (int): Repeat number, from 1.
        split (str): ``dev`` or ``test``.
    Returns:
        dict: The trial record with its answer, citations, trace, calls, timing and verdict.
    """
    meter.calls, reranker.calls = [], []
    actor = question.get("actor", "officer-a")
    trial = {"variant": name, "repeat": repeat, "split": split, "question_id": question["id"],
             "category": question["category"], "expect": question["expect"], "question": question["question"],
             "actor": actor}
    started = perf_counter()
    try:
        result = controller.answer(store, actor, info["id"], question["question"])
    except ERRORS as exc:
        # Record the type only: never provider bodies or private text.
        result, trial["error_type"] = None, type(exc).__name__
    trial["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
    searches = len(result["attempts"]) if result else 0
    trial["calls"] = meter.summary(reranker.calls, searches)
    model_calls = trial["calls"]["embed"] + trial["calls"]["language_model_calls"] + trial["calls"]["rerank"]
    if result:
        trial.update(stop_reason=result["stop_reason"], abstained=result["abstained"], answer=result["answer"],
                     claims=result["claims"],
                     sources=[{"id": s["id"], "position": info["positions"].get(s["id"]), "text": s["text"]}
                              for s in result["sources"]],
                     attempts=result["attempts"])
        if result["stop_reason"] in ("model_unavailable", "invalid_model_plan"):
            # Synthetic text only: the raw output shows why a quote or a plan failed its check.
            trial["raw_outputs"] = meter.raw_outputs()
    trial["verdict"] = judge_trial(question, {"result": result, "error_type": trial.get("error_type"),
                                              "model_calls": model_calls}, info)
    print(json.dumps({"variant": name, "repeat": repeat, "q": question["id"], "kind": trial["verdict"]["kind"],
                      "stop": trial.get("stop_reason", trial.get("error_type")), "ms": trial["elapsed_ms"]}), flush=True)
    return trial


def run(eval_set_path, protocol_path, split, repeats, out, variants, reranker_url, gpu_lock, settings):
    """Run the selected variants on one split and write the run record after every trial.

    Args:
        eval_set_path (str): The evaluation set written before the runs.
        protocol_path (str): The protocol written before the runs.
        split (str): ``dev`` or ``test``.
        repeats (int): Number of repeats; the variant order rotates at each repeat.
        out (str): A new JSON file; an existing file is never overwritten.
        variants (list[str]): Variant names, from ``VARIANTS``.
        reranker_url (str): Loopback URL of the reranker service.
        gpu_lock (str | None): Shared GPU lock file.
        settings (dict): ``min_probability`` and ``link_candidates``.
    Returns:
        dict: The run record.
    Raises:
        ValueError: ``out`` exists or a variant name is unknown.
    """
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new evidence filename; existing results are preserved.")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    inference = LinkingInference("http://127.0.0.1:11434", "bge-m3:latest", EMBEDDING_DIGEST, "qwen3:4b", GENERATION_DIGEST)
    meter, reranker = Meter(inference), RerankerClient(reranker_url)
    lock = Path(gpu_lock) if gpu_lock else None
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "split": split, "repeats": repeats,
              "variants": variants, "settings": settings,
              "environment": environment([eval_set_path, protocol_path], reranker),
              "conditions": {"generation_model": "qwen3:4b", "generation_digest": GENERATION_DIGEST,
                             "embedding_model": "bge-m3:latest", "embedding_digest": EMBEDDING_DIGEST,
                             "device": ("CPU only: num_gpu 0 on every embedding and generation call, keep_alive 0"
                                        if settings.get("cpu_only") else
                                        "Ollama default GPU placement (no num_gpu override); keep_alive 0 on every call"),
                             "limits": "90 s per question, num_ctx at most 4096, at most 384 generated tokens per call; "
                                       "agentic: at most 3 searches"}}
    acquire(lock)
    try:
        # The synthetic store lives in the system temporary folder, never beside the evidence.
        with TemporaryDirectory(prefix="rag-benchmark-") as temporary:
            store, cases = build_store(Path(temporary), eval_set, split)
            base = ExactVectorRetriever(inference)
            reranked = RerankedRetriever(base, reranker, min_probability=settings["min_probability"])
            cpu = bool(settings.get("cpu_only"))
            controllers = {"fixed": FixedSearch(base, inference, max_seconds=90, cpu_only=cpu),
                           "fixed_reranked": FixedSearch(reranked, inference, max_seconds=90, cpu_only=cpu),
                           "agentic": AgenticSearch(base, inference, max_attempts=3, max_seconds=90, cpu_only=cpu),
                           "identifier_linked": IdentifierLinkedSearch(reranked, inference, max_seconds=90,
                                                                       cpu_only=cpu,
                                                                       link_candidates=settings["link_candidates"])}
            controllers["fixed_reranked"].mode = "fixed_rag_reranked"
            # Moment 1 of RAG: index every case of the split once, before any question.
            report["index_preparation"] = []
            for spec in eval_set["cases"]:
                if spec["split"] == split:
                    meter.calls = []
                    started = perf_counter()
                    # The same device policy as the questions: CPU-only runs force num_gpu 0 here too.
                    with inference.request_budget(monotonic() + 300, cpu_only=cpu):
                        base.synchronize(store.get_case(spec["owner"], cases[spec["id"]]["id"]))
                    report["index_preparation"].append({"case": spec["id"],
                                                        "embed_calls": sum(c["phase"] == "embed" for c in meter.calls),
                                                        "ms": round((perf_counter() - started) * 1000, 1)})
            report["trials"] = []
            for repeat in range(1, repeats + 1):
                order = variants[(repeat - 1) % len(variants):] + variants[:(repeat - 1) % len(variants)]
                for name in order:
                    for spec in eval_set["cases"]:
                        if spec["split"] != split:
                            continue
                        for question in spec["questions"]:
                            report["trials"].append(run_trial(name, controllers[name], store, cases[spec["id"]],
                                                              question, meter, reranker, repeat, split))
                            out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=sorted),
                                           encoding="utf-8")
    finally:
        release(lock)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=sorted), encoding="utf-8")
    return report


def summarize(run_path, out):
    """Summarise a run record per variant and compare every variant with the reranked baseline.

    Args:
        run_path (str): A run record written by ``run``.
        out (str): A new JSON file for the summary.
    Returns:
        dict: Per-variant summaries and paired comparisons.
    Raises:
        ValueError: ``out`` already exists.
    """
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new summary filename; existing results are preserved.")
    report = json.loads(Path(run_path).read_text(encoding="utf-8"))
    by_variant = {}
    for trial in report["trials"]:
        by_variant.setdefault(trial["variant"], []).append(trial)
    summary = {"run_sha256": sha256(run_path), "split": report["split"], "repeats": report["repeats"],
               "settings": report["settings"],
               "variants": {name: summarize_variant(trials) for name, trials in by_variant.items()},
               "paired_against_fixed_reranked": {name: paired_comparison(trials, by_variant["fixed_reranked"])
                                                 for name, trials in by_variant.items()
                                                 if name != "fixed_reranked" and "fixed_reranked" in by_variant}}
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, item in summary["variants"].items():
        print(f"{name:18} acc {item['accuracy']} {item['accuracy_95ci']} wrong/unsupported "
              f"{item['wrong_or_unsupported_answers']} false-abst {item['false_abstention_rate']} "
              f"p50 {item['latency_ms']['p50']} ms p95 {item['latency_ms']['p95']} ms "
              f"LM calls {item['mean_per_question']['language_model_calls']}")
    return summary


def freeze(eval_set_path, protocol_path, amendments_path, out):
    """Write the freeze record: hashes of the evaluation set, the protocol, the code and the amendments.

    The test run must use exactly these files. Its own run record repeats the same hashes, so anyone can
    check that nothing changed between the freeze and the test run.

    Args:
        eval_set_path (str): The evaluation set.
        protocol_path (str): The protocol written before any run.
        amendments_path (str): JSON list of the development changes and the final settings.
        out (str): A new JSON file for the freeze record.
    Returns:
        dict: The freeze record.
    Raises:
        ValueError: ``out`` already exists.
    """
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new freeze filename; existing records are preserved.")
    record = {"frozen_at": datetime.now(timezone.utc).isoformat(),
              "input_sha256": {Path(p).name: sha256(p) for p in (eval_set_path, protocol_path)},
              "source_sha256": {p: sha256(ROOT / p) for p in SOURCES},
              **json.loads(Path(amendments_path).read_text(encoding="utf-8"))}
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"frozen": str(out)}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Run the variants on one split")
    runner.add_argument("--eval-set", required=True)
    runner.add_argument("--protocol", required=True)
    runner.add_argument("--split", choices=("dev", "test"), required=True)
    runner.add_argument("--repeats", type=int, default=1)
    runner.add_argument("--variants", default=",".join(VARIANTS))
    runner.add_argument("--reranker", default="http://127.0.0.1:8790")
    runner.add_argument("--gpu-lock", help="Shared lock file for GPU measurements")
    runner.add_argument("--min-probability", type=float, default=0.01, help="Reranker probability threshold")
    runner.add_argument("--link-candidates", type=int, default=6, help="Passages shown to the link model")
    runner.add_argument("--cpu-only", action="store_true", help="Force num_gpu 0 on every model call")
    runner.add_argument("--out", required=True)
    summarizer = commands.add_parser("summarize", help="Summarise a run record")
    summarizer.add_argument("--run", required=True)
    summarizer.add_argument("--out", required=True)
    freezer = commands.add_parser("freeze", help="Record the hashes and settings used by the test run")
    freezer.add_argument("--eval-set", required=True)
    freezer.add_argument("--protocol", required=True)
    freezer.add_argument("--amendments", required=True, help="JSON object with the development changes and settings")
    freezer.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.eval_set, args.protocol, args.split, args.repeats, args.out, args.variants.split(","),
            args.reranker, args.gpu_lock, {"min_probability": args.min_probability,
                                           "link_candidates": args.link_candidates, "cpu_only": args.cpu_only})
    elif args.command == "freeze":
        freeze(args.eval_set, args.protocol, args.amendments, args.out)
    else:
        summarize(args.run, args.out)
