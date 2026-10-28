"""Compare reviewed-only TF-IDF and BGE-M3 + pgvector under the same bounded agent and Qwen3:4b.

Synthetic cases and simulated actors only: a small demonstration, not quality or deployment acceptance.
Questions, expected passages and rules come from a protocol file written before the run.
A failing backend is recorded as a failure; the pgvector run never falls back to TF-IDF.
No model pulls, service changes, remote fallback, or model-unload requests.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from lab.agentic_search import AgenticSearch, ReviewedLexicalRetriever  # noqa: E402
from lab.local_inference import InferenceUnavailable, LocalInference  # noqa: E402
from lab.retrieval import TfidfVectorizer, _lexical_cosine, _tokens  # noqa: E402
from lab.store import ConflictError, StatementStore  # noqa: E402
from lab.vector_retrieval import VectorIndex, reviewed_sources, source_hash, vector_literal  # noqa: E402


GENERATION_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
EMBEDDING_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
MAX_ATTEMPTS, MAX_SECONDS = 3, 90
QWEN_STOPS = ("model_abstained", "answer_abstained")
CONTROLLER_STOPS = ("no_reviewed_evidence", "repeated_query", "attempt_limit", "deadline", "model_unavailable",
                    "invalid_model_plan")
SOURCES = ("lab/agentic_search.py", "lab/vector_retrieval.py", "lab/local_inference.py", "lab/retrieval.py",
           "scripts/compare_agentic_retrieval.py")
ERRORS = (InferenceUnavailable, ValueError, PermissionError, KeyError, ConflictError, psycopg.Error)


class TimedInference:
    """Times each model call; every other attribute and the request budget stay on the pinned client."""

    def __init__(self, client):
        self._client, self.calls = client, []

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _timed(self, phase, method, *args, count=None):
        started = perf_counter()
        try:
            return method(*args)
        finally:
            step = {"phase": phase, "ms": round((perf_counter() - started) * 1000, 1)}
            if count is not None:
                step["texts"] = count
            self.calls.append(step)

    def embed(self, texts):
        return self._timed("embed", self._client.embed, texts, count=len(texts))

    def plan_search(self, *args):
        return self._timed("plan", self._client.plan_search, *args)

    def generate(self, *args):
        return self._timed("generate", self._client.generate, *args)


class TimedRetriever:
    """Records every search with its ranked passages; it never changes what the retriever returns."""

    def __init__(self, retriever, inference=None, after=None):
        self._retriever, self.inference, self._after = retriever, inference, after
        self.calls, self.searches = [], []

    def retrieve(self, store, actor, case_id, question):
        started = perf_counter()
        try:
            retrieval = self._retriever.retrieve(store, actor, case_id, question)
        finally:
            self.calls.append({"phase": "retrieve", "ms": round((perf_counter() - started) * 1000, 1)})
        self.searches.append({"query": question, "backend": retrieval.get("retrieval_backend"),
                              "passages": [{"id": s["id"], "score": s.get("score"), "text": s["text"]}
                                           for s in retrieval["sources"]]})
        if self._after:
            self._after()
        return retrieval


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import httpx
    import sklearn

    def run(*command):
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=ROOT).stdout.strip()
        except OSError:
            return None
    machine = None
    if os.name == "nt":
        machine = run("powershell", "-NoProfile", "-Command",
                      "$c=Get-CimInstance Win32_Processor;$m=Get-CimInstance Win32_ComputerSystem;"
                      "'{0} | {1} GB RAM' -f $c.Name.Trim(),[math]::Round($m.TotalPhysicalMemory/1GB)")
    return {"commit": run("git", "rev-parse", "HEAD"), "uncommitted": run("git", "status", "--porcelain").splitlines(),
            "source_sha256": {path: sha256(ROOT / path) for path in SOURCES},
            "python": platform.python_version(), "os": platform.platform(), "machine": machine,
            "logical_cpus": os.cpu_count(),
            "ollama": httpx.get("http://127.0.0.1:11434/api/version", timeout=10, trust_env=False).json().get("version"),
            "httpx": httpx.__version__, "psycopg": psycopg.__version__, "scikit_learn": sklearn.__version__,
            "gpu_policy": "num_gpu 0 forced on every embedding and generation call (CPU only)"}


def vector_options():
    # Same private runtime settings as lab.rag.configured_rag; the password stays in a file outside the repo.
    return {"host": os.environ.get("LAB_VECTOR_HOST", "127.0.0.1"),
            "port": int(os.environ.get("LAB_VECTOR_PORT", "19543")),
            "dbname": os.environ.get("LAB_VECTOR_DATABASE", "lab_vectors"),
            "user": os.environ.get("LAB_VECTOR_USER", "vector_client"),
            "password": Path(os.environ["LAB_VECTOR_PASSWORD_FILE"]).read_text().strip()}


def cpu_budget(inference, seconds=180):
    return inference.request_budget(monotonic() + seconds, cpu_only=True)


def review_all(store, case):
    for segment in list(case["segments"]):
        case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
    return case


def new_case(store, title, texts, reviewed=True):
    case = store.create_case("officer-a", title)
    case = store.save_transcript("officer-a", case["id"], {"engine": "synthetic-fixture", "language": "en", "segments": [
        {"start": 5.0 * i, "end": 5.0 * i + 5, "speaker": "Unverified speaker", "text": text}
        for i, text in enumerate(texts)]})
    return review_all(store, case) if reviewed else case


def lexical_ranking(case, question):
    """All passages scored exactly as ReviewedLexicalRetriever scores them, with the ones it keeps."""
    sources = reviewed_sources(case)
    documents = [source["text"] for source in sources]
    scores = _lexical_cosine(question, documents)
    if sources and TfidfVectorizer is not None:
        try:
            vectorizer = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False)
            matrix = vectorizer.fit_transform(documents)
            scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel().tolist()
        except ValueError:
            pass
    tokens = set(_tokens(question))
    rows = sorted(((score, source) for score, source in zip(scores, sources)), key=lambda r: (-r[0], r[1]["id"]))
    eligible = [source["id"] for score, source in rows
                if math.isfinite(score) and score >= 0.10 and tokens.intersection(_tokens(source["text"]))][:3]
    return [{"rank": rank, "score": round(score, 6), "kept": source["id"] in eligible, "text": source["text"]}
            for rank, (score, source) in enumerate(rows, 1)]


def vector_ranking(index, inference, case, question):
    """All indexed passages by cosine similarity; VectorIndex keeps the three nearest at or above 0.45."""
    with cpu_budget(inference):
        query = vector_literal(inference.embed([question])[0])
    with index.connect() as db:
        rows = db.execute("""SELECT source_id, 1-(embedding <=> %s::vector) FROM retrieval.chunks
            WHERE case_id=%s AND revision=%s AND model_digest=%s ORDER BY embedding <=> %s::vector, source_id""",
                          (query, case["id"], case["revision"], EMBEDDING_DIGEST, query)).fetchall()
    texts = {source["id"]: source["text"] for source in reviewed_sources(case)}
    return [{"rank": rank, "score": round(score, 6), "kept": rank <= 3 and score >= 0.45, "text": texts[source_id]}
            for rank, (source_id, score) in enumerate(rows, 1)]


def expected_summary(ranking, expected):
    found = {item["text"]: item for item in ranking}
    return [{"text": text, "rank": found[text]["rank"] if text in found else None,
             "score": found[text]["score"] if text in found else None,
             "kept": bool(found.get(text, {}).get("kept"))} for text in expected]


def quotes_checked(result):
    sources = result["sources"]
    return all(type(c["source"]) is int and 1 <= c["source"] <= len(sources) and c["quote"] in sources[c["source"]-1]["text"]
               for c in result["claims"])


def stop_kind(reason):
    return "answered" if reason == "answered" else "chosen_by_qwen" if reason in QWEN_STOPS else (
        "imposed_by_controller" if reason in CONTROLLER_STOPS else "other")


def judge(spec, outcome, revision):
    """Historical rule unchanged; answer rule separates answer success from the loop requirement."""
    result = outcome.get("result")
    if result is None:
        return {"historical_rule": False, "answer_rule": False}
    word = spec.get("expected_answer_word")
    named = bool(word) and word in result["answer"].casefold()
    sound = quotes_checked(result) and result["case_revision"] == revision and result["attempt_count"] <= MAX_ATTEMPTS
    if spec["expectation"] == "abstain":
        verdict = sound and result["abstained"] and result["stop_reason"] in QWEN_STOPS + ("no_reviewed_evidence",)
        return {"historical_rule": verdict, "answer_rule": verdict,
                "unsupported_answer_returned": not result["abstained"]}
    answered = sound and not result["abstained"] and named
    if spec["expectation"] == "loop":
        return {"loop_rule": answered and result["attempt_count"] > 1, "answer_named_expected_place": answered}
    return {"historical_rule": answered and (spec["expectation"] != "refinement" or result["attempt_count"] > 1),
            "answer_rule": answered}


def run_trial(agent, inference, retriever, store, case, spec):
    inference.calls = retriever.calls = []
    retriever.searches = []
    outcome = {"id": spec["id"], "question": spec["question"], "expectation": spec["expectation"]}
    try:
        outcome["result"] = agent.answer(store, "officer-a", case["id"], spec["question"])
    except ERRORS as exc:
        # Record the type only: never provider bodies, credentials or private text.
        outcome["error_type"] = type(exc).__name__
    outcome["steps"], outcome["searches"] = list(inference.calls), retriever.searches
    if "result" in outcome:
        outcome["stop_kind"] = stop_kind(outcome["result"]["stop_reason"])
    outcome["verdicts"] = judge(spec, outcome, store.get_case("officer-a", case["id"])["revision"])
    print(json.dumps({"trial": spec["id"], "stop": outcome.get("result", {}).get("stop_reason", outcome.get("error_type")),
                      "attempts": outcome.get("result", {}).get("attempt_count"), **outcome["verdicts"]}), flush=True)
    return outcome


def guard_trials(index, inference, store, demo, demo_texts):
    guards, question = [], "Where was the blue bicycle left?"

    def record(name, passed, **details):
        guards.append({"id": name, "passed": bool(passed), **details})
        print(json.dumps({"guard": name, "passed": bool(passed)}), flush=True)

    agent = AgenticSearch(TimedRetriever(index, inference), inference, max_attempts=MAX_ATTEMPTS,
                          max_seconds=MAX_SECONDS, cpu_only=True)
    inference.calls = []
    try:
        agent.answer(store, "officer-b", demo["id"], question)
        record("forbidden_case_refused_before_models", False, detail="an answer was returned")
    except PermissionError:
        record("forbidden_case_refused_before_models", not inference.calls, model_calls=len(inference.calls))

    unreviewed = new_case(store, "Unreviewed synthetic statement", demo_texts, reviewed=False)
    inference.calls = []
    with cpu_budget(inference):
        retrieval = index.retrieve(store, "officer-a", unreviewed["id"], question)
    record("unreviewed_evidence_never_embedded_or_returned", retrieval["abstained"] and not retrieval["sources"]
           and not inference.calls, model_calls=len(inference.calls))

    edited = new_case(store, "Edited synthetic statement", demo_texts)
    with cpu_budget(inference):
        before = index.retrieve(store, "officer-a", edited["id"], question)
    edited = store.update_segment("officer-a", edited["id"], edited["segments"][0]["id"],
                                  demo_texts[0].replace("library", "station"), edited["revision"])
    with cpu_budget(inference):
        after = index.retrieve(store, "officer-a", edited["id"], question)
    with index.connect() as db:
        revisions = [r[0] for r in db.execute("SELECT DISTINCT revision FROM retrieval.chunks WHERE case_id=%s",
                                               (edited["id"],)).fetchall()]
    record("edited_source_replaces_stale_vectors",
           any("library" in s["text"] for s in before["sources"]) and any("station" in s["text"] for s in after["sources"])
           and all("library" not in s["text"] for s in after["sources"]) and revisions == [edited["revision"]],
           indexed_revisions=revisions, current_revision=edited["revision"])

    state = {"revoked": False}
    revoking = AgenticSearch(TimedRetriever(index, inference, after=lambda: state.update(revoked=True)), inference,
                             max_attempts=MAX_ATTEMPTS, max_seconds=MAX_SECONDS, cpu_only=True)
    inference.calls = []
    try:
        revoking.answer(store, "officer-a", demo["id"], question, reauthorize=lambda: not state["revoked"])
        record("revocation_after_search_stops_the_agent", False, detail="an answer was returned")
    except PermissionError:
        later = [c["phase"] for c in inference.calls if c["phase"] in ("plan", "generate")]
        record("revocation_after_search_stops_the_agent", not later, later_model_steps=later)

    inference.calls = []
    started = perf_counter()
    with cpu_budget(inference):
        index.synchronize(demo, reviewed_sources(demo))
    record("unchanged_index_reused_without_embedding", not inference.calls,
           elapsed_ms=round((perf_counter() - started) * 1000, 1), model_calls=len(inference.calls))
    return guards, [unreviewed["id"], edited["id"]]


def indexed_rows(index, case_ids):
    with index.connect() as db:
        rows = db.execute("""SELECT case_id, revision, model_digest, source_id, content_hash, vector_dims(embedding)
            FROM retrieval.chunks WHERE case_id = ANY(%s) ORDER BY case_id, source_id""", (case_ids,)).fetchall()
    return [{"case_id": r[0], "revision": r[1], "embedding_digest": r[2], "source_id": r[3], "content_hash": r[4],
             "dimensions": r[5]} for r in rows]


def measure(backend, protocol_path, out, fixture_dir=None):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new evidence filename; existing results are preserved.")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    inference = TimedInference(LocalInference("http://127.0.0.1:11434", "bge-m3:latest", EMBEDDING_DIGEST,
                                              "qwen3:4b", GENERATION_DIGEST))
    index = VectorIndex(vector_options(), inference) if backend == "pgvector" else None
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(), "backend": backend,
        "protocol_sha256": sha256(protocol_path), "environment": environment(),
        "generation_model": "qwen3:4b", "generation_digest": GENERATION_DIGEST,
        "retrieval": ("BGE-M3 (bge-m3:latest, 1024 dimensions) through CPU Ollama into private local OKD pgvector; "
                      "cosine distance; the three nearest passages kept when 1 - distance >= 0.45"
                      if index else "reviewed-only TF-IDF cosine fitted at each search; score >= 0.10 and at least "
                      "one shared word; top 3; no embedding call"),
        "options": {"num_gpu": 0, "num_ctx_max": 4096, "num_predict_max": 384, "keep_alive": 0,
                    "max_attempts": MAX_ATTEMPTS, "max_seconds_per_question": MAX_SECONDS},
        "limitations": protocol["limits_stated_in_advance"],
    }
    if index:
        report["embedding_model"], report["embedding_digest"] = "bge-m3:latest", EMBEDDING_DIGEST
        with index.connect() as db:
            report["database"] = {"postgresql": db.execute("SHOW server_version").fetchone()[0],
                                  "pgvector": db.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0],
                                  "preexisting_rows_for_demo_case": db.execute(
                                      "SELECT count(*) FROM retrieval.chunks WHERE case_id='demo-case-a'").fetchone()[0]}
    with TemporaryDirectory(prefix="agentic-comparison-", dir=out.parent) as temporary:
        root = Path(fixture_dir).resolve() if fixture_dir else Path(temporary)
        store = StatementStore(root)
        demo = review_all(store, store.get_case("officer-a", "demo-case-a"))
        demo_texts = [segment["text"] for segment in demo["segments"]]
        loop_spec = protocol["loop_trial"]
        loop_case = new_case(store, "Synthetic statement: recovered bicycle", loop_spec["segments"])
        report["cases"] = {"demo": {"id": demo["id"], "revision": demo["revision"]},
                           "loop": {"id": loop_case["id"], "revision": loop_case["revision"]}}
        if index:
            # Moment 1 of RAG: index each case once before any question, on CPU like the trials.
            report["index_preparation"] = []
            for case in (demo, loop_case):
                inference.calls = []
                started = perf_counter()
                with cpu_budget(inference):
                    index.synchronize(case, reviewed_sources(case))
                report["index_preparation"].append({"case_id": case["id"], "passages": len(reviewed_sources(case)),
                                                    "steps": list(inference.calls),
                                                    "elapsed_ms": round((perf_counter() - started) * 1000, 1)})
        retriever = TimedRetriever(index, inference) if index else TimedRetriever(ReviewedLexicalRetriever())
        agent = AgenticSearch(retriever, inference, max_attempts=MAX_ATTEMPTS, max_seconds=MAX_SECONDS, cpu_only=True)
        report["trials"] = []
        for spec, case in [(spec, demo) for spec in protocol["trials"]] + [(loop_spec, loop_case)]:
            report["trials"].append(run_trial(agent, inference, retriever, store, case, spec))
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["retrieval_only"] = []
        for spec, case in [(spec, demo) for spec in protocol["trials"]] + [(loop_spec, loop_case)]:
            ranking = (vector_ranking(index, inference, case, spec["question"]) if index
                       else lexical_ranking(case, spec["question"]))
            report["retrieval_only"].append({"id": spec["id"], "question": spec["question"], "ranking": ranking,
                                             "expected": expected_summary(ranking, spec["expected_passages"])})
        if index:
            report["guards"], extra = guard_trials(index, inference, store, demo, demo_texts)
            report["indexed_rows"] = indexed_rows(index, [demo["id"], loop_case["id"]])
            report["cleanup_case_ids"] = [demo["id"], loop_case["id"]] + extra
            with index.connect() as db:
                # Guard cases are removed now; the two trial cases stay for the persistence check.
                db.execute("DELETE FROM retrieval.chunks WHERE case_id = ANY(%s)", (extra,))
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"evidence": str(out)}), flush=True)
    return report


def persistence(run_path, fixture_dir, out):
    """After a restart of the pgvector pod: the rows are still there and an unchanged case is not re-embedded."""
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new evidence filename; existing results are preserved.")
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    inference = TimedInference(LocalInference("http://127.0.0.1:11434", "bge-m3:latest", EMBEDDING_DIGEST,
                                              "qwen3:4b", GENERATION_DIGEST))
    index = VectorIndex(vector_options(), inference)
    store = StatementStore(Path(fixture_dir).resolve())
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "run_sha256": sha256(run_path), "checks": []}
    with index.connect() as db:
        report["postgres_started_at"] = str(db.execute("SELECT pg_postmaster_start_time()").fetchone()[0])
    for name in ("demo", "loop"):
        case = store.get_case("officer-a", run["cases"][name]["id"])
        manifest = {source["id"]: source_hash(source) for source in reviewed_sources(case)}
        rows = {row["source_id"]: row["content_hash"] for row in indexed_rows(index, [case["id"]])
                if row["revision"] == case["revision"]}
        inference.calls = []
        with cpu_budget(inference):
            index.synchronize(case, reviewed_sources(case))
        check = {"case_id": case["id"], "rows_match_reviewed_sources": rows == manifest, "rows": len(rows),
                 "embedding_calls_on_resync": len(inference.calls)}
        check["passed"] = check["rows_match_reviewed_sources"] and not inference.calls
        report["checks"].append(check)
        print(json.dumps(check), flush=True)
    with index.connect() as db:
        db.execute("DELETE FROM retrieval.chunks WHERE case_id = ANY(%s)", (run["cleanup_case_ids"],))
        remaining = db.execute("SELECT count(*) FROM retrieval.chunks WHERE case_id = ANY(%s)",
                               (run["cleanup_case_ids"],)).fetchone()[0]
    report["cleanup"] = {"case_ids": run["cleanup_case_ids"], "rows_remaining": remaining}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("lexical", "pgvector"))
    parser.add_argument("--protocol", help="Protocol JSON written before the run")
    parser.add_argument("--out", required=True, help="A new local evidence JSON file outside the source repo")
    parser.add_argument("--fixture-dir", help="Keep the synthetic case store here (needed for --persistence)")
    parser.add_argument("--persistence", metavar="RUN_JSON", help="Check a pgvector run after a pod restart")
    args = parser.parse_args()
    if args.persistence:
        persistence(args.persistence, args.fixture_dir, args.out)
    else:
        measure(args.backend, args.protocol, args.out, args.fixture_dir)
