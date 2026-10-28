"""Compare reviewed-only TF-IDF and BGE-M3 + pgvector under the same bounded agent and Qwen3:4b.

Synthetic cases and simulated actors only: a small demonstration, not quality or deployment acceptance.
Questions, expected passages and rules come from a protocol file written before the run.
A failing backend is recorded as a failure; the pgvector run never falls back to TF-IDF.
No model pulls, service changes, remote fallback, or model-unload requests.

What one run does (``measure``):

1. It records the environment: commit, source hashes, versions, machine and model digests.
2. With ``--backend pgvector`` it prepares the index once (moment 1 of RAG): BGE-M3 turns every reviewed
   passage of the two synthetic cases into a vector, stored in pgvector with its case, revision and hash.
3. It asks every protocol question through the same ``AgenticSearch`` controller and Qwen3:4b (moment 2 of
   RAG), records each search, each model step and its duration, and judges the outcome with the rules
   written in the protocol.
4. It ranks the passages again without any language model (retrieval only), so that the search itself can
   be compared with the expected passages.
5. With pgvector it also runs the guard checks (rights, unreviewed evidence, edits, revocation, index reuse)
   and records the indexed rows.

``persistence`` is a separate step run after a restart of the pgvector pod: it checks that the indexed rows
survived and are reused without new embeddings, then deletes the test rows.

Usage::

    python scripts/compare_agentic_retrieval.py --backend lexical --protocol protocol.json --out tfidf.json
    python scripts/compare_agentic_retrieval.py --backend pgvector --protocol protocol.json --out bge.json \\
        --fixture-dir store
    python scripts/compare_agentic_retrieval.py --persistence bge.json --fixture-dir store --out persistence.json

The pgvector steps read LAB_VECTOR_HOST, LAB_VECTOR_PORT, LAB_VECTOR_DATABASE, LAB_VECTOR_USER and
LAB_VECTOR_PASSWORD_FILE. The password file stays outside the repository.
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


# Pinned Ollama model digests: a run fails instead of silently using another model build.
GENERATION_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
EMBEDDING_DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
# The same limits as the historical 27 October run: three searches and 90 seconds per question.
MAX_ATTEMPTS, MAX_SECONDS = 3, 90
# Stop reasons chosen by the model itself, as opposed to protective stops imposed by the controller.
QWEN_STOPS = ("model_abstained", "answer_abstained")
CONTROLLER_STOPS = ("no_reviewed_evidence", "repeated_query", "attempt_limit", "deadline", "model_unavailable",
                    "invalid_model_plan")
# Source files whose SHA-256 is written into every run record.
SOURCES = ("lab/agentic_search.py", "lab/vector_retrieval.py", "lab/local_inference.py", "lab/retrieval.py",
           "scripts/compare_agentic_retrieval.py")
# Errors recorded as a failed trial (by type only) instead of stopping the whole run.
ERRORS = (InferenceUnavailable, ValueError, PermissionError, KeyError, ConflictError, psycopg.Error)


class TimedInference:
    """Times each model call; every other attribute and the request budget stay on the pinned client.

    The wrapper is given to ``AgenticSearch`` and ``VectorIndex`` in place of ``LocalInference``. It only
    measures: the requests, the answers and the pinned model checks are those of the wrapped client.
    """

    def __init__(self, client):
        """Wrap ``client`` (a ``LocalInference``) and start an empty list of timed calls.

        Args:
            client: the pinned local inference client that really performs every call.
        """
        self._client, self.calls = client, []

    def __getattr__(self, name):
        """Forward every attribute that is not timed here, for example ``request_budget``, to the client.

        Args:
            name: the attribute name looked up on the wrapper.

        Returns:
            The client's attribute of the same name.
        """
        return getattr(self._client, name)

    def _timed(self, phase, method, *args, count=None):
        """Call ``method(*args)`` and append its duration to ``calls``, even when the call fails.

        Args:
            phase: the label stored with the duration ("embed", "plan" or "generate").
            method: the client method to call.
            *args: the arguments passed unchanged to ``method``.
            count: for embeddings, the number of texts sent in this call.

        Returns:
            What ``method`` returns; an exception raised by ``method`` is raised again unchanged.
        """
        started = perf_counter()
        try:
            return method(*args)
        finally:
            step = {"phase": phase, "ms": round((perf_counter() - started) * 1000, 1)}
            if count is not None:
                step["texts"] = count
            self.calls.append(step)

    def embed(self, texts):
        """Embed ``texts`` with BGE-M3 through the client and record the call as an "embed" step.

        Args:
            texts: the passages or the question to turn into vectors.

        Returns:
            One vector per text, as returned by the client.
        """
        return self._timed("embed", self._client.embed, texts, count=len(texts))

    def plan_search(self, *args):
        """Ask Qwen3:4b for the next search action and record the call as a "plan" step.

        Args:
            *args: the planner arguments of ``LocalInference.plan_search``, passed unchanged.

        Returns:
            The planner's decision, as returned by the client.
        """
        return self._timed("plan", self._client.plan_search, *args)

    def generate(self, *args):
        """Ask Qwen3:4b for the cited answer and record the call as a "generate" step.

        Args:
            *args: the generator arguments of ``LocalInference.generate``, passed unchanged.

        Returns:
            The generated answer, as returned by the client.
        """
        return self._timed("generate", self._client.generate, *args)


class TimedRetriever:
    """Records every search with its ranked passages; it never changes what the retriever returns."""

    def __init__(self, retriever, inference=None, after=None):
        """Wrap ``retriever`` and prepare the lists of timed calls and recorded searches.

        Args:
            retriever: ``ReviewedLexicalRetriever`` (TF-IDF) or ``VectorIndex`` (BGE-M3 + pgvector).
            inference: the shared inference client, exposed as ``self.inference`` like on ``VectorIndex``,
                so that ``AgenticSearch`` does not open a second request budget for the same client.
            after: an optional callback run after each search; the revocation guard uses it to withdraw
                access between the search and the next model step.
        """
        self._retriever, self.inference, self._after = retriever, inference, after
        self.calls, self.searches = [], []

    def retrieve(self, store, actor, case_id, question):
        """Run one search through the wrapped retriever, then record its duration and its passages.

        Args:
            store: the authoritative ``StatementStore`` that holds the cases.
            actor: the simulated user on whose behalf the search runs.
            case_id: the only case this search may read.
            question: the original question or a query rewritten by the planner.

        Returns:
            The retriever's result, unchanged.
        """
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
    """Return the SHA-256 hex digest of the file at ``path``, read as raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    """Describe where and with what the run happens, for the run record.

    Returns:
        A dictionary with the git commit and uncommitted files, the SHA-256 of every file in ``SOURCES``,
        the Python, operating-system, Ollama, httpx, psycopg and scikit-learn versions, the processor and
        memory (on Windows), and the CPU-only policy.
    """
    import httpx
    import sklearn

    def run(*command):
        """Run a local command in the repository and return its trimmed output, or None if it cannot start."""
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
    """Return the PostgreSQL connection settings of the pgvector database.

    Returns:
        The psycopg options (host, port, dbname, user, password). The values come from the LAB_VECTOR_*
        environment variables, and the password is read from the file named by LAB_VECTOR_PASSWORD_FILE.
    """
    # Same private runtime settings as lab.rag.configured_rag; the password stays in a file outside the repo.
    return {"host": os.environ.get("LAB_VECTOR_HOST", "127.0.0.1"),
            "port": int(os.environ.get("LAB_VECTOR_PORT", "19543")),
            "dbname": os.environ.get("LAB_VECTOR_DATABASE", "lab_vectors"),
            "user": os.environ.get("LAB_VECTOR_USER", "vector_client"),
            "password": Path(os.environ["LAB_VECTOR_PASSWORD_FILE"]).read_text().strip()}


def cpu_budget(inference, seconds=180):
    """Open a CPU-only request budget for model calls made outside ``AgenticSearch``.

    Args:
        inference: the inference client (or its ``TimedInference`` wrapper).
        seconds: how long the calls inside the budget may take in total.

    Returns:
        The context manager of ``LocalInference.request_budget`` with ``num_gpu 0`` forced.
    """
    return inference.request_budget(monotonic() + seconds, cpu_only=True)


def review_all(store, case):
    """Mark every segment of ``case`` as reviewed by its owner, so that it becomes searchable evidence.

    Args:
        store: the ``StatementStore`` that holds the case.
        case: the case to review; each segment keeps its text unchanged.

    Returns:
        The case after the last update, with its new revision.
    """
    for segment in list(case["segments"]):
        case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
    return case


def new_case(store, title, texts, reviewed=True):
    """Create a synthetic case owned by officer-a, with one transcript segment per text.

    Args:
        store: the ``StatementStore`` in which the case is created.
        title: the case title.
        texts: the segment texts, in order.
        reviewed: when True, every segment is reviewed so that it can be indexed and cited.

    Returns:
        The created case, reviewed or not.
    """
    case = store.create_case("officer-a", title)
    case = store.save_transcript("officer-a", case["id"], {"engine": "synthetic-fixture", "language": "en", "segments": [
        {"start": 5.0 * i, "end": 5.0 * i + 5, "speaker": "Unverified speaker", "text": text}
        for i, text in enumerate(texts)]})
    return review_all(store, case) if reviewed else case


def lexical_ranking(case, question):
    """All passages scored exactly as ReviewedLexicalRetriever scores them, with the ones it keeps.

    Args:
        case: the case whose reviewed passages are ranked.
        question: the question to compare with every passage.

    Returns:
        One entry per passage, best first: rank, TF-IDF cosine score, whether the retriever keeps it
        (score of at least 0.10, at least one shared word, top three) and the passage text.
    """
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
    """All indexed passages by cosine similarity; VectorIndex keeps the three nearest at or above 0.45.

    Args:
        index: the ``VectorIndex`` whose rows are queried.
        inference: the client used to embed the question with BGE-M3, on CPU.
        case: the case whose current revision is ranked.
        question: the question to compare with every indexed passage.

    Returns:
        One entry per indexed passage, nearest first: rank, similarity (1 - cosine distance), whether the
        retriever keeps it, and the passage text.
    """
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
    """Report where each expected passage of the protocol landed in a retrieval-only ranking.

    Args:
        ranking: the output of ``lexical_ranking`` or ``vector_ranking``.
        expected: the passage texts that the protocol expects for this question.

    Returns:
        One entry per expected passage with its rank, its score and whether the retriever keeps it; rank
        and score are None when the passage is not in the ranking.
    """
    found = {item["text"]: item for item in ranking}
    return [{"text": text, "rank": found[text]["rank"] if text in found else None,
             "score": found[text]["score"] if text in found else None,
             "kept": bool(found.get(text, {}).get("kept"))} for text in expected]


def quotes_checked(result):
    """Check that every claim cites an existing source number and that its quote appears in that source.

    An exact quote proves the citation only. It does not prove that the passage answers the question.

    Args:
        result: the answer returned by ``AgenticSearch.answer``.

    Returns:
        True when every claim passes, including when there is no claim.
    """
    sources = result["sources"]
    return all(type(c["source"]) is int and 1 <= c["source"] <= len(sources) and c["quote"] in sources[c["source"]-1]["text"]
               for c in result["claims"])


def stop_kind(reason):
    """Say who ended a run: "answered", "chosen_by_qwen", "imposed_by_controller" or "other".

    Args:
        reason: the ``stop_reason`` returned by ``AgenticSearch.answer``.
    """
    return "answered" if reason == "answered" else "chosen_by_qwen" if reason in QWEN_STOPS else (
        "imposed_by_controller" if reason in CONTROLLER_STOPS else "other")


def judge(spec, outcome, revision):
    """Historical rule unchanged; answer rule separates answer success from the loop requirement.

    The rules come from the protocol written before the run:

    - historical rule: the 27 October criteria, including more than one search for the paraphrase;
    - answer rule: the same answer criteria, with any number of searches from one to three;
    - loop rule (two-step case only): the expected place after more than one search.

    Every rule also requires valid quotes, the same case revision and at most three searches.

    Args:
        spec: the protocol entry of the question (expectation and expected answer word).
        outcome: the trial record; ``outcome["result"]`` is missing when the trial failed with an error.
        revision: the case revision after the run, to detect a change during the run.

    Returns:
        The verdict of each rule that applies to this question, as booleans.
    """
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
    """Ask one protocol question through the agent and record everything needed to judge it.

    Args:
        agent: the ``AgenticSearch`` controller under test.
        inference: the ``TimedInference`` wrapper shared by the agent.
        retriever: the ``TimedRetriever`` wrapper shared by the agent.
        store: the ``StatementStore`` that holds the case.
        case: the case the question is about.
        spec: the protocol entry of the question.

    Returns:
        The trial record: question, expectation, the agent's result (or the error type), the timed model
        steps, every search with its passages, who stopped the run, and the verdicts of ``judge``.
    """
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
    """Check the rights, review, edit, revocation and index-reuse guards on the pgvector backend.

    Five checks run in order:

    1. officer-b asks about officer-a's case: access is refused before any model call;
    2. an unreviewed case gives no passage and no embedding call;
    3. after an edit, only the new revision stays indexed and the old text is no longer returned;
    4. access revoked right after a search stops the agent before any later model step;
    5. synchronising an unchanged case makes no embedding call.

    Args:
        index: the ``VectorIndex`` under test.
        inference: the ``TimedInference`` wrapper, used to count model calls.
        store: the ``StatementStore`` that holds the cases.
        demo: the reviewed demonstration case owned by officer-a.
        demo_texts: the demonstration texts, reused to create the extra guard cases.

    Returns:
        The list of guard results and the ids of the extra cases created for the guards.
    """
    guards, question = [], "Where was the blue bicycle left?"

    def record(name, passed, **details):
        """Store and print one guard result with its supporting details."""
        guards.append({"id": name, "passed": bool(passed), **details})
        print(json.dumps({"guard": name, "passed": bool(passed)}), flush=True)

    # 1. Another user's case: AgenticSearch must refuse before any embedding or model call.
    agent = AgenticSearch(TimedRetriever(index, inference), inference, max_attempts=MAX_ATTEMPTS,
                          max_seconds=MAX_SECONDS, cpu_only=True)
    inference.calls = []
    try:
        agent.answer(store, "officer-b", demo["id"], question)
        record("forbidden_case_refused_before_models", False, detail="an answer was returned")
    except PermissionError:
        record("forbidden_case_refused_before_models", not inference.calls, model_calls=len(inference.calls))

    # 2. Unreviewed evidence: nothing is embedded and nothing is returned.
    unreviewed = new_case(store, "Unreviewed synthetic statement", demo_texts, reviewed=False)
    inference.calls = []
    with cpu_budget(inference):
        retrieval = index.retrieve(store, "officer-a", unreviewed["id"], question)
    record("unreviewed_evidence_never_embedded_or_returned", retrieval["abstained"] and not retrieval["sources"]
           and not inference.calls, model_calls=len(inference.calls))

    # 3. Edited source: the stale vector of the old revision disappears from the index.
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

    # 4. Revocation: the callback withdraws access after the first search; no plan or answer may follow.
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

    # 5. Index reuse: an unchanged case is synchronised again without any embedding call.
    inference.calls = []
    started = perf_counter()
    with cpu_budget(inference):
        index.synchronize(demo, reviewed_sources(demo))
    record("unchanged_index_reused_without_embedding", not inference.calls,
           elapsed_ms=round((perf_counter() - started) * 1000, 1), model_calls=len(inference.calls))
    return guards, [unreviewed["id"], edited["id"]]


def indexed_rows(index, case_ids):
    """List the pgvector rows of the given cases, without the vectors themselves.

    Args:
        index: the ``VectorIndex`` whose database is read.
        case_ids: the cases to list.

    Returns:
        One entry per row: case id, revision, embedding-model digest, source id, content hash and vector
        dimension.
    """
    with index.connect() as db:
        rows = db.execute("""SELECT case_id, revision, model_digest, source_id, content_hash, vector_dims(embedding)
            FROM retrieval.chunks WHERE case_id = ANY(%s) ORDER BY case_id, source_id""", (case_ids,)).fetchall()
    return [{"case_id": r[0], "revision": r[1], "embedding_digest": r[2], "source_id": r[3], "content_hash": r[4],
             "dimensions": r[5]} for r in rows]


def measure(backend, protocol_path, out, fixture_dir=None):
    """Run the whole comparison for one backend and write the run record (see the module docstring).

    Args:
        backend: "lexical" for the historical TF-IDF retriever, or "pgvector" for BGE-M3 + pgvector.
        protocol_path: the protocol JSON written before the run.
        out: a new JSON file for the run record; an existing file is never overwritten.
        fixture_dir: where to keep the synthetic case store, needed later by ``persistence``; a
            temporary directory is used when it is omitted.

    Returns:
        The run record, also written to ``out`` after every trial and at the end.

    Raises:
        ValueError: when ``out`` already exists.
    """
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
        # Moment 2 of RAG: the same agent and Qwen3:4b answer every question; only the retriever differs.
        retriever = TimedRetriever(index, inference) if index else TimedRetriever(ReviewedLexicalRetriever())
        agent = AgenticSearch(retriever, inference, max_attempts=MAX_ATTEMPTS, max_seconds=MAX_SECONDS, cpu_only=True)
        report["trials"] = []
        for spec, case in [(spec, demo) for spec in protocol["trials"]] + [(loop_spec, loop_case)]:
            report["trials"].append(run_trial(agent, inference, retriever, store, case, spec))
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        # Retrieval only: the same questions ranked without any language model.
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
    """After a restart of the pgvector pod: the rows are still there and an unchanged case is not re-embedded.

    For each of the two trial cases, the rows of the current revision are compared with the hashes of the
    reviewed passages, then the case is synchronised again while embedding calls are counted. All test rows
    of the run are deleted at the end.

    Args:
        run_path: the pgvector run record written by ``measure``.
        fixture_dir: the synthetic case store kept by that run (``--fixture-dir``).
        out: a new JSON file for the persistence record; an existing file is never overwritten.

    Returns:
        The persistence record: database start time, one check per case, and the clean-up result.

    Raises:
        ValueError: when ``out`` already exists.
    """
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
        # The start time proves that the database restarted after the run.
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
