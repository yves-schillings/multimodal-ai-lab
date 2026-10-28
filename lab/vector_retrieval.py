"""Case/revision-scoped pgvector search tied to an authoritative case store.

Reviewed passage text remains in the case store. PostgreSQL stores source IDs,
content hashes, revisions, embedding-model digests and 1,024-dimensional vectors;
indexed text/identity is never accepted as a source of authority on retrieval.
Ollama supplies document/query embeddings. Local synthetic actors and explicit
caller checks do not constitute production identity or database row security.
"""
import hashlib
import json
import math

from .store import ConflictError


def reviewed_sources(case):
    """Build bounded source dictionaries from reviewed case segments/documents.

    Args:
        case (dict): Authoritative snapshot with segments and optional documents.
    Returns:
        list[dict]: Reviewed transcript segments truncated to 1,500 characters and
        reviewed document text split into non-overlapping 1,500-character chunks,
        with source IDs, titles, case ID and timing/document metadata.
    Raises:
        ValueError: More than 128 reviewed chunks would be returned.
    This in-memory extraction does not authorise the caller or inspect model output.
    Unreviewed items are excluded; transcript truncation can omit later information.
    """
    sources = []
    for position, segment in enumerate(case["segments"]):
        if segment["reviewed"]:
            sources.append({"id": segment["id"], "title": f"Reviewed transcript segment {position+1}",
                            "text": segment["text"][:1500], "start": segment["start"], "end": segment["end"],
                            "case_id": case["id"], "kind": "reviewed_transcript"})
    for document in case.get("documents", []):
        if document["reviewed"]:
            for offset in range(0, len(document["text"]), 1500):
                sources.append({"id": document["id"]+":"+str(offset), "document_id": document["id"],
                                "title": document["name"], "text": document["text"][offset:offset+1500],
                                "case_id": case["id"], "kind": "reviewed_document", "start": None, "end": None})
    if len(sources) > 128:
        raise ValueError("This bounded local retrieval supports at most 128 reviewed chunks per case.")
    return sources


def source_hash(source):
    """Hash one source object's full JSON representation for integrity comparison.

    Args:
        source (dict): JSON-serialisable source metadata and text.
    Returns:
        str: Hexadecimal SHA-256 of sorted-key, Unicode-preserving JSON bytes.
    Raises:
        TypeError: A member cannot be JSON serialised.
    Pure helper: no store, database or model call. Adding a score or other field
    changes this hash, so callers remove retrieval-only metadata before comparison.
    """
    return hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def vector_literal(vector):
    """Serialise one finite 1,024-dimensional vector for a bound SQL parameter.

    Args:
        vector (sequence[float]): Exactly 1,024 finite numeric components.
    Returns:
        str: pgvector literal in square brackets, passed using SQL parameters.
    Raises:
        ValueError: Dimension or finiteness is invalid.
    Does not normalise vectors, query a database or infer semantic relevance.
    """
    if len(vector) != 1024 or any(not math.isfinite(x) for x in vector):
        raise ValueError("Expected a finite 1024-dimensional embedding.")
    return "["+",".join(str(float(x)) for x in vector)+"]"


class VectorIndex:
    """Persist/reuse reviewed-source embeddings and retrieve authorised passages.

    PostgreSQL must already expose the vector extension and retrieval schema.
    Searches use exact cosine-distance ordering, a top-three bound and similarity
    >= 0.45. No ANN index, lexical fallback or semantic-support verifier is created.
    Authorisation belongs to retrieve/the case store; direct synchronise calls
    require a trusted, already-authorised case snapshot from their caller.
    """
    def __init__(self, connection_options, inference):
        """Retain database connection options and the pinned embedding client.

        Args:
            connection_options (dict): psycopg connection arguments, including any
                private runtime password; callers must keep these out of evidence.
            inference (LocalInference-like): Exposes embedding_model/digest and embed.
        Returns None. Performs no connection, inference, schema creation or permission
        check; capabilities are exercised by subsequent methods.
        """
        self.connection_options, self.inference = connection_options, inference

    def connect(self):
        """Open a psycopg connection with bounded connection/query timeouts.

        Returns:
            psycopg.Connection: Caller owns context-manager commit/rollback and close.
        Raises:
            psycopg.Error: Connection/authentication/query setup fails.
        Uses stored private options with a five-second connection timeout and a
        10,000-millisecond statement timeout. It makes a database connection but does
        not check user/case rights, create a schema or return secrets in metadata.
        """
        import psycopg
        # Connection options are private runtime configuration; never included in status/evidence.
        return psycopg.connect(**self.connection_options, connect_timeout=5, options="-c statement_timeout=10000")

    def synchronize(self, case, sources):
        """Reuse an unchanged manifest or atomically replace this model's case rows.

        Args:
            case (dict): Trusted, authorised case snapshot with ID and revision.
            sources (list[dict]): Reviewed sources extracted from that same snapshot.
        Returns:
            None: Stored source/hash manifest matches or replacement completes.
        Raises:
            psycopg.Error: Connection, schema, transaction or SQL operation fails.
            InferenceUnavailable/InferenceDeadlineExceeded: Embedding client fails.
            ValueError: An embedding does not satisfy vector_literal's contract.
        Reads/creates retrieval.chunks, compares ID/hash manifests for this revision
        and model digest, and skips embedding when identical. Otherwise embeds all
        sources outside the write transaction, then takes a per-case advisory lock,
        deletes this model's old rows and inserts the replacement atomically. An
        already-matching empty manifest returns early; older-revision rows can stay
        stored, although retrieval excludes them. This method trusts caller access
        and review filtering; it is not a user-facing authorisation entry point. Case
        changes are checked by retrieve/the controller, not prevented by this method.
        """
        manifest = {s["id"]: source_hash(s) for s in sources}
        model = self.inference.embedding_digest
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS retrieval.chunks (
                case_id text NOT NULL, revision integer NOT NULL, model_digest text NOT NULL,
                source_id text NOT NULL, content_hash text NOT NULL, embedding vector(1024) NOT NULL,
                PRIMARY KEY (case_id, model_digest, source_id))""")
            rows = db.execute("SELECT source_id,content_hash FROM retrieval.chunks WHERE case_id=%s AND revision=%s AND model_digest=%s",
                              (case["id"], case["revision"], model)).fetchall()
        if dict(rows) == manifest:
            return
        vectors = self.inference.embed([s["text"] for s in sources]) if sources else []
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (case["id"],))
            db.execute("DELETE FROM retrieval.chunks WHERE case_id=%s AND model_digest=%s", (case["id"], model))
            for source, vector in zip(sources, vectors):
                db.execute("INSERT INTO retrieval.chunks VALUES (%s,%s,%s,%s,%s,%s::vector)",
                           (case["id"], case["revision"], model, source["id"], manifest[source["id"]], vector_literal(vector)))

    def retrieve(self, store, actor, case_id, question):
        # Deny access before embedding any case data or querying its index.
        """Search one authorised case and return sources reloaded from its store.

        Args:
            store (StatementStore-like): Authoritative permission/revision source.
            actor (str): Server-selected actor passed to each store access.
            case_id (str): Case to search; never supplied by a model action.
            question (str): Nonblank query of at most 2,000 characters.
        Returns:
            dict: Case ID/revision, embedding identity, explicit pgvector backend,
            at most three reviewed source objects with scores, and abstention state.
            The answer field is an extractive excerpt list, not generated reasoning.
        Raises:
            PermissionError/KeyError: Case access fails or the case is missing.
            ValueError: Query, source count or vector contract fails.
            ConflictError: Case revision changes during retrieval.
            psycopg.Error/InferenceUnavailable: Database or embedding operations fail.
        Checks access before synchronisation, embeds the query, ranks only rows of the
        same case/revision/model, and reloads text from current authoritative sources
        whose hashes match. Similarity >= 0.45 is a relevance filter, not evidence of
        entailment, correct entity identity, anonymity or production acceptance.
        """
        case = store.get_case(actor, case_id)
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Provide a question of 1 to 2000 characters.")
        sources = reviewed_sources(case)
        result = {"case_id": case_id, "case_revision": case["revision"], "mode": "vector_extractive",
                  "retrieval_backend": "pgvector_bge_m3_cosine", "embedding_model": self.inference.embedding_model,
                  "embedding_digest": self.inference.embedding_digest, "sources": [], "abstained": True,
                  "simulated_identity": case.get("simulated_identity", True),
                  "answer": "No sufficiently relevant reviewed source was found; I cannot answer from the available evidence."}
        self.synchronize(case, sources)
        if not sources:
            return result
        query = vector_literal(self.inference.embed([question])[0])
        with self.connect() as db:
            rows = db.execute("""SELECT source_id, content_hash, 1-(embedding <=> %s::vector) AS score
                FROM retrieval.chunks WHERE case_id=%s AND revision=%s AND model_digest=%s
                ORDER BY embedding <=> %s::vector, source_id LIMIT 3""",
                (query, case_id, case["revision"], self.inference.embedding_digest, query)).fetchall()
        # Never trust indexed text, identity or revision as the source of authority.
        current = store.get_case(actor, case_id)
        if current["revision"] != case["revision"]:
            raise ConflictError("Case changed during retrieval. Reload before asking again.")
        authoritative = {s["id"]: s for s in reviewed_sources(current)}
        for source_id, digest, score in rows:
            source = authoritative.get(source_id)
            if source and source_hash(source) == digest and math.isfinite(score) and score >= 0.45:
                result["sources"].append({**source, "score": round(score, 6)})
        if result["sources"]:
            result["abstained"] = False
            result["answer"] = "Relevant reviewed excerpts:\n\n"+"\n\n".join(
                f"[{i+1}] {s['text']}" for i, s in enumerate(result["sources"]))
        return result
