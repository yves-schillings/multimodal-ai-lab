"""Case/revision scoped pgvector retrieval, checked against the authoritative case store."""
import hashlib
import json
import math

from .store import ConflictError


def reviewed_sources(case):
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
    return hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def vector_literal(vector):
    if len(vector) != 1024 or any(not math.isfinite(x) for x in vector):
        raise ValueError("Expected a finite 1024-dimensional embedding.")
    return "["+",".join(str(float(x)) for x in vector)+"]"


class VectorIndex:
    def __init__(self, connection_options, inference):
        self.connection_options, self.inference = connection_options, inference

    def connect(self):
        import psycopg
        # Connection options are private runtime configuration; never included in status/evidence.
        return psycopg.connect(**self.connection_options, connect_timeout=5, options="-c statement_timeout=10000")

    def synchronize(self, case, sources):
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
