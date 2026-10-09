"""Bounded local RAG. Source validation and permissions precede and follow inference."""
import os
from pathlib import Path

from .local_inference import LocalInference
from .store import ConflictError
from .vector_retrieval import VectorIndex, reviewed_sources, source_hash


def configured_rag():
    backend = os.environ.get("LAB_RETRIEVAL_BACKEND", "lexical")
    if backend == "lexical":
        return None
    if backend != "pgvector":
        raise ValueError("LAB_RETRIEVAL_BACKEND must be lexical or pgvector.")
    secret_path = Path(os.environ["LAB_VECTOR_PASSWORD_FILE"])
    inference = LocalInference(os.environ["LAB_INFERENCE_URL"], os.environ["LAB_EMBEDDING_MODEL"],
                               os.environ["LAB_EMBEDDING_DIGEST"], os.environ.get("LAB_GENERATION_MODEL"),
                               os.environ.get("LAB_GENERATION_DIGEST"))
    options = {"host": os.environ.get("LAB_VECTOR_HOST", "lab-vector"),
               "port": int(os.environ.get("LAB_VECTOR_PORT", "5432")),
               "dbname": os.environ.get("LAB_VECTOR_DATABASE", "lab_vectors"),
               "user": os.environ.get("LAB_VECTOR_USER", "vector_client"), "password": secret_path.read_text().strip()}
    return LocalRag(VectorIndex(options, inference))


class LocalRag:
    def __init__(self, index):
        self.index = index

    def answer(self, store, actor, case_id, question):
        result = self.index.retrieve(store, actor, case_id, question)
        result.update({"mode": "local_generative_rag", "generation_model": self.index.inference.generation_model,
                       "generation_digest": self.index.inference.generation_digest,
                       "requires_human_validation": True,
                       "notice": "Local model answer with checked citation numbers and exact supporting quotes. Human review remains required; citation checks do not guarantee factual entailment."})
        if result["abstained"]:
            return result
        claims = self.index.inference.generate(question, result["sources"])
        case = store.get_case(actor, case_id)
        current = {s["id"]: source_hash(s) for s in reviewed_sources(case)}
        if case["revision"] != result["case_revision"] or any(
                current.get(s["id"]) != source_hash({k:v for k,v in s.items() if k != "score"}) for s in result["sources"]):
            raise ConflictError("Case changed during generation. Reload before asking again.")
        if not claims:
            result.update({"abstained": True, "answer": "The reviewed sources do not support an answer to this question.", "claims": []})
        else:
            result.update({"answer": "\n".join(f"{c['text']} [{c['source']}]" for c in claims), "claims": claims})
        return result
