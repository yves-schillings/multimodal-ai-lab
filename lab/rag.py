"""Configure the optional classic BGE-M3/pgvector RAG path.

The default lexical setting returns no LocalRag instance; explicit pgvector
configuration binds a pinned local inference client to a vector index. This
module does not implement the agentic search loop: LocalRag performs one
retrieval and, when passages exist, one generation step. The separate
AgenticSearch controller supplies iterative actions and its own time budget.

Case access, reviewed-source hashes and revision are checked through the store
before/after inference. Verbatim quote checks are delegated to the inference
client; they do not establish semantic support, correct entity joins or truth.
"""
import os
from pathlib import Path

from .local_inference import LocalInference
from .store import ConflictError
from .vector_retrieval import VectorIndex, reviewed_sources, source_hash


def configured_rag():
    """Build the explicitly selected local RAG dependencies from environment settings.

    Returns:
        LocalRag | None: None for LAB_RETRIEVAL_BACKEND=lexical (the default),
        otherwise LocalRag(VectorIndex(...)) for the pgvector setting.
    Raises:
        ValueError: Backend, local endpoint/model pins or numeric port is invalid.
        KeyError/OSError: Required inference settings or the private vector
            password file are absent/unreadable. Gateway token-file errors may
            also propagate from LocalInference construction.
    Reads LAB_INFERENCE_URL, LAB_EMBEDDING_MODEL/DIGEST and optional generation
    model/digest; database host, port, name and user have local defaults. Reads
    the password named by LAB_VECTOR_PASSWORD_FILE but never logs it. Construction
    opens no database/HTTP connection, loads no model and provisions no schema.
    A failing pgvector configuration raises rather than silently choosing lexical.
    """
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
    """One-search generative RAG facade over a trusted VectorIndex dependency.

    The vector index owns passage selection and pre-inference access checking;
    the model client owns answer schema/quote validation. This facade rechecks
    the authoritative case after generation and marks human validation required.
    It provides no iterative planning or independent semantic-fidelity verifier.
    """

    def __init__(self, index):
        """Retain a vector index and its inference client without calling either.

        Args:
            index (VectorIndex-like): Exposes retrieve and an inference client
                with generation identity and generate(question, sources).
        Returns None. No access check, model request or database connection occurs.
        """
        self.index = index

    def answer(self, store, actor, case_id, question):
        """Retrieve once, optionally generate claims, then recheck the case snapshot.

        Args:
            store (StatementStore-like): Permission-checked authoritative case store.
            actor (str): Server-selected actor used for retrieval and final access.
            case_id (str): Fixed case identifier passed to the vector index.
            question (str): Query validated by retrieve, at most 2,000 characters.
        Returns:
            dict: Retrieval metadata plus local_generative_rag mode, model pins,
            human-validation notice, answer and, after generation, claim list.
            Retrieval with no passages returns its existing abstention immediately;
            model abstention sets claims=[] and an explicit unsupported-answer text.
        Raises:
            PermissionError/KeyError: Case access fails or the case is missing.
            ConflictError: Revision or selected-source hashes change during generation.
            ValueError/InferenceUnavailable/psycopg.Error: Delegated retrieval,
                embedding, output parsing or database operations fail.
        Makes real vector/embedding requests through index.retrieve and a generation
        request when sources exist. It writes no answer back to the case store.
        This method does not create AgenticSearch's end-to-end deadline; an outer
        request context may supply one. Exact citations are not entailment or a
        verified join between an owner, object identifier and event record.
        """
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
