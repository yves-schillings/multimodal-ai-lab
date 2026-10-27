"""One model-driven, bounded search agent over authorized reviewed case evidence.

The controller owns identity, case, limits and source validation. The model may
only ask for another query, answer, or abstain. Traces record actions, not thoughts.
"""
from contextlib import ExitStack
import math
from time import monotonic

from .local_inference import (InferenceDeadlineExceeded, InferenceUnavailable,
                              InvalidSearchPlan, validate_search_plan)
from .retrieval import TfidfVectorizer, _lexical_cosine, _tokens
from .store import ConflictError
from .vector_retrieval import reviewed_sources, source_hash


def _manifest(case):
    return {source["id"]: source_hash(source) for source in reviewed_sources(case)}


class ReviewedLexicalRetriever:
    """Explicit fallback adapter: reviewed sources only, no vector database or LLM.

    Unlike the older extractive demo, synthetic procedure cards are not added.
    """

    def retrieve(self, store, actor, case_id, question):
        case = store.get_case(actor, case_id)
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Provide a question of 1 to 2000 characters.")
        sources = reviewed_sources(case)
        original = _manifest(case)
        documents = [source["text"] for source in sources]
        scores = _lexical_cosine(question, documents)
        backend = "reviewed_only_lexical_count_cosine"
        if sources and TfidfVectorizer is not None:
            try:
                vectorizer = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False)
                matrix = vectorizer.fit_transform(documents)
                scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel().tolist()
                backend = "reviewed_only_tfidf_cosine"
            except ValueError:
                pass  # An empty vocabulary follows the explicit no-match path.
        tokens = set(_tokens(question))
        matches = [(score, source) for score, source in zip(scores, sources)
                   if math.isfinite(score) and score >= 0.10 and tokens.intersection(_tokens(source["text"]))]
        matches.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        current = store.get_case(actor, case_id)
        if current["revision"] != case["revision"] or _manifest(current) != original:
            raise ConflictError("Case changed during retrieval. Reload before asking again.")
        selected = [{**source, "score": round(score, 6)} for score, source in matches[:3]]
        return {"case_id": case_id, "case_revision": case["revision"],
                "mode": "reviewed_only_lexical_extractive", "retrieval_backend": backend,
                "sources": selected, "abstained": not selected,
                "simulated_identity": case.get("simulated_identity", True),
                "answer": "Reviewed excerpts retrieved." if selected else "No relevant reviewed source found."}


class AgenticSearch:
    def __init__(self, retriever, inference, max_attempts=3, max_seconds=60, cpu_only=False,
                 clock=monotonic):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("Search attempts must be an integer from 1 to 3.")
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                or not math.isfinite(max_seconds) or not 0 < max_seconds <= 300):
            raise ValueError("Search deadline must be greater than zero and at most 300 seconds.")
        self.retriever, self.inference = retriever, inference
        self.max_attempts, self.max_seconds = max_attempts, max_seconds
        self.cpu_only, self.clock = cpu_only, clock

    def answer(self, store, actor, case_id, question, reauthorize=None):
        """Actor and case are server-selected; the model cannot replace either.

        Reauthorize should recheck the existing authenticated session/CPE and raise
        PermissionError on revocation. It may return None, True, or a principal
        dict containing the same actor/id. Store permissions are always checked too.
        """
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Provide a question of 1 to 2000 characters.")
        started = self.clock()
        deadline = started + self.max_seconds

        def authorization():
            if reauthorize is not None:
                principal = reauthorize()
                if principal is False or (isinstance(principal, dict)
                                         and principal.get("actor", principal.get("id", actor)) != actor):
                    raise PermissionError("Case access is no longer authorized.")

        authorization()
        baseline = store.get_case(actor, case_id)
        revision, manifest = baseline["revision"], _manifest(baseline)

        def checkpoint(check_time=True):
            authorization()
            current = store.get_case(actor, case_id)
            if current["revision"] != revision or _manifest(current) != manifest:
                raise ConflictError("Reviewed case evidence changed during agent search. Reload before asking again.")
            if check_time and self.clock() >= deadline:
                raise InferenceDeadlineExceeded("Agent search deadline reached; answer withheld.")
            return current

        result = {
            "case_id": case_id, "case_revision": revision, "mode": "bounded_agentic_rag",
            "generation_model": self.inference.generation_model,
            "generation_digest": self.inference.generation_digest,
            "simulated_identity": baseline.get("simulated_identity", True),
            "requires_human_validation": True, "sources": [], "claims": [], "abstained": True,
            "answer": "The reviewed evidence did not support an answer. Human review is required.",
            "attempts": [], "attempt_count": 0, "max_attempts": self.max_attempts,
            "max_seconds": self.max_seconds, "stop_reason": "not_started",
            "authorization_mode": "session_callback_and_store" if reauthorize else "store_only",
            "notice": ("One bounded local agent. Reviewed case sources only. Citation numbers and exact quotes "
                       "are checked; these checks do not prove factual entailment. Human validation remains required."),
        }
        selected, seen, query = [], set(), question

        def validate_sources(retrieval):
            if retrieval.get("case_id", case_id) != case_id or retrieval.get("case_revision") != revision:
                raise ConflictError("Retriever returned another case or revision; answer withheld.")
            sources = retrieval.get("sources")
            if not isinstance(sources, list) or len(sources) > 3:
                raise ValueError("Retriever must return at most three reviewed sources.")
            validated, ids = [], set()
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("Invalid retrieval source.")
                clean = {key: value for key, value in source.items() if key != "score"}
                source_id = clean.get("id")
                if (not isinstance(source_id, str) or source_id in ids or clean.get("case_id") != case_id
                        or source_id not in manifest or source_hash(clean) != manifest[source_id]):
                    raise ConflictError("Retriever evidence does not match the reviewed case source; answer withheld.")
                ids.add(source_id)
                validated.append(dict(source))
            if sum(len(source["text"]) for source in validated) > 4500:
                raise ValueError("Reviewed evidence exceeds 4500 characters.")
            return validated

        def stop(reason, clear_sources=False):
            # Recheck access even when the budget is exhausted or a model failed.
            # Never return a trace or cached source after logout/revocation/edit.
            checkpoint(check_time=False)
            result["stop_reason"] = reason
            result["attempt_count"] = len(result["attempts"])
            result["elapsed_ms"] = round((self.clock() - started) * 1000, 3)
            if clear_sources:
                result["sources"], result["claims"] = [], []
                result["abstained"] = True
            return result

        try:
            with ExitStack() as stack:
                clients = [self.inference]
                retrieval_client = getattr(self.retriever, "inference", None)
                if retrieval_client is not None and retrieval_client is not self.inference:
                    clients.append(retrieval_client)
                for client in clients:
                    # Real LocalInference propagates the remaining deadline and guards to
                    # every verification, embedding, planning and generation HTTP request.
                    if hasattr(client, "request_budget"):
                        stack.enter_context(client.request_budget(deadline, guard=checkpoint,
                                                                 cpu_only=self.cpu_only, max_predict=384,
                                                                 context_tokens=4096, clock=self.clock))
                for attempt in range(1, self.max_attempts + 1):
                    checkpoint()
                    key = " ".join(query.casefold().split())
                    if key in seen:
                        return stop("repeated_query")
                    seen.add(key)
                    retrieval = self.retriever.retrieve(store, actor, case_id, query)
                    checkpoint()
                    hits = validate_sources(retrieval)
                    result["retrieval_backend"] = retrieval.get("retrieval_backend", "unspecified")
                    for name in ("embedding_model", "embedding_digest"):
                        if name in retrieval:
                            result[name] = retrieval[name]
                    # Bounded evidence window: latest hits first, then prior nonduplicate
                    # hits, at most three chunks. It is not an unbounded conversation cache.
                    hit_ids = {source["id"] for source in hits}
                    selected = (hits + [source for source in selected if source["id"] not in hit_ids])[:3]
                    result["sources"] = selected
                    trace = {"attempt": attempt, "query": query, "case_revision": revision,
                             "retrieved_source_ids": [source["id"] for source in hits],
                             "selected_source_hashes": {source["id"]: manifest[source["id"]] for source in selected}}
                    result["attempts"].append(trace)
                    checkpoint()
                    plan = validate_search_plan(self.inference.plan_search(question, query, selected,
                                                                          attempt, self.max_attempts))
                    checkpoint()
                    trace["action"] = plan["action"]
                    trace["elapsed_ms"] = round((self.clock() - started) * 1000, 3)
                    if plan["action"] == "abstain":
                        return stop("model_abstained")
                    if plan["action"] == "retrieve":
                        proposed = " ".join(plan["query"].casefold().split())
                        if proposed in seen:
                            return stop("repeated_query")
                        if attempt == self.max_attempts:
                            return stop("attempt_limit")
                        query = plan["query"]
                        continue
                    if not selected:
                        return stop("no_reviewed_evidence")
                    checkpoint()
                    # Search terms can evolve; the user's requested answer must not.
                    claims = self.inference.generate(question, selected)
                    checkpoint()
                    if not claims:
                        return stop("answer_abstained")
                    # LocalInference validates exact supporting quotes. Validate again at
                    # this boundary so another client cannot skip the citation contract.
                    checked = []
                    if not isinstance(claims, list) or not 1 <= len(claims) <= 3:
                        raise InferenceUnavailable("Invalid answer claims; answer withheld.")
                    for claim in claims:
                        if not isinstance(claim, dict):
                            raise InferenceUnavailable("Invalid answer claim; answer withheld.")
                        number, text, quote = claim.get("source"), claim.get("text"), claim.get("quote")
                        if (type(number) is not int or not 1 <= number <= len(selected)
                                or not isinstance(text, str) or not 1 <= len(text) <= 500
                                or not isinstance(quote, str) or not 8 <= len(quote) <= 1500
                                or quote not in selected[number-1]["text"]):
                            raise InferenceUnavailable("Invalid supporting quote; answer withheld.")
                        checked.append({"text": text, "source": number, "quote": quote})
                    result.update(claims=checked, abstained=False,
                                  answer="\n".join(f"{claim['text']} [{claim['source']}]" for claim in checked))
                    return stop("answered")
        except InferenceDeadlineExceeded:
            return stop("deadline", clear_sources=True)
        except InvalidSearchPlan:
            return stop("invalid_model_plan", clear_sources=True)
        except InferenceUnavailable:
            return stop("model_unavailable", clear_sources=True)
        return stop("attempt_limit")
