"""Behavioral safety tests. Fake models exercise control flow, not LLM quality."""
import json

import pytest

from lab.agentic_search import AgenticSearch, ReviewedLexicalRetriever
from lab.auth import LocalSessions, password_hash
from lab.local_inference import (InferenceDeadlineExceeded, InferenceUnavailable,
                                 InvalidSearchPlan, LocalInference)
from lab.store import ConflictError, StatementStore
from lab.vector_retrieval import reviewed_sources


DIGEST = "a" * 64


def reviewed_store(tmp_path):
    store = StatementStore(tmp_path)
    case = store.get_case("officer-a", "demo-case-a")
    for segment in list(case["segments"]):
        case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
    return store, case


class FakeInference:
    generation_model, generation_digest = "qwen3:4b", DIGEST

    def __init__(self, plans):
        self.plans = iter(plans)
        self.planned, self.generated = [], []

    def plan_search(self, question, query, sources, attempt, max_attempts):
        self.planned.append((question, query, sources, attempt, max_attempts))
        return next(self.plans)

    def generate(self, question, sources):
        self.generated.append((question, sources))
        number = next((i + 1 for i, source in enumerate(sources)
                       if "blue bicycle beside the library" in source["text"]), 1)
        return [{"text": "The statement says the bicycle was left beside the library.",
                 "source": number, "quote": "blue bicycle beside the library"}]


def client():
    return LocalInference("http://127.0.0.1:11434", "bge-m3:latest", DIGEST, "qwen3:4b", DIGEST)


def test_forbidden_case_never_retrieves_or_calls_model(tmp_path):
    class NoRetrieval:
        def retrieve(self, *args):
            pytest.fail("Denied user reached retrieval")
    inference = FakeInference([])
    with pytest.raises(PermissionError):
        AgenticSearch(NoRetrieval(), inference).answer(StatementStore(tmp_path), "officer-b", "demo-case-a", "Where?")
    assert not inference.planned and not inference.generated


def test_iterative_search_preserves_original_question_identity_and_case(tmp_path):
    store, case = reviewed_store(tmp_path)
    source = reviewed_sources(case)[0]
    calls = []

    class Retriever:
        def retrieve(self, store, actor, case_id, query):
            calls.append((actor, case_id, query))
            return {"case_id": case_id, "case_revision": case["revision"],
                    "sources": [] if len(calls) == 1 else [source], "retrieval_backend": "test_fake"}

    inference = FakeInference([{"action": "retrieve", "query": "bicycle library"}, {"action": "answer", "query": ""}])
    question = "Which place was the cycle parked at?"
    result = AgenticSearch(Retriever(), inference).answer(store, "officer-a", case["id"], question)
    assert [call[:2] for call in calls] == [("officer-a", case["id"])] * 2
    assert calls[0][2] == question and calls[1][2] == "bicycle library"
    assert inference.generated[0][0] == question
    assert result["stop_reason"] == "answered" and not result["abstained"]
    assert result["attempt_count"] == 2 and result["simulated_identity"] is True
    assert [item["action"] for item in result["attempts"]] == ["retrieve", "answer"]
    assert source["id"] in result["attempts"][1]["selected_source_hashes"]
    assert all("reasoning" not in item for item in result["attempts"])


@pytest.mark.parametrize("max_attempts,plans,reason,count", [
    (3, [{"action": "retrieve", "query": "  WHERE was the bicycle?  "}], "repeated_query", 1),
    (2, [{"action": "retrieve", "query": "library bicycle"},
         {"action": "retrieve", "query": "a third new query"}], "attempt_limit", 2),
    (3, [{"action": "abstain", "query": ""}], "model_abstained", 1),
])
def test_repeated_query_attempt_limit_and_abstention(tmp_path, max_attempts, plans, reason, count):
    store, case = reviewed_store(tmp_path)
    inference = FakeInference(plans)
    result = AgenticSearch(ReviewedLexicalRetriever(), inference, max_attempts=max_attempts).answer(
        store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["abstained"] and result["stop_reason"] == reason and result["attempt_count"] == count
    assert not inference.generated


@pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"max_attempts": 4}, {"max_attempts": True},
                                    {"max_seconds": 0}, {"max_seconds": float("inf")}, {"max_seconds": True}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        AgenticSearch(ReviewedLexicalRetriever(), FakeInference([]), **kwargs)


@pytest.mark.parametrize("stage", ["plan", "generate"])
def test_case_change_during_either_model_withholds_result(tmp_path, stage):
    store, case = reviewed_store(tmp_path)
    inference = FakeInference([{"action": "answer", "query": ""}])
    original = inference.plan_search if stage == "plan" else inference.generate

    def edit(*args):
        result = original(*args)
        store.update_segment("officer-a", case["id"], case["segments"][0]["id"],
                             "Corrected reviewed statement.", case["revision"])
        return result

    setattr(inference, "plan_search" if stage == "plan" else "generate", edit)
    with pytest.raises(ConflictError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    if stage == "plan":
        assert not inference.generated


def test_changed_hash_without_revision_and_forged_retrieval_are_rejected(tmp_path):
    store, case = reviewed_store(tmp_path)
    source = reviewed_sources(case)[0]
    source["text"] = "Forged material outside the current reviewed source."

    class Forged:
        def retrieve(self, *args):
            return {"case_revision": case["revision"], "sources": [source]}

    inference = FakeInference([])
    with pytest.raises(ConflictError):
        AgenticSearch(Forged(), inference).answer(store, "officer-a", case["id"], "Where?")
    assert not inference.planned

    inference = FakeInference([{"action": "answer", "query": ""}])
    original = inference.plan_search

    def mutate_without_revision(*args):
        with store._connection(write=True) as db:
            db.execute("UPDATE segments SET text=? WHERE id=?", ("Tampered source text.", case["segments"][0]["id"]))
        return original(*args)

    inference.plan_search = mutate_without_revision
    with pytest.raises(ConflictError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    assert not inference.generated


def test_actual_session_logout_during_plan_blocks_further_steps(tmp_path):
    store, case = reviewed_store(tmp_path)
    credential_file = tmp_path / "temporary-test-credentials.json"
    password = "temporary-unit-test-password-only"
    credential_file.write_text(json.dumps({"officer-a": {"enabled": True, "password_hash": password_hash(password)}}))
    sessions = LocalSessions(tmp_path, credential_file)
    token, csrf, _ = sessions.login("officer-a", password, "test-only")
    inference = FakeInference([{"action": "answer", "query": ""}])
    original = inference.plan_search

    def logout(*args):
        result = original(*args)
        sessions.logout(token)
        return result

    inference.plan_search = logout
    with pytest.raises(PermissionError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(
            store, "officer-a", case["id"], "Where was the bicycle?",
            reauthorize=lambda: sessions.authenticate(token, csrf, mutation=True))
    assert not inference.generated


def test_callback_cannot_change_actor(tmp_path):
    inference = FakeInference([])
    with pytest.raises(PermissionError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(
            StatementStore(tmp_path), "officer-a", "demo-case-a", "Where?", reauthorize=lambda: {"id": "reviewer"})
    assert not inference.planned


def test_lexical_adapter_has_no_synthetic_cards_or_unreviewed_evidence(tmp_path):
    store = StatementStore(tmp_path)
    retriever = ReviewedLexicalRetriever()
    for question in ("Where was the bicycle?", "Who approves a current draft?"):
        assert retriever.retrieve(store, "officer-a", "demo-case-a", question)["sources"] == []
    store, case = reviewed_store(tmp_path)
    result = retriever.retrieve(store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["sources"] and result["retrieval_backend"].startswith("reviewed_only_")
    assert all(source["kind"] == "reviewed_transcript" and source["case_id"] == case["id"] for source in result["sources"])


@pytest.mark.parametrize("plan", [
    {"action": "retrieve", "query": ""}, {"action": "answer", "query": "secret new instructions"},
    {"action": "answer", "query": "", "actor": "reviewer"}, {"action": "shell", "query": "run"},
])
def test_nonpermitted_model_plan_is_withheld(tmp_path, plan):
    store, case = reviewed_store(tmp_path)
    inference = FakeInference([plan])
    result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["abstained"] and result["stop_reason"] == "invalid_model_plan" and not result["sources"]
    assert not inference.generated


def test_model_unavailability_and_deadline_are_visible_abstentions(tmp_path):
    store, case = reviewed_store(tmp_path)
    for failure, reason in [(InferenceUnavailable("Unavailable"), "model_unavailable"),
                            (InferenceDeadlineExceeded("Timeout"), "deadline")]:
        inference = FakeInference([])

        def fail(*args):
            raise failure

        inference.plan_search = fail
        result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
        assert result["abstained"] and result["stop_reason"] == reason and result["sources"] == []
        assert not inference.generated


def test_model_answer_requires_evidence_and_valid_quote(tmp_path):
    inference = FakeInference([{"action": "answer", "query": ""}])
    result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(
        StatementStore(tmp_path), "officer-a", "demo-case-a", "Where?")
    assert result["stop_reason"] == "no_reviewed_evidence" and not inference.generated
    store, case = reviewed_store(tmp_path)
    inference = FakeInference([{"action": "answer", "query": ""}])
    inference.generate = lambda *args: [{"text": "Invented", "source": 1, "quote": "Not in reviewed testimony"}]
    result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["abstained"] and result["claims"] == [] and result["stop_reason"] == "model_unavailable"


def test_real_client_uses_strict_action_schema(monkeypatch):
    inference, sent = client(), {}
    monkeypatch.setattr(inference, "verify", lambda *args: None)

    def request(method, path, payload=None):
        sent.update(payload)
        return {"response": '{"action":"retrieve","query":"bicycle library"}'}

    monkeypatch.setattr(inference, "_request", request)
    assert inference.plan_search("Original question", "old query", [], 1, 3)["action"] == "retrieve"
    assert sent["format"]["additionalProperties"] is False
    assert sent["think"] is False and sent["options"]["num_predict"] == 160
    assert "ORIGINAL question: Original question" in sent["prompt"]
    monkeypatch.setattr(inference, "_request", lambda *args: {"response": '{"action":"answer","query":"","reasoning":"hidden"}'})
    with pytest.raises(InvalidSearchPlan):
        inference.plan_search("Original question", "old query", [], 1, 3)


def test_http_budget_propagates_cpu_options_and_checks_guard_on_both_sides(monkeypatch):
    from lab import local_inference

    captured, guards, now = [], [], [100.0]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            now[0] += 1
            return {"response": "{}"}

    class HttpClient:
        def __init__(self, **kwargs):
            captured.append({"client": kwargs})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method, url, **kwargs):
            captured[-1].update(method=method, url=url, **kwargs)
            return Response()

    monkeypatch.setattr(local_inference.httpx, "Client", HttpClient)
    inference = client()
    payload = {"options": {"num_ctx": 4096, "num_predict": 600}}
    with inference.request_budget(105, guard=lambda: guards.append(True), cpu_only=True,
                                  max_predict=128, context_tokens=2048, clock=lambda: now[0]):
        inference._request("POST", "/api/generate", payload)
        now[0] = 106
        with pytest.raises(InferenceDeadlineExceeded):
            inference._request("GET", "/api/tags")
    assert len(captured) == 1 and len(guards) == 3
    assert captured[0]["client"]["timeout"] == 5
    assert captured[0]["json"]["options"] == {"num_ctx": 2048, "num_predict": 128, "num_gpu": 0}
    assert captured[0]["json"]["keep_alive"] == 0 and "num_gpu" not in payload["options"]
    inference._request("POST", "/api/generate", payload)
    assert captured[1]["client"]["timeout"] == 180
    assert captured[1]["json"] == payload  # Existing non-agent defaults remain unchanged.


def test_late_http_output_is_rejected(monkeypatch):
    from lab import local_inference

    now = [10.0]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            now[0] = 12.0
            return {"response": "late answer"}

    class HttpClient:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(local_inference.httpx, "Client", HttpClient)
    inference = client()
    with inference.request_budget(11, clock=lambda: now[0]):
        with pytest.raises(InferenceDeadlineExceeded):
            inference._request("GET", "/api/tags")
