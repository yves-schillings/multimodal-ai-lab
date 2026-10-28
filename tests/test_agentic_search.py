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
    """Create an isolated seeded case whose transcript segments are explicitly reviewed.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        Tuple (StatementStore, latest case dictionary) with reviewed source revisions.

    Notes:
        Writes only the pytest temporary SQLite store.
    """
    store = StatementStore(tmp_path)
    case = store.get_case("officer-a", "demo-case-a")
    for segment in list(case["segments"]):
        case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
    return store, case


class FakeInference:
    """Scripted planner/generator with call logs; tests control flow, not LLM quality.
    """
    generation_model, generation_digest = "qwen3:4b", DIGEST

    def __init__(self, plans):
        """Prepare deterministic planner outputs and empty call logs.

        Args:
            plans: Ordered scripted action dictionaries; no language model is invoked.

        Returns:
            None; stores an iterator and mutable planned/generated call lists.
        """
        self.plans = iter(plans)
        self.planned, self.generated = [], []

    def plan_search(self, question, query, sources, attempt, max_attempts):
        """Record a planning call and return the next scripted action.

        Args:
            question: Original question retained by the controller.
            query: Current search query for this step.
            sources: Selected reviewed-source dictionaries.
            attempt: One-based attempt number.
            max_attempts: Configured search cap.

        Returns:
            Next plan dictionary from the supplied iterator.

        Raises:
            StopIteration: The test did not supply enough scripted plans.
        """
        self.planned.append((question, query, sources, attempt, max_attempts))
        return next(self.plans)

    def generate(self, question, sources):
        """Return a deterministic cited library claim without running a model.

        Args:
            question: Original question recorded for assertions.
            sources: Reviewed sources searched for the known fixture quote.

        Returns:
            One claim citing the matching library source, or source one if not found.

        Notes:
            Appends the call to generated. The test must supply a suitable source;
            this fake is not a semantic answer generator.
        """
        self.generated.append((question, sources))
        number = next((i + 1 for i, source in enumerate(sources)
                       if "blue bicycle beside the library" in source["text"]), 1)
        return [{"text": "The statement says the bicycle was left beside the library.",
                 "source": number, "quote": "blue bicycle beside the library"}]


def client():
    """Build a pinned loopback LocalInference client for request-mocking tests.

    Returns:
        Client using placeholder 64-character digests; construction makes no HTTP call.
    """
    return LocalInference("http://127.0.0.1:11434", "bge-m3:latest", DIGEST, "qwen3:4b", DIGEST)


def test_forbidden_case_never_retrieves_or_calls_model(tmp_path):
    """Prove a forbidden actor is rejected before retrieval and model calls.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    class NoRetrieval:
        """Sentinel retriever that must never be reached by a forbidden actor.
        """
        def retrieve(self, *args):
            """Fail immediately if a denied request ever reaches the retriever.

            Args:
                *args: Unused store, actor, case and question arguments.

            Raises:
                pytest.fail.Exception: Any invocation violates the access-before-retrieval invariant.
            """
            pytest.fail("Denied user reached retrieval")
    inference = FakeInference([])
    with pytest.raises(PermissionError):
        AgenticSearch(NoRetrieval(), inference).answer(StatementStore(tmp_path), "officer-b", "demo-case-a", "Where?")
    assert not inference.planned and not inference.generated


def test_iterative_search_preserves_original_question_identity_and_case(tmp_path):
    """Verify reformulation preserves actor, case, original question and bounded trace.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    source = reviewed_sources(case)[0]
    calls = []

    class Retriever:
        """Two-step scripted retriever for checking unchanged case and actor identity.
        """
        def retrieve(self, store, actor, case_id, query):
            """Simulate an initial miss followed by one reviewed-source hit.

            Args:
                store: Unused isolated store argument.
                actor: Fixed actor captured for comparison across attempts.
                case_id: Fixed case ID captured for comparison across attempts.
                query: Original or reformulated query captured in order.

            Returns:
                Case/revision-scoped retrieval dictionary; sources are empty on the first call.

            Notes:
                Appends to the enclosing calls list; no search index or model is accessed.
            """
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
    """Exercise repeated-query, attempt-limit and model-abstention stop paths.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.
        max_attempts: Parametrized search limit used for this stop-rule example.
        plans: Ordered scripted action dictionaries; no language model is invoked.
        reason: Expected controller stop_reason for this parametrized example.
        count: Expected number of recorded search attempts.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    inference = FakeInference(plans)
    result = AgenticSearch(ReviewedLexicalRetriever(), inference, max_attempts=max_attempts).answer(
        store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["abstained"] and result["stop_reason"] == reason and result["attempt_count"] == count
    assert not inference.generated


@pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"max_attempts": 4}, {"max_attempts": True},
                                    {"max_seconds": 0}, {"max_seconds": float("inf")}, {"max_seconds": True}])
def test_invalid_limits_rejected(kwargs):
    """Reject booleans, non-finite values and out-of-range controller limits.

    Args:
        kwargs: Deliberately invalid constructor options, or captured fake-client options.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    with pytest.raises(ValueError):
        AgenticSearch(ReviewedLexicalRetriever(), FakeInference([]), **kwargs)


@pytest.mark.parametrize("stage", ["plan", "generate"])
def test_case_change_during_either_model_withholds_result(tmp_path, stage):
    """Withhold output when the case revision changes during either model phase.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.
        stage: Whether a source edit is injected during planning or generation.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    inference = FakeInference([{"action": "answer", "query": ""}])
    original = inference.plan_search if stage == "plan" else inference.generate

    def edit(*args):
        """Run the scripted phase and then change a reviewed source revision.

        Args:
            *args: Arguments forwarded unchanged to the original scripted phase.

        Returns:
            Original scripted phase result; the following checkpoint must reject it.

        Notes:
            Writes a corrected segment to the temporary test store.
        """
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
    """Detect forged sources and direct text tampering even without a revision increment.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    source = reviewed_sources(case)[0]
    source["text"] = "Forged material outside the current reviewed source."

    class Forged:
        """Adversarial retriever returning a modified authoritative-source fixture.
        """
        def retrieve(self, *args):
            """Return a source whose text was forged outside the authoritative store.

            Args:
                *args: Unused retrieval arguments.

            Returns:
                Deliberately inconsistent source/revision dictionary.
            """
            return {"case_revision": case["revision"], "sources": [source]}

    inference = FakeInference([])
    with pytest.raises(ConflictError):
        AgenticSearch(Forged(), inference).answer(store, "officer-a", case["id"], "Where?")
    assert not inference.planned

    inference = FakeInference([{"action": "answer", "query": ""}])
    original = inference.plan_search

    def mutate_without_revision(*args):
        """Tamper with SQLite source text without incrementing the case revision.

        Args:
            *args: Arguments forwarded to the original fake planning method.

        Returns:
            Scripted planning result; hash verification must still reject the source.

        Notes:
            Uses the isolated store connection to exercise the stronger manifest check.
        """
        with store._connection(write=True) as db:
            db.execute("UPDATE segments SET text=? WHERE id=?", ("Tampered source text.", case["segments"][0]["id"]))
        return original(*args)

    inference.plan_search = mutate_without_revision
    with pytest.raises(ConflictError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    assert not inference.generated


def test_actual_session_logout_during_plan_blocks_further_steps(tmp_path):
    """Verify a real local-session logout during planning prevents generation.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    credential_file = tmp_path / "temporary-test-credentials.json"
    password = "temporary-unit-test-password-only"
    credential_file.write_text(json.dumps({"officer-a": {"enabled": True, "password_hash": password_hash(password)}}))
    sessions = LocalSessions(tmp_path, credential_file)
    token, csrf, _ = sessions.login("officer-a", password, "test-only")
    inference = FakeInference([{"action": "answer", "query": ""}])
    original = inference.plan_search

    def logout(*args):
        """Invalidate the local session after the scripted planning call.

        Args:
            *args: Arguments passed through to the original fake planner.

        Returns:
            Scripted plan; subsequent reauthorization must fail.
        """
        result = original(*args)
        sessions.logout(token)
        return result

    inference.plan_search = logout
    with pytest.raises(PermissionError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(
            store, "officer-a", case["id"], "Where was the bicycle?",
            # No-argument callback: re-read the actual isolated session on each checkpoint.
            reauthorize=lambda: sessions.authenticate(token, csrf, mutation=True))
    assert not inference.generated


def test_callback_cannot_change_actor(tmp_path):
    """Reject a reauthorization callback that substitutes another actor.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    inference = FakeInference([])
    with pytest.raises(PermissionError):
        AgenticSearch(ReviewedLexicalRetriever(), inference).answer(
            # Adversarial callback: deliberately substitutes an unauthorized principal.
            StatementStore(tmp_path), "officer-a", "demo-case-a", "Where?", reauthorize=lambda: {"id": "reviewer"})
    assert not inference.planned


def test_lexical_adapter_has_no_synthetic_cards_or_unreviewed_evidence(tmp_path):
    """Restrict the lexical adapter to actual reviewed sources of the authorized case.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
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
    """Convert invalid, extra-field or nonpermitted model actions to explicit abstention.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.
        plan: Deliberately malformed or overprivileged model action object.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    inference = FakeInference([plan])
    result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["abstained"] and result["stop_reason"] == "invalid_model_plan" and not result["sources"]
    assert not inference.generated


def test_model_unavailability_and_deadline_are_visible_abstentions(tmp_path):
    """Preserve distinct stop reasons for unavailable models and exceeded deadlines.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    store, case = reviewed_store(tmp_path)
    for failure, reason in [(InferenceUnavailable("Unavailable"), "model_unavailable"),
                            (InferenceDeadlineExceeded("Timeout"), "deadline")]:
        inference = FakeInference([])

        def fail(*args):
            """Raise the exception selected by the enclosing failure/stop-reason pair.

            Args:
                *args: Unused planning call arguments.

            Raises:
                InferenceUnavailable: Simulated unavailable-model or deadline exception; no network call occurs.
            """
            raise failure

        inference.plan_search = fail
        result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
        assert result["abstained"] and result["stop_reason"] == reason and result["sources"] == []
        assert not inference.generated


def test_model_answer_requires_evidence_and_valid_quote(tmp_path):
    """Block answers with no reviewed evidence or with fabricated supporting quotes.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    inference = FakeInference([{"action": "answer", "query": ""}])
    result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(
        StatementStore(tmp_path), "officer-a", "demo-case-a", "Where?")
    assert result["stop_reason"] == "no_reviewed_evidence" and not inference.generated
    store, case = reviewed_store(tmp_path)
    inference = FakeInference([{"action": "answer", "query": ""}])
    # Generator callback: unused arguments; return a fabricated quote that must be rejected.
    inference.generate = lambda *args: [{"text": "Invented", "source": 1, "quote": "Not in reviewed testimony"}]
    result = AgenticSearch(ReviewedLexicalRetriever(), inference).answer(store, "officer-a", case["id"], "Where was the bicycle?")
    assert result["abstained"] and result["claims"] == [] and result["stop_reason"] == "model_unavailable"


def test_real_client_uses_strict_action_schema(monkeypatch):
    """Inspect the client payload and reject output with unexpected action fields.

    Args:
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    inference, sent = client(), {}
    # Pin-verification callback: bypass external HTTP checks only in this payload-unit test.
    monkeypatch.setattr(inference, "verify", lambda *args: None)

    def request(method, path, payload=None):
        """Capture the outgoing JSON payload and return a valid scripted search plan.

        Args:
            method: HTTP method argument, ignored by this fake.
            path: API path argument, ignored by this fake.
            payload: JSON payload copied into the enclosing sent dictionary.

        Returns:
            Ollama-shaped response containing one permitted retrieve action.
        """
        sent.update(payload)
        return {"response": '{"action":"retrieve","query":"bicycle library"}'}

    monkeypatch.setattr(inference, "_request", request)
    assert inference.plan_search("Original question", "old query", [], 1, 3)["action"] == "retrieve"
    assert sent["format"]["additionalProperties"] is False
    assert sent["think"] is False and sent["options"]["num_predict"] == 160
    assert "ORIGINAL question: Original question" in sent["prompt"]
    # Response callback: unused HTTP arguments; inject an extra forbidden output field.
    monkeypatch.setattr(inference, "_request", lambda *args: {"response": '{"action":"answer","query":"","reasoning":"hidden"}'})
    with pytest.raises(InvalidSearchPlan):
        inference.plan_search("Original question", "old query", [], 1, 3)


def test_http_budget_propagates_cpu_options_and_checks_guard_on_both_sides(monkeypatch):
    """Check HTTP budget clipping, CPU options, authorization callbacks and default restoration.

    Args:
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    from lab import local_inference

    captured, guards, now = [], [], [100.0]

    class Response:
        """Scripted successful HTTP response whose JSON decoding advances a fake clock.
        """
        def raise_for_status(self):
            """Simulate an HTTP-success status without a real response.

            Returns:
                None; HTTP status failures are outside this timing test.
            """
            pass

        def json(self):
            """Advance the fake monotonic clock while decoding the scripted response.

            Returns:
                Ollama-shaped JSON dictionary.

            Notes:
                Advances time by one second so budget checks are observable.
            """
            now[0] += 1
            return {"response": "{}"}

    class HttpClient:
        """Context-manager fake for inspecting HTTP budgets without opening sockets.
        """
        def __init__(self, **kwargs):
            """Validate or capture the timeout configuration of a fake HTTP client.

            Args:
                **kwargs: HTTP client options; includes the deadline-derived timeout.

            Returns:
                None; appends a mutable options record to captured.
            """
            captured.append({"client": kwargs})

        def __enter__(self):
            """Enter the fake HTTP client's context manager.

            Returns:
                This scripted client instance; no socket is opened.
            """
            return self

        def __exit__(self, *args):
            """Leave the fake HTTP client's context without suppressing exceptions.

            Args:
                *args: Exception type, value and traceback supplied by the context manager.

            Returns:
                None, preserving the default exception propagation.
            """
            pass

        def request(self, method, url, **kwargs):
            """Return a scripted response and optionally record request options.

            Args:
                method: HTTP method stored in the captured record.
                url: Request URL stored for assertions only; it is never contacted.
                **kwargs: Request options, including the bounded JSON payload.

            Returns:
                Fake Response object; no HTTP operation occurs.
            """
            captured[-1].update(method=method, url=url, **kwargs)
            return Response()

    monkeypatch.setattr(local_inference.httpx, "Client", HttpClient)
    inference = client()
    payload = {"options": {"num_ctx": 4096, "num_predict": 600}}
    # Guard callback records each checkpoint; clock callback reads the controlled fake time.
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
    """Reject an HTTP response that arrives after the shared monotonic deadline.

    Args:
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    from lab import local_inference

    now = [10.0]

    class Response:
        """Scripted successful HTTP response whose JSON decoding advances a fake clock.
        """
        def raise_for_status(self):
            """Simulate an HTTP-success status without a real response.

            Returns:
                None; HTTP status failures are outside this timing test.
            """
            pass

        def json(self):
            """Advance the fake monotonic clock while decoding the scripted response.

            Returns:
                Ollama-shaped JSON dictionary.

            Notes:
                Sets time after the deadline to exercise late-output rejection.
            """
            now[0] = 12.0
            return {"response": "late answer"}

    class HttpClient:
        """Context-manager fake for inspecting HTTP budgets without opening sockets.
        """
        def __init__(self, **kwargs):
            """Validate or capture the timeout configuration of a fake HTTP client.

            Args:
                **kwargs: HTTP client options; includes the deadline-derived timeout.

            Returns:
                None; asserts the one-second remaining timeout.
            """
            assert kwargs["timeout"] == 1

        def __enter__(self):
            """Enter the fake HTTP client's context manager.

            Returns:
                This scripted client instance; no socket is opened.
            """
            return self

        def __exit__(self, *args):
            """Leave the fake HTTP client's context without suppressing exceptions.

            Args:
                *args: Exception type, value and traceback supplied by the context manager.

            Returns:
                None, preserving the default exception propagation.
            """
            pass

        def request(self, *args, **kwargs):
            """Return a scripted response and optionally record request options.

            Args:
                *args: Unused request positional arguments.
                **kwargs: Request keyword options.

            Returns:
                Fake Response object; no HTTP operation occurs.
            """
            return Response()

    monkeypatch.setattr(local_inference.httpx, "Client", HttpClient)
    inference = client()
    # Clock callback reads the mutable test time advanced by Response.json.
    with inference.request_budget(11, clock=lambda: now[0]):
        with pytest.raises(InferenceDeadlineExceeded):
            inference._request("GET", "/api/tags")
