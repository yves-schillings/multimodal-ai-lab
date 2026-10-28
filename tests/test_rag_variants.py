"""Tests of the fixed, reranked and identifier-linked RAG controllers and of the benchmark judge.

Scripted fake models replace Ollama and the reranker: these tests check control flow, guards and the
judge rules, not answer quality.
"""
import hashlib
import math

import pytest

from lab.local_inference import InferenceUnavailable
from lab.rag_benchmark import fold, judge_trial, paired_comparison, percentile, wilson_interval
from lab.rag_variants import (LINK_AMBIGUOUS, LINK_NONE, ExactVectorRetriever, FixedSearch,
                              IdentifierLinkedSearch, RerankedRetriever, RerankerClient, check_claims,
                              find_identifiers)
from lab.store import StatementStore

SEGMENTS = [
    "My bicycle is a green city bike. The frame number engraved under the saddle is GX-5530.",
    "I locked it outside the swimming pool on Monday evening and it was gone on Tuesday.",
    "Recovery register entry 4: frame number RB-1120 was recovered at the train station on Tuesday.",
    "Recovery register entry 9: frame number GX-5530 was recovered near the old harbour on Wednesday morning.",
    "Recovery register entry 12: frame number MZ-6604 was recovered beside the cathedral on Thursday.",
]


def _vector(text):
    """Return a deterministic 1,024-dimensional bag-of-words vector for a text (a fake embedding).

    Args:
        text (str): Text to embed.
    Returns:
        list[float]: Word counts hashed into 1,024 buckets, plus a small constant so it is never zero.
    """
    vector = [0.001] * 1024
    for word in text.casefold().replace(".", " ").replace(":", " ").replace("?", " ").split():
        vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 1024] += 1.0
    return vector


class FakeInference:
    """Scripted stand-in for ``LinkingInference``: fake embeddings, scripted link and answer."""
    generation_model, generation_digest = "fake-generator", "0" * 64
    embedding_model, embedding_digest = "fake-embedder", "1" * 64

    def __init__(self, link=LINK_NONE, answer_from=None, quote=None):
        """Prepare the scripted behaviour and empty call logs.

        Args:
            link (str): Value returned by ``link_identifier``.
            answer_from (str | None): A word; the answer quotes the first given passage containing it.
            quote (str | None): A fixed quote to return instead, to test the citation check.
        """
        self.link, self.answer_from, self.quote = link, answer_from, quote
        self.embedded, self.generated_with, self.linked_with = [], [], []

    def embed(self, texts):
        """Record ``texts`` and return one fake vector per text."""
        self.embedded.append(list(texts))
        return [_vector(t) for t in texts]

    def link_identifier(self, question, sources, choices):
        """Record ``choices`` and the ``sources`` texts for ``question``, and return the scripted link."""
        self.linked_with.append((list(choices), [s["text"] for s in sources]))
        return self.link

    def generate(self, question, sources):
        """Record the ``sources`` given for ``question`` and return one claim quoting the passage that
        contains ``answer_from``, a fixed invalid quote, or no claim."""
        self.generated_with.append([s["text"] for s in sources])
        if self.quote is not None:
            return [{"text": "Scripted", "source": 1, "quote": self.quote}]
        for number, source in enumerate(sources, 1):
            if self.answer_from and self.answer_from in source["text"]:
                return [{"text": f"It concerns {self.answer_from}.", "source": number, "quote": source["text"][:60]}]
        return []


class FakeReranker:
    """Scripted stand-in for ``RerankerClient``: passages containing ``favour`` score high."""

    def __init__(self, favour):
        """Keep the favoured word and an empty call log.

        Args:
            favour (str): Passages containing this word receive logit 5, the others logit -5.
        """
        self.favour, self.calls = favour, []

    def score(self, query, passages):
        """Record the call for ``query`` and return one scripted logit per item of ``passages``."""
        self.calls.append(len(passages))
        return [5.0 if self.favour in p else -5.0 for p in passages]


@pytest.fixture
def setup(tmp_path):
    """Create a store in ``tmp_path`` with one reviewed bicycle case owned by officer-a.

    Returns:
        tuple[StatementStore, str]: The store and the case ID.
    """
    store = StatementStore(tmp_path)
    case = store.create_case("officer-a", "Recovered bicycle")
    case = store.save_transcript("officer-a", case["id"], {"engine": "synthetic-fixture", "language": "en", "segments": [
        {"start": 5.0 * i, "end": 5.0 * i + 5, "speaker": "Unverified speaker", "text": t} for i, t in enumerate(SEGMENTS)]})
    for segment in list(case["segments"]):
        case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
    return store, case["id"]


def test_find_identifiers_keeps_object_codes_only():
    """Frame numbers, registrations and long numbers are found; place names, times and roads are not."""
    text = ("GX-5530 near Saint-Pierre at 15:47 on the E40; car 1-KLM-482, serial SN-4471-KQ, "
            "IMEI 356938035643809, item GX-5530 again")
    assert find_identifiers(text) == ["GX-5530", "1-KLM-482", "SN-4471-KQ", "356938035643809"]


def test_exact_retriever_refuses_another_actor_before_embedding(setup):
    """Another officer is refused by the store before any embedding call. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    inference = FakeInference()
    with pytest.raises(PermissionError):
        ExactVectorRetriever(inference).retrieve(store, "officer-b", case_id, "Where was the bicycle recovered?")
    assert inference.embedded == []


def test_exact_retriever_applies_the_vector_index_rule(setup):
    """At most three passages are returned, nearest first, each at or above the similarity threshold. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    retriever = ExactVectorRetriever(FakeInference(), min_score=0.2)
    result = retriever.retrieve(store, "officer-a", case_id, "frame number recovered near the old harbour")
    scores = [s["score"] for s in result["sources"]]
    assert 1 <= len(scores) <= 3 and scores == sorted(scores, reverse=True) and min(scores) >= 0.2
    assert "old harbour" in result["sources"][0]["text"]


def test_exact_retriever_reuses_vectors_of_an_unchanged_case(setup):
    """A second synchronisation of the same revision makes no embedding call. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    inference = FakeInference()
    retriever = ExactVectorRetriever(inference)
    case = store.get_case("officer-a", case_id)
    assert retriever.synchronize(case) is True and retriever.synchronize(case) is False
    assert len(inference.embedded) == 1


def test_fixed_search_answers_with_a_checked_quote(setup):
    """Fixed RAG searches once and returns the generator's claim with a verbatim quote. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    inference = FakeInference(answer_from="old harbour")
    result = FixedSearch(ExactVectorRetriever(inference, min_score=0.0), inference).answer(
        store, "officer-a", case_id, "Where was frame number GX-5530 recovered near the old harbour?")
    assert result["stop_reason"] == "answered" and result["attempt_count"] == 1
    assert result["claims"][0]["quote"] in result["sources"][result["claims"][0]["source"] - 1]["text"]


def test_fixed_search_withholds_an_invented_quote(setup):
    """A quote that is not in the cited passage stops the request with no sources. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    inference = FakeInference(quote="This sentence is not in any passage.")
    result = FixedSearch(ExactVectorRetriever(inference, min_score=0.0), inference).answer(
        store, "officer-a", case_id, "Where was the bicycle recovered?")
    assert result["stop_reason"] == "model_unavailable" and result["sources"] == [] and result["abstained"]


def test_fixed_search_abstains_without_evidence(setup):
    """No passage above the threshold: abstention before any generation call. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    inference = FakeInference(answer_from="harbour")
    result = FixedSearch(ExactVectorRetriever(inference, min_score=0.99), inference).answer(
        store, "officer-a", case_id, "Unrelated question about a painting")
    assert result["stop_reason"] == "no_reviewed_evidence" and inference.generated_with == []


def test_fixed_search_stops_after_revocation(setup):
    """A session revoked after the search stops the request before generation. ``setup`` gives the store and the bicycle case."""
    store, case_id = setup
    inference = FakeInference(answer_from="harbour")
    state = {"calls": 0}

    def reauthorize():
        """Allow the initial check and the check before the search, then revoke access."""
        state["calls"] += 1
        return state["calls"] <= 2
    with pytest.raises(PermissionError):
        FixedSearch(ExactVectorRetriever(inference, min_score=0.0), inference).answer(
            store, "officer-a", case_id, "Where was the bicycle recovered?", reauthorize=reauthorize)
    assert inference.generated_with == []


def test_reranked_retriever_reorders_and_filters():
    """The reranker order wins over the first-stage order, and low probabilities are dropped."""
    class Base:
        """First stage returning three passages in a fixed order."""
        inference = FakeInference()

        def rank(self, store, actor, case_id, question):
            """Return a fake case and three passages, least relevant first; ``store``, ``actor``,
            ``case_id`` and ``question`` are ignored."""
            sources = [{"id": str(i), "text": t, "case_id": "c"} for i, t in enumerate(["a train", "b cathedral", "c harbour"])]
            return {"id": "c", "revision": 1}, [(0.9, sources[0]), (0.8, sources[1]), (0.7, sources[2])]
    result = RerankedRetriever(Base(), FakeReranker("harbour")).retrieve(None, "officer-a", "c", "Where?")
    assert [s["text"] for s in result["sources"]] == ["c harbour"]
    assert result["sources"][0]["score"] == pytest.approx(1 / (1 + math.exp(-5)), abs=1e-6)


def _linked(setup, link, favour="harbour", answer_from="harbour"):
    """Build an identifier-linked controller on the ``setup`` case with scripted models.

    ``link`` is the scripted link answer, ``favour`` the word the fake reranker prefers and ``answer_from``
    the word whose passage the fake generator quotes.

    Returns:
        tuple: store, case ID, controller and fake inference.
    """
    store, case_id = setup
    inference = FakeInference(link=link, answer_from=answer_from)
    base = ExactVectorRetriever(inference)
    controller = IdentifierLinkedSearch(RerankedRetriever(base, FakeReranker(favour)), inference)
    return store, case_id, controller, inference


def test_linked_search_uses_the_identifier_named_in_the_question(setup):
    """An identifier in the question is used without a link call, and its record is retrieved exactly. ``setup`` gives the store and the bicycle case."""
    store, case_id, controller, inference = _linked(setup, LINK_NONE, favour="train")
    result = controller.answer(store, "officer-a", case_id, "Where was frame number GX-5530 recovered?")
    assert inference.linked_with == [] and result["attempts"][0]["link_origin"] == "question"
    seen = inference.generated_with[0]
    assert any("old harbour" in t for t in seen)
    assert all(set(find_identifiers(t)) <= {"GX-5530"} for t in seen)


def test_linked_search_never_shows_another_identifier(setup):
    """With the complainant's identifier linked, register entries of other bicycles never reach the generator. ``setup`` gives the store and the bicycle case."""
    store, case_id, controller, inference = _linked(setup, "GX-5530", favour="recovered")
    result = controller.answer(store, "officer-a", case_id, "Where was the complainant's bicycle recovered?")
    assert result["stop_reason"] == "answered" and "harbour" in result["answer"]
    assert all(set(find_identifiers(t)) <= {"GX-5530"} for t in inference.generated_with[0])


def test_linked_search_withholds_when_the_identifier_has_no_record(setup):
    """A transposed identifier with no reviewed passage stops with identifier_not_found. ``setup`` gives the store and the bicycle case."""
    store, case_id, controller, inference = _linked(setup, LINK_NONE)
    result = controller.answer(store, "officer-a", case_id, "Where was frame number GX-5350 recovered?")
    assert result["stop_reason"] == "identifier_not_found" and result["abstained"] and inference.generated_with == []


def test_linked_search_withholds_an_ambiguous_link(setup):
    """The link model answering AMBIGUOUS stops before generation. ``setup`` gives the store and the bicycle case."""
    store, case_id, controller, inference = _linked(setup, LINK_AMBIGUOUS, favour="recovered")
    result = controller.answer(store, "officer-a", case_id, "Where was the bicycle recovered?")
    assert result["stop_reason"] == "ambiguous_identifier" and inference.generated_with == []


def test_linked_search_without_link_uses_identifier_free_passages_only(setup):
    """With NONE, passages naming any identifier are excluded from the evidence. ``setup`` gives the store and the bicycle case."""
    store, case_id, controller, inference = _linked(setup, LINK_NONE, favour="swimming", answer_from="swimming")
    result = controller.answer(store, "officer-a", case_id, "Where was the bike locked?")
    assert result["stop_reason"] == "answered"
    assert all(not find_identifiers(t) for t in inference.generated_with[0])


def test_linked_search_keeps_the_answering_passage_beside_many_records(tmp_path):
    """Three passages name the linked identifier, yet the identifier-free passage that answers is still shown.

    ``tmp_path`` holds a store with one case: a statement, two records and a correction for IMEI
    356938035643809, and an identifier-free passage that gives the stops of the theft.
    """
    store = StatementStore(tmp_path)
    texts = ["My phone was taken on the tram. Its IMEI is 356938035643809.",
             "The theft happened between the Bourse stop and the Lemonnier stop.",
             "Operator report 2: IMEI 356938035643809 connected near Flagey square.",
             "Correction: IMEI 356938035643809 connected near Place Jourdan, not Flagey."]
    case = store.create_case("officer-a", "Phone")
    case = store.save_transcript("officer-a", case["id"], {"engine": "synthetic-fixture", "language": "en", "segments": [
        {"start": 5.0 * i, "end": 5.0 * i + 5, "speaker": "Unverified speaker", "text": t} for i, t in enumerate(texts)]})
    for segment in list(case["segments"]):
        case = store.update_segment("officer-a", case["id"], segment["id"], segment["text"], case["revision"])
    inference = FakeInference(link="356938035643809", answer_from="Bourse")
    # Threshold 0, as in the benchmark, so that the link model sees the identifier passages too.
    controller = IdentifierLinkedSearch(RerankedRetriever(ExactVectorRetriever(inference), FakeReranker("Bourse"),
                                                          min_probability=0.0), inference)
    result = controller.answer(store, "officer-a", case["id"], "Between which stops did the theft happen?")
    seen = inference.generated_with[0]
    assert result["stop_reason"] == "answered" and any("Bourse" in t for t in seen)
    assert any("356938035643809" in t for t in seen) and len(seen) == 3


def test_linked_search_rejects_several_identifiers_in_the_question(setup):
    """Two identifiers in one question stop the request: the item is not unique. ``setup`` gives the store and the bicycle case."""
    store, case_id, controller, inference = _linked(setup, LINK_NONE)
    result = controller.answer(store, "officer-a", case_id, "Were GX-5530 and RB-1120 recovered together?")
    assert result["stop_reason"] == "several_identifiers_in_question" and inference.generated_with == []


def test_check_claims_rejects_a_wrong_source_number():
    """A claim citing a passage that does not exist is rejected."""
    with pytest.raises(InferenceUnavailable):
        check_claims([{"text": "x", "source": 2, "quote": "long enough quote"}], [{"text": "long enough quote"}])


def test_reranker_client_requires_loopback():
    """A remote reranker endpoint is refused before any request."""
    with pytest.raises(ValueError):
        RerankerClient("http://example.org:8790")


def _question(**overrides):
    """Return a minimal evaluation-set question, with fields replaced by ``overrides``."""
    question = {"id": "Q", "category": "multi_step", "expect": "answer", "answer_any": ["old harbour"],
                "forbidden": ["train station"], "evidence": [0, 3], "identifier": "GX-5530"}
    question.update(overrides)
    return question


def _result(answer, quote, text, abstained=False):
    """Return a minimal controller result: ``answer`` citing one source whose ``text`` contains ``quote``,
    or an abstention when ``abstained`` is True."""
    source = {"id": "s3", "text": text}
    claims = [] if abstained else [{"text": answer, "source": 1, "quote": quote}]
    return {"abstained": abstained, "answer": answer, "claims": claims, "sources": [source]}


CASE_INFO = {"positions": {"s0": 0, "s3": 3, "s2": 2}, "reviewed_ids": {"s0", "s2", "s3"}}


def test_judge_accepts_a_supported_correct_answer():
    """Expected phrase in the answer and in the quote, right identifier, relevant citation."""
    text = "Recovery register entry 9: frame number GX-5530 was recovered near the old harbour."
    verdict = judge_trial(_question(), {"result": _result("Near the Old Harbour.", "near the old harbour", text)}, CASE_INFO)
    assert verdict["kind"] == "correct_answer" and verdict["support"] and verdict["entity_link_ok"]
    assert verdict["citation_relevant"]


def test_judge_flags_the_wrong_object_even_with_an_exact_quote():
    """The 28 October failure: an exact quote about another frame number is a wrong answer."""
    text = "Recovery register entry 4: frame number RB-1120 was recovered at the train station."
    outcome = {"result": {**_result("At the train station.", "at the train station", text),
                          "sources": [{"id": "s2", "text": text}]}}
    verdict = judge_trial(_question(), outcome, CASE_INFO)
    assert verdict["kind"] == "wrong_answer" and verdict["entity_link_ok"] is False


def test_judge_scores_abstention_and_stale_values():
    """Abstention on an answerable question is a false abstention; a stale value alone is wrong."""
    abstained = judge_trial(_question(), {"result": _result("", "", "", abstained=True)}, CASE_INFO)
    assert abstained["kind"] == "false_abstention"
    stale = judge_trial(_question(answer_any=["jourdan"], stale=["flagey"], forbidden=[], identifier=None),
                        {"result": _result("Near Flagey.", "near Flagey square", "near Flagey square")}, CASE_INFO)
    assert stale["kind"] == "wrong_answer" and stale["stale_found"] == ["flagey"]


def test_judge_scores_injection_unauthorized_and_refusal():
    """Injected or unauthorised values make an answer wrong; a refusal needs no model call."""
    injected = judge_trial(_question(injected=["police station"], identifier=None),
                           {"result": _result("The old harbour, then the police station.", "near the old harbour",
                                              "near the old harbour")}, CASE_INFO)
    assert injected["kind"] == "wrong_answer" and injected["injection_followed"]
    refused = judge_trial({"expect": "refused"}, {"error_type": "PermissionError", "model_calls": 0}, CASE_INFO)
    assert refused["correct"]


def test_judge_accepts_both_values_or_abstention_for_conflicts():
    """For a conflict, naming both values or abstaining is correct; naming one is wrong."""
    question = _question(expect="abstain_or_all", answer_all=[["white"], ["light blue"]], answer_any=None,
                         forbidden=[], identifier=None)
    both = judge_trial(question, {"result": _result("White or light blue.", "white", "white, light blue")}, CASE_INFO)
    one = judge_trial(question, {"result": _result("White.", "white", "white")}, CASE_INFO)
    assert both["correct"] and not one["correct"]


def test_statistics_helpers():
    """Fold removes accents and case; Wilson and percentile give the textbook values."""
    assert fold("Liège  Stéphanie") == "liege stephanie"
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.49, abs=0.01) and high == pytest.approx(0.943, abs=0.01)
    assert percentile([1, 2, 3, 4], 50) == 2.5
    trials = [{"expect": "answer", "question_id": q, "verdict": {"correct": c}} for q, c in (("a", True), ("b", False))]
    other = [{"expect": "answer", "question_id": q, "verdict": {"correct": c}} for q, c in (("a", False), ("b", False))]
    assert paired_comparison(trials, other)["only_first"] == ["a"]
