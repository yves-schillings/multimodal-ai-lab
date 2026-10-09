"""Privacy and integrity checks at local inference/retrieval boundaries."""
import pytest

from lab.local_inference import InferenceUnavailable, LocalInference
from lab.rag import LocalRag
from lab.store import ConflictError, StatementStore
from lab.vector_retrieval import VectorIndex


DIGEST = "a"*64


def client():
    return LocalInference("http://127.0.0.1:11434", "bge-m3:latest", DIGEST, "qwen3:4b", DIGEST)


@pytest.mark.parametrize("url", ["https://api.example.com", "http://localhost.evil.example", "http://127.0.0.1:11434/api", "http://user:secret@localhost"])
def test_remote_or_ambiguous_endpoints_rejected(url):
    with pytest.raises(ValueError):
        LocalInference(url, "bge-m3:latest", DIGEST)


def test_changed_model_digest_blocks_inference(monkeypatch):
    inference = client()
    monkeypatch.setattr(inference, "_request", lambda *args: {"models": [{"name": "bge-m3:latest", "digest": "b"*64}]})
    with pytest.raises(InferenceUnavailable):
        inference.embed(["Private case text"])


def test_generation_cannot_cite_another_source_or_invent_quote(monkeypatch):
    inference = client()
    monkeypatch.setattr(inference, "verify", lambda *args: None)
    monkeypatch.setattr(inference, "_request", lambda *args: {"response": '{"abstained":false,"claims":[{"text":"Invented claim","source":1,"quote":"Invented supporting text"}]}'})
    with pytest.raises(InferenceUnavailable):
        inference.generate("What happened?", [{"text": "Reviewed testimony about a blue bicycle."}])


def test_denied_case_never_contacts_embedding_or_vector_server(tmp_path, monkeypatch):
    index = VectorIndex({}, client())
    monkeypatch.setattr(index, "connect", lambda: pytest.fail("Unauthorized database access"))
    monkeypatch.setattr(index.inference, "embed", lambda *args: pytest.fail("Unauthorized embedding access"))
    with pytest.raises(PermissionError):
        index.retrieve(StatementStore(tmp_path), "officer-b", "demo-case-a", "Where was the bicycle?")


def test_edit_during_generation_withholds_answer(tmp_path):
    store = StatementStore(tmp_path)
    case = store.get_case("officer-a", "demo-case-a")
    case = store.update_segment("officer-a", case["id"], case["segments"][0]["id"], case["segments"][0]["text"], case["revision"])
    segment = case["segments"][0]
    source = {"id": segment["id"], "title": "Reviewed transcript segment 1", "text": segment["text"],
              "start": segment["start"], "end": segment["end"], "case_id": case["id"], "kind": "reviewed_transcript"}
    class Inference:
        generation_model, generation_digest = "qwen3:4b", DIGEST
        def generate(self, *args):
            store.update_segment("officer-a", case["id"], segment["id"], "Corrected testimony.", case["revision"])
            return [{"text": "Old testimony", "source": 1, "quote": source["text"]}]
    class Index:
        inference = Inference()
        def retrieve(self, *args):
            return {"case_revision": case["revision"], "sources": [source], "abstained": False}
    with pytest.raises(ConflictError):
        LocalRag(Index()).answer(store, "officer-a", case["id"], "Where?")
