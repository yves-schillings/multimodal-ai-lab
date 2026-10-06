"""Case-scoped extractive retrieval; no embeddings, generative model or API call."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .fixtures import PROCEDURE_CARDS

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # The local demo remains usable with the Python standard library.
    TfidfVectorizer = None


_STOP = set("a an and are as at be been but by can could did do does for from had has have how i in into is it its me my not of on or our that the their them there these they this to was we were what when where which who why will with would you your about any please tell show say said le la les un une des du de et est ce cette dans au aux en quel quelle quels quelles qui que quoi comment quand pour sur je il elle leze het de een en van op is wat hoe wie waar wanneer mij mijn ik was kan over".split())
_CONCEPTS = {
    "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle", "velo": "bicycle", "fiets": "bicycle",
    "returned": "return", "returning": "return", "retourne": "return", "retour": "return",
    "reviewed": "review", "reviewing": "review", "reviews": "review", "revision": "review",
    "approved": "approval", "approve": "approval", "approving": "approval", "goedkeuring": "approval",
    "corrections": "correction", "correct": "correction", "corrected": "correction",
    "segments": "segment", "transcripts": "transcript", "statements": "statement",
    "documents": "document", "sources": "source", "recordings": "recording", "drafting": "draft",
    "broken": "damage", "damaged": "damage", "damages": "damage", "windows": "window",
}


def _tokens(text):
    normalized = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return [_CONCEPTS.get(token, token) for token in re.findall(r"[\w]+", normalized)
            if len(token) > 1 and token not in _STOP]


def _lexical_cosine(query, documents):
    # Count-vector cosine, accurately labelled as lexical rather than embedding search.
    query_vector = Counter(_tokens(query))
    norm_query = math.sqrt(sum(value * value for value in query_vector.values()))
    result = []
    for document in documents:
        vector = Counter(_tokens(document))
        norm = norm_query * math.sqrt(sum(value * value for value in vector.values()))
        result.append(sum(value * vector[token] for token, value in query_vector.items()) / norm if norm else 0.0)
    return result


def answer_question(store, actor, case_id, question):
    # This permission check occurs BEFORE constructing any ranking corpus.
    case = store.get_case(actor, case_id)
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Provide a question of 1 to 2000 characters.")
    sources = []
    for index, segment in enumerate(case["segments"]):
        if segment["reviewed"]:
            sources.append({"id": segment["id"], "title": f"Reviewed transcript segment {index + 1}",
                            "text": segment["text"], "start": segment["start"], "end": segment["end"],
                            "case_id": case_id, "kind": "reviewed_transcript"})
    sources.extend({**card, "kind": "synthetic_procedure", "case_id": None} for card in PROCEDURE_CARDS)
    for document in case.get('documents', []):
        if document['reviewed']:
            # Bounded chunks retain the original document identifier for citations.
            for offset in range(0, len(document['text']), 1500):
                sources.append({'id': document['id'] + ':' + str(offset), 'document_id': document['id'],
                                'title': document['name'], 'text': document['text'][offset:offset+1500],
                                'case_id': case_id, 'kind': 'reviewed_document', 'start': None, 'end': None})
    documents = [source["text"] for source in sources]
    backend = "lexical_count_cosine"
    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False)
        try:
            matrix = vectorizer.fit_transform(documents)
            query = vectorizer.transform([question])
            scores = (matrix @ query.T).toarray().ravel().tolist()
            backend = "tfidf_cosine"
        except ValueError:  # Empty vocabulary: return the same explicit abstention path.
            scores = _lexical_cosine(question, documents)
    else:
        scores = _lexical_cosine(question, documents)
    tokens = set(_tokens(question))
    matches = [(score, source) for score, source in zip(scores, sources)
               if score >= 0.10 and tokens.intersection(_tokens(source["text"]))]
    matches.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    result = {"case_id": case_id, "case_revision": case["revision"], "mode": "extractive",
              "retrieval_backend": backend, "sources": [], "simulated_identity": True,
              "notice": "Quoted source excerpts only. No LLM, embeddings, legal inference or cross-case search. Procedure cards are synthetic demo instructions."}
    if not matches:
        result.update({"answer": "No relevant reviewed source was found in this case or the synthetic demo procedure cards. I cannot answer from the available evidence.",
                       "abstained": True})
        return result
    selected = [{**source, "score": round(score, 6)} for score, source in matches[:3]]
    result.update({"answer": "Relevant source excerpts (these do not establish verified facts):\n\n" +
                             "\n\n".join(f"[{index + 1}] {source['text']}" for index, source in enumerate(selected)),
                   "sources": selected, "abstained": False})
    return result
