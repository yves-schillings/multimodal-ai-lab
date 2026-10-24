"""M2: retrieval relevance of BGE-M3 vector search against the lexical baseline on one annotated corpus.

This separate evaluation script runs inside the deployed lab-app pod. It reuses the service objects
(VectorIndex, LocalInference, reviewed_sources and the lexical tokeniser) but issues its own ranking
queries for 10 candidates with scores. The service itself keeps `LIMIT 3` and its 0.45 cosine threshold
unchanged. Recall at 3 is computed twice: on the raw top 3, and on what the service would pass to the
generator (top 3 with cosine similarity of at least 0.45; for the lexical service, score of at least
0.10 with a shared token). Usage:

    python scripts/evaluate_retrieval_relevance.py --manifest <manifest.json> --out <evidence-dir>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measurement_common import file_sha256, now, pod_python, runtime_context, summary, write_json

CORPUS = ROOT / "tests" / "fixtures" / "measurements" / "rag_corpus.json"
CANDIDATES = 10
SERVICE_LIMIT = 3
VECTOR_THRESHOLD = 0.45
LEXICAL_THRESHOLD = 0.10

RANK_CODE = r'''
import json, os, sys, time
from pathlib import Path
from lab.rag import configured_rag
from lab.backend_store import LabStore
from lab.vector_retrieval import reviewed_sources, vector_literal
from lab.retrieval import _tokens
from sklearn.feature_extraction.text import TfidfVectorizer
P = PARAMS
rag = configured_rag()
index, inference = rag.index, rag.index.inference
store = LabStore(Path(os.environ.get("LAB_DATA_DIR", "/app/data")))
case = store.get_case(P["actor"], P["case_id"])
sources = reviewed_sources(case)
index.synchronize(case, sources)
texts = [s["text"] for s in sources]
names = [s["title"] for s in sources]
vectorizer = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False)
matrix = vectorizer.fit_transform(texts)
out = {"case_revision": case["revision"], "sources": len(sources), "questions": []}
for q in P["questions"]:
    vector = vector_literal(inference.embed([q["text"]])[0])
    with index.connect() as db:
        rows = db.execute("""SELECT source_id, 1-(embedding <=> %s::vector) AS score
            FROM retrieval.chunks WHERE case_id=%s AND revision=%s AND model_digest=%s
            ORDER BY embedding <=> %s::vector, source_id LIMIT %s""",
            (vector, P["case_id"], case["revision"], inference.embedding_digest, vector, P["k"])).fetchall()
    by_id = {s["id"]: s["title"] for s in sources}
    vector_ranked = [{"passage": by_id[r[0]][:-4], "score": round(float(r[1]), 6)} for r in rows]
    scores = (matrix @ vectorizer.transform([q["text"]]).T).toarray().ravel().tolist()
    query_tokens = set(_tokens(q["text"]))
    order = sorted(range(len(sources)), key=lambda i: (-scores[i], sources[i]["id"]))[:P["k"]]
    lexical_ranked = [{"passage": names[i][:-4], "score": round(scores[i], 6),
                       "shares_token": bool(query_tokens.intersection(_tokens(texts[i])))} for i in order]
    out["questions"].append({"id": q["id"], "vector": vector_ranked, "lexical": lexical_ranked})
    print(q["id"], file=sys.stderr, flush=True)
print(json.dumps(out))
'''


def recall(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked, relevant, k):
    for position, passage in enumerate(ranked[:k], 1):
        if passage in relevant:
            return 1 / position
    return 0.0


def score(raw, corpus):
    questions = {q["id"]: q for q in corpus["questions"]}
    rows = []
    for item in raw["questions"]:
        q = questions[item["id"]]
        vector = [c["passage"] for c in item["vector"]]
        lexical = [c["passage"] for c in item["lexical"]]
        # What the services would actually return to the generator or to the user.
        vector_service = [c["passage"] for c in item["vector"][:SERVICE_LIMIT] if c["score"] >= VECTOR_THRESHOLD]
        lexical_service = [c["passage"] for c in item["lexical"][:SERVICE_LIMIT] if c["score"] >= LEXICAL_THRESHOLD and c["shares_token"]]
        rows.append({"id": q["id"], "language": q["language"], "relevant": q["relevant"],
                     "vector": {"recall_at_3_service": recall(vector_service, q["relevant"], 3), "recall_at_3": recall(vector, q["relevant"], 3),
                                "recall_at_10": recall(vector, q["relevant"], 10), "rr_at_10": reciprocal_rank(vector, q["relevant"], 10),
                                "top1": vector[0], "top1_score": item["vector"][0]["score"],
                                "relevant_scores": {p: next((c["score"] for c in item["vector"] if c["passage"] == p), None) for p in q["relevant"]}},
                     "lexical": {"recall_at_3_service": recall(lexical_service, q["relevant"], 3), "recall_at_3": recall(lexical, q["relevant"], 3),
                                 "recall_at_10": recall(lexical, q["relevant"], 10), "rr_at_10": reciprocal_rank(lexical, q["relevant"], 10),
                                 "top1": lexical[0], "top1_score": item["lexical"][0]["score"]}})
    def mean(values):
        return round(sum(values) / len(values), 4) if values else None
    groups = {}
    for language in ("en", "fr", "nl", "all"):
        selected = [r for r in rows if language == "all" or r["language"] == language]
        groups[language] = {"questions": len(selected)}
        for method in ("vector", "lexical"):
            groups[language][method] = {metric: mean([r[method][metric] for r in selected])
                                        for metric in ("recall_at_3_service", "recall_at_3", "recall_at_10", "rr_at_10")}
        groups[language]["vector_beats_lexical_on_recall_at_3_service"] = groups[language]["vector"]["recall_at_3_service"] > groups[language]["lexical"]["recall_at_3_service"]
    relevant_scores = [s for r in rows for s in r["vector"]["relevant_scores"].values() if s is not None]
    return rows, groups, summary(relevant_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = manifest["cases"]["corpus_officer_a"]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out, "runtime-context.json", runtime_context())
    raw = pod_python(RANK_CODE, {"case_id": case["case_id"], "actor": case["owner"], "k": CANDIDATES,
                                 "questions": [{"id": q["id"], "text": q["text"]} for q in corpus["questions"]]})
    raw.update({"recorded_at": now(), "corpus_sha256": file_sha256(CORPUS), "candidates": CANDIDATES})
    write_json(args.out, "rankings-raw.json", raw)
    rows, groups, relevant_scores = score(raw, corpus)
    decision = all(groups[language]["vector_beats_lexical_on_recall_at_3_service"] for language in ("en", "fr", "nl"))
    write_json(args.out, "relevance-results.json", {
        "recorded_at": now(), "service_limit": SERVICE_LIMIT, "service_vector_threshold": VECTOR_THRESHOLD,
        "service_lexical_threshold": LEXICAL_THRESHOLD, "candidates_per_method": CANDIDATES,
        "decision_rule": "BGE-M3 stays the default only if it beats the lexical baseline on recall at 3 (service view) in each language.",
        "decision_bge_m3_stays_default": decision, "by_language": groups,
        "vector_similarity_of_relevant_passages": relevant_scores, "questions": rows})


if __name__ == "__main__":
    main()
