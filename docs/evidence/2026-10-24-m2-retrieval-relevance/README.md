# M2 retrieval relevance - 24 October 2026

Status: **executed. Decision rule not met:** BGE-M3 beats the lexical baseline on recall at 3 in English and French, and ties it in Dutch. The rule fixed before the run requires a win in each language.

## Method

- Corpus and questions: the [shared fictional corpus](../2026-10-24-measurement-corpus/), 60 reviewed passages and 50 questions (17 English, 17 French, 16 Dutch), each with one or two marked answer passages. Both methods rank the same 60 passages of the same case revision (120).
- [`scripts/evaluate_retrieval_relevance.py`](../../../scripts/evaluate_retrieval_relevance.py) is a separate evaluation script. It runs inside the deployed `lab-app` pod and reuses the service objects, but asks each method for **10 ranked candidates with scores**:
  - **Vector:** BGE-M3 through the deployed Ollama, then the service's exact pgvector cosine query on the case, revision and model digest, with `LIMIT 10` instead of 3.
  - **Lexical baseline:** the service's TF-IDF ranking (`lab/retrieval.py` tokeniser, stop words and concept map) over the same 60 passages. The synthetic procedure cards that the lexical service adds are left out, so both methods see the same corpus.
- The service itself was **not changed**: `lab/vector_retrieval.py` keeps `LIMIT 3` and the cosine similarity threshold of **0.45**.
- Recall at 3 is computed on what each service passes on: the top 3 with a cosine similarity of at least 0.45 (vector), or the top 3 with a score of at least 0.10 and a shared token (lexical). Recall at 10 and mean reciprocal rank at 10 use the raw rankings. Recall is the share of marked passages found, averaged over questions.

## Results

| Language | Questions | Method | Recall@3 (service view, decision metric) | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| English | 17 | BGE-M3 vector | **0.971** | 1.000 | 0.971 |
| English | 17 | Lexical TF-IDF | 0.882 | 0.941 | 0.802 |
| French | 17 | BGE-M3 vector | **1.000** | 1.000 | 0.931 |
| French | 17 | Lexical TF-IDF | 0.941 | 1.000 | 0.881 |
| Dutch | 16 | BGE-M3 vector | 1.000 | 1.000 | 0.938 |
| Dutch | 16 | Lexical TF-IDF | 1.000 | 1.000 | **0.969** |
| All | 50 | BGE-M3 vector | 0.990 | 1.000 | 0.947 |
| All | 50 | Lexical TF-IDF | 0.940 | 0.980 | 0.882 |

Raw recall at 3 equals the service view for both methods: every marked passage that reached the top 3 also passed its method's threshold.

## Reading the result

- **Decision:** the pre-registered rule says BGE-M3 stays the default only if it beats the lexical baseline on recall at 3 in each language. It wins in English (+0.089) and French (+0.059) and **ties in Dutch** (1.000 each), where the lexical baseline even ranks slightly better (MRR 0.969 against 0.938). The rule is therefore **not met**. Read literally, the result does not justify BGE-M3 as the default on this corpus; it does not show that the lexical baseline is better overall either.
- **Ceiling effect:** recall at 3 is 0.94 or more everywhere. With one topic per language and 20 short passages each, the corpus is too easy to separate the methods well. A larger corpus with near-duplicate passages, cross-language questions and longer documents is needed before any decision about the limit of 3 or a reranker.
- **Limit and reranker:** recall at 10 reaches 1.000 for BGE-M3, and its only recall@3 miss (`Q-EN-01`, two marked passages) found one of the two. On this corpus, raising the limit above 3 or adding a reranker would gain little.
- **The 0.45 threshold barely filters:** the marked passages score 0.50 to 0.75 (median 0.63), but 351 of the 446 unmarked passages that appear in the top 10 also score 0.45 or more (median 0.48). On in-domain questions the threshold almost never causes an abstention; abstention then depends on the generator. M1 measures that consequence.
- Three vector top-1 errors crossed languages: an English passage outranked the marked French or Dutch passage (`Q-FR-05`, `Q-FR-17`, `Q-NL-06`). The marked passage still came second or third.

## Files

| File | Content |
| --- | --- |
| `runtime-context.json` | Cluster, images, model digests and source hashes of the running pod |
| `rankings-raw.json` | The 10 ranked candidates and scores of each method for each question |
| `relevance-results.json` | Per-question metrics, per-language means, the decision and the similarity distribution of marked passages |
| `SHA256SUMS` | Digests of every file in this folder |
