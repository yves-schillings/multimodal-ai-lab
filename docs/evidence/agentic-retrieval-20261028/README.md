# Agentic search with TF-IDF and with BGE-M3 + pgvector — 28 October 2026

This package compares two search backends under the **same** bounded agent and the **same** Qwen3:4b model, on synthetic cases with simulated actors. It is a small local demonstration. It is not a quality, multilingual or deployment acceptance. The historical 27 October record, [`agentic-search-v3.json`](../ai-engineering-20261027/agentic-search-v3.json), is unchanged.

| Record | Meaning |
| --- | --- |
| [`protocol.json`](protocol.json) | Questions, expected passages, rules and limits, written before any run |
| [`agentic-search-tfidf-20261028.json`](agentic-search-tfidf-20261028.json) | Rerun with the historical reviewed-only TF-IDF retriever |
| [`agentic-search-bge-m3-pgvector-20261028.json`](agentic-search-bge-m3-pgvector-20261028.json) | Run with BGE-M3 embeddings stored and searched in PostgreSQL with pgvector, plus the guard checks |
| [`agentic-index-persistence-20261028.json`](agentic-index-persistence-20261028.json) | Index rows after a restart of the pgvector pod, and the clean-up |
| [`candidate-acceptance-tfidf-20261028.json`](candidate-acceptance-tfidf-20261028.json) | Evidence gate applied to the TF-IDF run |
| [`candidate-acceptance-bge-m3-pgvector-20261028.json`](candidate-acceptance-bge-m3-pgvector-20261028.json) | Evidence gate applied to the BGE-M3 + pgvector run |

## The five functions

| Function | Who provides it |
| --- | --- |
| Encode the reviewed passages, once, and each question or rewritten query | BGE-M3 (`bge-m3:latest`, 1,024 dimensions), on CPU through Ollama |
| Store the passage vectors and return the nearest passages of the authorised case | PostgreSQL 16.15 with pgvector 0.8.7, on the local OKD cluster |
| Plan the next step, then write the cited answer | Qwen3:4b, called first as the planner and then as the generator |
| Orchestrate, control and bound the run | `AgenticSearch`, the lab's Python controller |
| Run both models locally | Ollama 0.32.13 |

The same Qwen3:4b in two roles is one agent, not two. LangGraph, OpenFGA, vLLM, KServe and OpenShift AI were **not** part of this run.

Only the retriever changes between the two runs. Controller, prompts, JSON schemas, model digests, limits (three searches, 90 seconds, CPU only, `keep_alive 0`) and hardware (Intel Core i9-9900K, 64 GB RAM) are the same. The pgvector run cannot fall back to TF-IDF.

- **TF-IDF** fits its vocabulary at each search and keeps a passage at cosine score 0.10 or more that shares a word with the question, top 3.
- **BGE-M3 + pgvector** indexes each reviewed passage once with its case, revision, content hash and embedding-model digest. A search embeds the question, takes the three nearest passages of the authorised case and keeps those with 1 − cosine distance of 0.45 or more. Access is checked before any embedding, and rights and revision are rechecked before an answer is returned.

## Results

| Question | TF-IDF | BGE-M3 + pgvector |
| --- | --- | --- |
| 1. Where was the blue bicycle left? | Answered "library", 1 search, 43.5 s | Answered "library", 1 search, 46.4 s |
| 2. At what location was the cycle parked? (paraphrase) | Answered "library" after 3 searches, 55.3 s | Answered "library" at the first search, 46.1 s |
| 3. What is the registration number of the vehicle? (not in the case) | No answer: controller stop `repeated_query`, 24.8 s | No answer: Qwen3:4b chose to abstain, 17.3 s |
| 4. Where was the complainant's bicycle recovered? (two steps) | **Failed**: "train station", 2 searches, 49.1 s | **Failed**: "train station" and "university car park", 1 search, 50.3 s |

- **Three basic questions.** With BGE-M3 + pgvector, all three met the answer rule written before the run. With TF-IDF, two did: question 3 ended with a protective controller stop, not an abstention chosen by the model. Neither backend returned an answer to question 3.
- **Historical rule.** The historical rule asks the paraphrase to need more than one search. BGE-M3 found the evidence at the first search, so it fails that rule while its answer is correct. Both verdicts are recorded.
- **Fourth case.** The answer requires two passages: the frame number of the complainant's bicycle (GX-5530), then the register entry for that frame number (the old harbour). Both backends **failed**. Each answer cites an exact quote from a register entry about **another** frame number. The quote check passed, yet the answer is false: an exact quote does not prove that a passage answers the question.

**Retrieval only**, with no language model. TF-IDF scores every passage 0 on the paraphrase. BGE-M3 ranks the library passage second (0.632). On the two-step question, TF-IDF ranks the frame-number passage first and the right register entry fifth. BGE-M3 ranks the right register entry fourth (0.500) and the frame-number passage sixth (0.490), so neither is in its top three. A semantically close passage is not evidence: on question 3, BGE-M3 kept "I did not see who removed it…" (0.535), which Qwen3:4b did not use.

**Time per step**, single CPU observations with model reloads. One TF-IDF search took about 5 ms. One BGE-M3 + pgvector search took about 4.1 to 4.8 s, of which the BGE-M3 embedding took about 4.0 to 4.7 s and the database about 0.1 s. Indexing took 5.1 s for 3 passages and 4.5 s for 7. A Qwen3:4b plan took about 11 to 17 s and an answer about 19 to 34 s. These are not capacity or latency figures.

## Rights, changes and storage

All checks passed:

- a request for another actor's case is refused before any embedding or model call;
- an unreviewed passage is never embedded or returned;
- after an edit, only the new revision stays indexed and the old passage is not cited;
- a revocation during the search stops the agent before any later model step;
- an unchanged case is synchronised again in 48 ms with no embedding call;
- after a restart of the pgvector pod, the 3 + 7 indexed rows match the reviewed passages and are reused with no embedding call. The test rows were then deleted.

Controller stops that the real runs did not reach (attempt limit, deadline, invalid plan, invalid quote) remain covered by the scripted unit tests.

## Evidence gate

[`engineering_acceptance.py`](../../../lab/engineering_acceptance.py) now has a separate check, `rag.multi_step_case`. It passes only when a predefined two-step question reaches its expected answer after more than one search. A wrong answer with an exact quote fails, and a missing two-step case fails. [`gate_agentic_retrieval.py`](../../../scripts/gate_agentic_retrieval.py) applies the rules from the protocol, with the existing vision and explanation records unchanged.

| Gate check | TF-IDF | BGE-M3 + pgvector |
| --- | --- | --- |
| `rag.supporting_quotes` | pass | pass |
| `rag.expected_outcomes` (three basic questions) | fail | pass |
| `rag.multi_step_case` (fourth case) | **fail** | **fail** |
| Decision | `blocked` | `blocked` |

The gate also stays blocked by the earlier vision and explanation failures (random vision weights, no held-out vision quality, LIME fidelity), and human review stays pending. **The fourth case keeps the search candidate blocked until a corrected agent passes it in a new recorded run.** Possible corrections, each to be tested before any claim: an answer check separate from the quote check, a rule that a linked identifier must be cited with the passage that uses it, hybrid lexical and BGE-M3 search for exact codes, or a stronger planner prompt or model.

## Limits

- Four synthetic English questions cannot establish general or multilingual answer quality.
- Simulated actors, not real authentication.
- Local OKD pgvector, not a deployment acceptance.
- An exact quote proves a citation, not the truth of a claim.

## Reproduce

```bash
python scripts/compare_agentic_retrieval.py --backend lexical --protocol protocol.json --out tfidf.json
python scripts/compare_agentic_retrieval.py --backend pgvector --protocol protocol.json --out bge.json --fixture-dir store
python scripts/compare_agentic_retrieval.py --persistence bge.json --fixture-dir store --out persistence.json
python scripts/gate_agentic_retrieval.py --run bge.json --vision vision-resnet-open-loop.json --explainability explainability-comparison.json --out gate.json
```

The runs used `compare_agentic_retrieval.py` as committed in `598a5b5` (SHA-256 `82a56e8c…`, recorded in each run file). A later commit only adds docstrings and comments to the comparison script, the gate script, the gate and its tests: their syntax trees, once docstrings are removed, are identical to the versions used for the runs.

The pgvector run reads `LAB_VECTOR_HOST`, `LAB_VECTOR_PORT`, `LAB_VECTOR_DATABASE`, `LAB_VECTOR_USER` and `LAB_VECTOR_PASSWORD_FILE`. Keep the password file outside the repository. Each run record holds its commit, source hashes, model digests, versions, hardware, ranked passages, actions, per-step timings and stop reasons. The synthetic case store used for the restart check is not included.
