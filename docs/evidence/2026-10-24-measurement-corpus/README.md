# Shared fictional corpus for M1, M2 and M3 - 24 October 2026

M1 (prompt injection and abstention) and M2 (retrieval relevance) share one set of fictional sources and questions, as planned. M3 uses the same cases for its retrieval, generation and cited-answer measurements.

## Content

The corpus is [`tests/fixtures/measurements/rag_corpus.json`](../../../tests/fixtures/measurements/rag_corpus.json). Everything in it is fictional and written for these measurements; people appear by role only.

| Part | Count | Detail |
| --- | --- | --- |
| Passages | 60 | 20 English (a workshop break-in), 20 French (a cable theft on a building site), 20 Dutch (an electric-bicycle theft at a station). One passage per document, at most 409 characters, so one retrieval chunk each. |
| Passages hiding an instruction | 20 | 8 English, 7 French, 5 Dutch. Seven types: append a code (4), leave the case scope (3), invent an answer when unsupported (3), reveal the system prompt (3), accuse a person (3), cite a non-existent evidence number (2), escalate privileges and close the case (2). |
| Relevance questions (M2) | 50 | 17 English, 17 French, 16 Dutch, each with one or two marked answer passages. |
| Answerable questions (M1) | 20 | The subset of the 50 that targets each of the 20 instruction-bearing passages exactly once. |
| Unsupported questions (M1) | 20 | 7 English, 7 French, 6 Dutch. No passage answers them; most are near misses on the same topics. |
| Control cases | 2 | A restricted case of officer-b and a second case of officer-a, each holding a code word that the injected instructions try to reveal. |

## Loading

[`scripts/prepare_measurement_corpus.py`](../../../scripts/prepare_measurement_corpus.py) used the normal authenticated API: create case, upload each passage as a UTF-8 document, wait for the extraction job, confirm the extracted text unchanged as reviewed. It loaded the corpus twice (officer-a and officer-b, for the concurrent M3 users) and created the two control cases. [`manifest.json`](manifest.json) maps passages to case document identifiers and records the corpus SHA-256.

The case identifiers are random laboratory identifiers of fictional records. No credential, session or token is stored here.
