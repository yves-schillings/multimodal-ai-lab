# Code guide and review route

## Responsibilities

| Component | Responsibility | Useful verification |
| --- | --- | --- |
| `app.py` | FastAPI routes, loopback and origin boundary, actor/case checks, bounded uploads | Deny a different officer access before processing input |
| `lab/store.py` | Source originals, revisions and change history | A stale revision is rejected |
| `lab/backend_store.py` | Documents, durable job records and reviewed-source draft approval | An unreviewed document blocks drafting; a correction invalidates approval |
| `lab/jobs.py` | One durable local processing worker | Restart recovery and idempotent insertion |
| `lab/documents.py` | Bounded text/PDF extraction and optional local image OCR | Unsupported input returns a visible error |
| `lab/speech.py` | Prepared local faster-whisper inference and audio limits | Transcript segments require human review |
| `lab/retrieval.py` | Reviewed, case-scoped TF-IDF/lexical evidence search | Missing evidence produces abstention |
| `lab/models.py` | Sklearn training, held-out evaluation, local MLflow tracking and controlled releases | Passing/rejected candidates and rollback |
| `lab/deployment_policy.py` | Fail-closed planning decisions for resource pools | Classification, geography, CPE and quota denials |
| `static/` | Interface that uses the Python API on loopback | Review, search, drafting and model lifecycle |
| `public-demo/` | Standalone synthetic browser exercise | Naive Bayes gate comparison; no upload or server identity |

The bounded workflow endpoint is deterministic orchestration. It is not an autonomous LLM agent. Retrieval is extractive, without embeddings or generative inference in this release.

## Suggested review tomorrow

1. Start `Start.cmd` and open the synthetic bicycle case as officer A.
2. Review one source, search its location, then ask for the missing jacket colour.
3. Review the remaining sources and prepare a source-linked draft. Switch to the reviewer for approval.
4. Correct a source as officer A and inspect approval invalidation and change history.
5. Import `samples/evidence-inventory.txt` and review the extracted document.
6. Train two local classifier candidates, promote them as reviewer, then roll back. Inspect numerical MLflow runs under the ignored runtime data directory.
7. Prepare the speech model, import a fictional audio recording, and inspect timestamps and original text before accepting it.
8. Read the deployment policy tests alongside the target architecture. No external pool is active.

## Verification evidence

On the development machine, 58 Python tests and 15 subtests passed with `pytest tests` on 7 October 2026 (49 earlier tests plus 4 network-mode tests and 5 model-publication tests; `python -m unittest discover` collects only the 37 unittest-style tests). Browser checks passed for classifier gates, source eligibility and abstention. The Docker Compose stack passed the end-to-end demonstration and the persistence check described in `docs/deployment.md`. Real CPU speech inference processed an approximately eleven-second synthetic English recording through the upload/job API and produced three unreviewed segments; another officer was denied case access. This is an integration smoke check, not a speech accuracy benchmark. French and Dutch quality remain to be evaluated.

The speech runtime pins PyAV 16.1.0 because the tested faster-whisper decoder uses an argument no longer accepted by PyAV 19.0.1. The pinned combination passed real decoding and inference. Image OCR is verified in Docker. Scanned PDFs, distributed queues, embedding RAG, generative assistants and trusted identity remain future work.

## Release and container acceptance

`tests/test_model_publication.py` reloads a model downloaded from MLflow and compares its
inference result, then checks that failed publication, altered artifacts, unavailable evidence
and legacy unpublished candidates cannot be released. An unavailable registry also blocks rollback.
`scripts/container_acceptance.py` exercises the application and synthetic image OCR with an
arbitrary non-root UID and network disabled. `deploy/openshift/Check-LocalHost.ps1` reports local
resources without installing software or changing the host.
