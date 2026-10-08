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

## Demonstration route

1. Start `Start.cmd` and open the synthetic bicycle case as officer A.
2. Review one source, search its location, then ask for the missing jacket colour.
3. Review the remaining sources and prepare a source-linked draft. Switch to the reviewer for approval.
4. Correct a source as officer A and inspect approval invalidation and change history.
5. Import `samples/evidence-inventory.txt` and review the extracted document.
6. Train two local classifier candidates, promote them as reviewer, then roll back. Inspect the actual MLflow server registry and published artifact hashes on port 5000 in Compose.
7. Prepare the speech model, import a fictional audio recording, and inspect timestamps and original text before accepting it.
8. Read the deployment policy tests alongside the target architecture. No external pool is active.

## Verification evidence

On 7 October 2026, the closing local run passed 63 Python tests and 15 subtests, including five added scanned-PDF tests. Use `python -m pytest tests` so function-style tests are not omitted by unittest-only discovery. Browser core checks passed. The rebuilt Compose application passed 27 checks before restart and 8 afterwards. Real CPU speech inference processed a 10.648-second synthetic English recording in 2.851 seconds and produced three segments. A different officer was denied access. These are integration checks, not a speech accuracy benchmark. A later offline fixture measurement gives French WER 67.11% and Dutch WER 71.01% on ten formant-generated clips per language. Representative human-speech quality remains to be evaluated. See [speech evaluation](speech-evaluation.md). Reports and tested-source hashes are retained in [the closing evidence package](evidence/2026-10-07-closing/README.md).

The speech runtime pins PyAV 16.1.0 because the tested faster-whisper decoder uses an argument no longer accepted by PyAV 19.0.1. The pinned combination passed real decoding and inference. Image OCR and image-only PDF fallback are verified in Docker, including an arbitrary non-root UID with network disabled. Distributed queues, embedding RAG, generative assistants and trusted identity remain extensions. The container vulnerability scan has unresolved findings recorded in the evidence package.

## Release and container acceptance

`tests/test_model_publication.py` reloads a model downloaded from MLflow and compares its
inference result, then checks that failed publication, altered artifacts, unavailable evidence
and legacy unpublished candidates cannot be released. An unavailable registry also blocks rollback.
`scripts/container_acceptance.py` exercises the application and synthetic image OCR with an
arbitrary non-root UID and network disabled. `deploy/openshift/Check-LocalHost.ps1` reports local
resources without installing software or changing the host.
