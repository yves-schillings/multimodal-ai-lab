# Local OKD deployment verification — 9 October 2026

The application, MLflow and its PostgreSQL backend were deployed and accepted on a
single-node CRC cluster using the OKD preset. The [successful run](attempt-2/README.md)
retains image digests, platform information, measured network paths and restart results.
An earlier failed run was kept privately for diagnosis and is not acceptance evidence.
Raw pod exports and execution logs are also excluded from publication.

## Verified results

- 27 synthetic workflow checks before pod replacement; 8 persistence checks afterwards.
- All 9 allowed/denied NetworkPolicy paths matched expectations in both phases.
- Application, MLflow and PostgreSQL used restricted arbitrary UID 1000650000.
- All three workload pods were replaced; all four PVCs retained the same volumes.
- Classifier training, MLflow metrics/artifact registration, reviewer-only promotion
  and rollback passed. Authorization actors remain simulated.
- The local Python suite passed 67 tests and 15 subtests; four probe-verdict tests
  passed after the acceptance harness corrections.

## Additional local speech and OCR checks

- [Weights copy](speech-weights.json) and [repeat copy](speech-weights-repeat.json):
  four prepared Whisper base files verified by SHA-256 without replacing existing files.
- [Speech before restart](speech-before.json): an 11.063-second synthetic English
  recording processed in 2.712 seconds with faster-whisper, CPU/int8; three segments
  saved and access denied to another simulated actor.
- [Speech after restart](speech-after.json): weights, saved job and transcript persisted;
  another simulated actor remained denied.
- [OCR](ocr.json): synthetic PNG and image-only PDF extraction passed using Tesseract.

## Scope and limits

Access uses local port-forwarding, with no application Route. The case store remains
SQLite; PostgreSQL holds MLflow metadata. Retrieval remains lexical/extractive.
Local embeddings/pgvector, generative RAG, genuine login, CPE provisioning and continuous
model-quality monitoring are not implemented by this deployment update.
OpenShift AI, KServe, GPU serving and shared production hosting are not installed or accepted.
Synthetic classifier and speech checks do not establish representative human-data quality.
