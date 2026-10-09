# Component registry

Fixed component names used in the architecture and article. Status: Implemented, Prepared, Target. Implemented means exercised locally, not production readiness.

| ID | Name on slides | Technology shown under the name | Status now | Target technology | Code module |
| --- | --- | --- | --- | --- | --- |
| C01 | Browser workspace | Browser UI: review, search, approval | Implemented | unchanged | `static/` |
| C02 | Case API | Python / FastAPI, case and action checks, bounded uploads | Implemented | same service on OpenShift | `app.py` |
| C03 | Job runner | JobRunner, durable SQLite job state, single worker, recovery | Implemented | workers + broker/outbox on OpenShift | `lab/jobs.py` |
| C04 | Deployment policy | Fail-closed planning policy (routing decisions, no provisioning) | Implemented (planning only) | feeds C05 | `lab/deployment_policy.py` |
| C05 | Inference gateway | Local scoped credentials, pinned model operations, request budget and serialized CPU inference | Implemented (two local AI CPEs) | organizational policy and approved remote resource routing | `lab/inference_gateway.py` |
| M01 | Speech-to-text | faster-whisper 1.2.1, PyAV 16.1.0, CPU int8, prepared local weights | Implemented (English smoke check and synthetic French/Dutch measurement; representative human quality not accepted) | OpenShift AI endpoint, GPU/CPU | `lab/speech.py` |
| M02 | PDF text extraction | Python extraction adapter | Implemented | unchanged | `lab/documents.py` |
| M03 | OCR | Tesseract adapter | Implemented in Docker (synthetic image and bounded scanned-PDF extraction tested) | OpenShift AI endpoint, pinned version | `lab/documents.py` |
| M04 | Document classifier | scikit-learn TF-IDF + logistic regression | Implemented (tiny synthetic dataset) | served pinned artifact | `lab/models.py` |
| M05 | Retrieval | Baseline lexical retrieval; optional offline BGE-M3 embeddings + pgvector on reviewed sources | Implemented (local OKD) | representative retrieval quality evaluation | `lab/retrieval.py`, `lab/vector_retrieval.py` |
| M06 | LLM generation | Local pretrained Qwen3:4b with cited quotes, current revisions and abstention | Implemented (local CPU RAG; human review required) | full-platform/GPU serving and representative evaluation | `lab/rag.py`, `lab/local_inference.py` |
| M07 | Bounded tool workflow | Deterministic extractive workflow, citations, abstention | Implemented | LLM tool agent with explicit allowlist and approval | `app.py` workflow endpoint |
| M08 | MLflow tracking and gate | MLflow metrics, published artifacts, registered versions; verified application promotion and rollback | Implemented (local) | shared registry, CI checks, serving integration | `lab/models.py`, `lab/evaluation.py` |
| D01 | LabStore | SQLite: revisions, approvals, audit, jobs | Implemented | PostgreSQL | `lab/store.py`, `lab/backend_store.py` |
| D02 | Source file store | Application-managed files: originals and derived | Implemented | controlled object storage | `lab/store.py` |
| D03 | Vector store | PostgreSQL / pgvector; separate CPE databases and roles | Implemented (local OKD) | shared production operations and physical storage quotas | `lab/vector_retrieval.py`, `scripts/deploy_local_vector.py` |
| D04 | Model artifact store | Local inference files plus published MLflow artifacts and registry references | Implemented (local) | shared model registry | `lab/models.py` |
| A01 | Identity | Default simulated personas; optional scrypt password sessions, CSRF and revocation | Implemented (local credentials; no organizational OIDC) | OpenID Connect users + scoped workload identities | `app.py`, `lab/auth.py` |
| A02 | Case access rules | Case and action checks before read, upload, search, export | Implemented | unchanged, backed by A01 | `app.py` |
| A03 | Reviewer approval | Source correction, versioned approval, invalidation after source edit | Implemented | unchanged | `lab/backend_store.py` |
| A04 | Audit records | Actor, case, action, decision, version in SQLite; no raw content | Implemented | PostgreSQL, retention policy | `lab/store.py` |
| A05 | CPE provisioning workflow | Bounded operator profiles, approval, isolated provisioning, verification, activation/revocation and expiry | Implemented (two local AI CPEs) | organization-scale approval automation | `lab/cpe.py`, `scripts/local_cpe.py`, `scripts/extend_cpe_rag.py` |
| H01 | Desktop loopback host | Python process on 127.0.0.1:8770, CPU int8, local files | Implemented | n/a | `Start.cmd`, `scripts/Start.ps1` |
| H02 | Docker container | Application + MLflow + PostgreSQL; loopback-published ports | Implemented (fresh Compose acceptance) | trusted identity before shared exposure | `Dockerfile` |
| H03 | On-site private hosts | Application VM, worker hosts, private data services, private GPU hosts | Target | n/a | `docs/deployment.md` |
| H04 | Red Hat OpenShift + OpenShift AI | Application namespace, data services, model serving, workbenches, pipelines | Target | n/a | `docs/deployment.md` |
| H05 | Secondary GPU Data Center | AI service + burst policy; GPU inventory and compatible models to confirm | Target | n/a | `lab/deployment_policy.py` |
| H06 | Sovereign Cloud | Approved cloud endpoint; GPU inventory and compatible models to confirm | Target | n/a | `lab/deployment_policy.py` |


M08 publishes parameters, metrics, a loadable sklearn model and the exact inference artifact.
D04 persists inference files locally and publishes model artifacts through MLflow. Promotion and
rollback require reviewer authority, successful tracking, a ready registered model version,
matching run identity and artifact checksums. Missing evidence blocks release. H04 application
manifests are Prepared; neither OpenShift nor OpenShift AI has been installed or accepted.

Evidence update, 8 October 2026: [offline speech measurement](evidence/2026-10-07-closing/speech-fr-nl.json) recorded 67.11% French and 71.01% Dutch word error rates on ten synthetic clips per language. These are measured limitations, not acceptance of human speech quality. [Container portability evidence](evidence/2026-10-07-fresh-clone/container-portability.txt) exercised image OCR and bounded scanned-PDF extraction. No observed GPU inventory or shared CPE is claimed. NVIDIA H100, H200, B200 and B300 are comparison examples in the article, not installed hardware assertions.

Current local acceptance: each AI CPE passes 39 verification and 13 lifecycle checks; Chromium login and restart RAG persistence also pass. MLflow-linked classifier canaries run every 1,800 seconds; they are not production drift monitoring. Selected KServe/Jupyter/training components pass 15 checks. See [current evidence](evidence/2026-10-09-local-ai-extensions/README.md). Diagram/article approval remains separate.
