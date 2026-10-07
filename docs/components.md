# Component registry

Fixed component names used in the architecture and article. Status: Implemented, Prepared, Target. Implemented means exercised locally, not production readiness.

| ID | Name on slides | Technology shown under the name | Status now | Target technology | Code module |
| --- | --- | --- | --- | --- | --- |
| C01 | Browser workspace | Browser UI: review, search, approval | Implemented | unchanged | `static/` |
| C02 | Case API | Python / FastAPI, case and action checks, bounded uploads | Implemented | same service on OpenShift | `app.py` |
| C03 | Job runner | JobRunner, durable SQLite job state, single worker, recovery | Implemented | workers + broker/outbox on OpenShift | `lab/jobs.py` |
| C04 | Deployment policy | Fail-closed planning policy (routing decisions, no provisioning) | Implemented (planning only) | feeds C05 | `lab/deployment_policy.py` |
| C05 | Inference gateway | Case policy, model catalogue, quota, route | Target | OpenShift service in front of model endpoints | none yet |
| M01 | Speech-to-text | faster-whisper 1.2.1, PyAV 16.1.0, CPU int8, prepared local weights | Implemented (synthetic English smoke check) | OpenShift AI endpoint, GPU/CPU | `lab/speech.py` |
| M02 | PDF text extraction | Python extraction adapter | Implemented | unchanged | `lab/documents.py` |
| M03 | OCR | Tesseract adapter | Prepared (engine not installed, scanned PDF not accepted) | OpenShift AI endpoint, pinned version | `lab/documents.py` |
| M04 | Document classifier | scikit-learn TF-IDF + logistic regression | Implemented (tiny synthetic dataset) | served pinned artifact | `lab/models.py` |
| M05 | Retrieval | TF-IDF lexical retrieval on reviewed sources | Implemented | local embedding model + pgvector, measured against the lexical baseline | `lab/retrieval.py` |
| M06 | LLM generation | none | Target | locally served LLM on OpenShift AI behind C05 | none yet |
| M07 | Bounded tool workflow | Deterministic extractive workflow, citations, abstention | Implemented | LLM tool agent with explicit allowlist and approval | `app.py` workflow endpoint |
| M08 | MLflow tracking and gate | MLflow metrics, application quality gate, promotion, rollback | Implemented (local) | shared registry, CI checks, serving integration | `lab/models.py`, `lab/evaluation.py` |
| D01 | LabStore | SQLite: revisions, approvals, audit, jobs | Implemented | PostgreSQL | `lab/store.py`, `lab/backend_store.py` |
| D02 | Source file store | Application-managed files: originals and derived | Implemented | controlled object storage | `lab/store.py` |
| D03 | Vector store | none | Target | PostgreSQL pgvector | none yet |
| D04 | Model artifact store | Application-managed model files and version references | Implemented (local) | shared model registry | `lab/models.py` |
| A01 | Identity | Simulated case personas | Implemented (simulation) | OpenID Connect users + scoped workload identities | `app.py` |
| A02 | Case access rules | Case and action checks before read, upload, search, export | Implemented | unchanged, backed by A01 | `app.py` |
| A03 | Reviewer approval | Source correction, versioned approval, invalidation after source edit | Implemented | unchanged | `lab/backend_store.py` |
| A04 | Audit records | Actor, case, action, decision, version in SQLite; no raw content | Implemented | PostgreSQL, retention policy | `lab/store.py` |
| A05 | CPE provisioning workflow | Request, approval, provisioning, verification, activation | Target | OpenShift project automation | none yet |
| H01 | Desktop loopback host | Python process on 127.0.0.1:8770, CPU int8, local files | Implemented | n/a | `Start.cmd`, `scripts/Start.ps1` |
| H02 | Docker container | Application + MLflow + PostgreSQL; loopback-published ports | Implemented (saved Compose acceptance) | trusted identity before shared exposure | `Dockerfile` |
| H03 | On-site private hosts | Application VM, worker hosts, private data services, private GPU hosts | Target | n/a | `docs/deployment.md` |
| H04 | Red Hat OpenShift + OpenShift AI | Application namespace, data services, model serving, workbenches, pipelines | Target | n/a | `docs/deployment.md` |
| H05 | Secondary GPU Data Center | AI service + burst policy, H100/B300-class capacity to confirm | Target | n/a | `lab/deployment_policy.py` |
| H06 | Sovereign Cloud | Approved cloud endpoint, B300-class capacity to confirm | Target | n/a | `lab/deployment_policy.py` |


M08 records parameters and metrics. D04 model files remain in the application volume; artifact upload and shared registry publication are not wired. Application promotion requires reviewer authority and the quality gate, but does not require successful MLflow logging. H04 manifests are Prepared; the complete OpenShift AI platform remains Target.
