# Local OKD implementation guide

This guide documents the local implementation on 9 October 2026: offline RAG (Retrieval-Augmented Generation), password accounts, recurring classifier monitoring and bounded CPE (Controlled Project Environment) activation. The lightweight Jupyter/KServe/CPU training extension has its own actual acceptance. Detailed reports and credentials remain private; source delivery is separate from article or diagram approval.

## 1. Actual runtime

CRC (CodeReady Containers) 2.64.0 runs OKD `4.22.0-okd-scos.10` with 6 logical CPUs, 24 GiB RAM and a 60 GiB disk. The Windows host has an Intel i9-9900K, 64 GiB RAM and an RTX 2080 Ti. There is no GPU pass-through to this cluster: models execute on CPU. The namespace is `multimodal-ai-lab`.

| Service | Function and exchanges | Storage |
| --- | --- | --- |
| `lab-app:8770` | FastAPI serves the browser, authenticates requests and enforces case rights. Its pod also contains the durable job worker, classifier release code and monitoring thread. | `lab-data`: SQLite cases/jobs/reviews/audit, sessions, classifier artifacts and monitor history. `lab-models`: Whisper base weights. |
| `lab-vector:5432` | PostgreSQL 16.15 / pgvector 0.8.7 stores embeddings. App connections use non-superuser `vector_client`. | `lab-vector-data` |
| `lab-inference:11434` | Ollama 0.32.13 serves BGE-M3 embeddings and Qwen3:4b generation over internal HTTP. | `lab-inference-models`, read-only for serving. |
| `lab-mlflow:5000` | MLflow accepts classifier metrics, model artifacts, registry operations and canary metrics. | `mlflow-artifacts` |
| `lab-postgres:5432` | PostgreSQL stores MLflow tracking/registry metadata, not case records. | `postgres-data` |

These are the original five service workloads, not one pod per logical architecture component. Application cases remain SQLite and local files. The optional CPE AI extension adds a scoped inference gateway and separate project vector databases. Shared case PostgreSQL, external object storage and distributed workers remain additional work.

All six original main-lab PVCs (Persistent Volume Claims) were Bound at acceptance. The completed extensions retain four additional project/workbench/result PVCs; all 11 cluster PVC identities, including the registry volume, survived the approved CRC restart. CRC hostpath volumes share one backing disk: do not add their advertised capacities or claim separate filesystem limits from PVC requests alone. Browser access uses loopback port-forward, with no application Route. CRC HTTP ingress uses port 8080 because Windows reserves port 80; HTTPS ingress remains 443.

## 2. Models and actual training

| Function | Implementation | Training performed here |
| --- | --- | --- |
| Transcription | Whisper base / faster-whisper, CPU/int8 | Pretrained weights; no fine-tuning. |
| Classification | scikit-learn TF-IDF (Term Frequency–Inverse Document Frequency) and logistic regression | Actual training on 24 fictional English documents in four classes, with eight distinct holdout examples. |
| Embeddings | BGE-M3, 1,024 dimensions | Pretrained weights; no fine-tuning. |
| Draft generation | Qwen3:4b | Pretrained weights; no fine-tuning. |
| OCR (Optical Character Recognition) | Tesseract | Existing engine; no training exercise. |

The classifier predicts `witness_statement`, `incident_report`, `evidence_inventory` or `correspondence`. Its pipeline uses unigrams/bigrams and `LogisticRegression(C=8, random_state=17, max_iter=500)`. The training exercise demonstrates reproducible evaluation, artifact verification and human release decisions. It does not establish representative multilingual quality.

The release gate requires accuracy and macro F1 of at least 0.85 and verified MLflow provenance/artifact identity. A reviewer can promote an accepted candidate or restore a verified previous release. Case-worker accounts cannot. Promotion changes the active classifier; it does not train Whisper, BGE-M3 or Qwen3.

## 3. How one RAG request works

1. The browser sends `POST /api/cases/{case_id}/question` with its session cookie and CSRF token. The server binds the request to the authenticated actor and checks case access before embedding or lookup.
2. `reviewed_sources()` selects reviewed transcripts/documents only. Documents split into 1,500-character chunks; each transcript segment contributes its first 1,500 characters. The limit is 128 reviewed chunks per case and 2,000 characters per question.
3. `VectorIndex.synchronize()` compares source hashes and revision against stored rows. If changed, BGE-M3 embeds the current reviewed sources. A transaction/advisory lock replaces that case/model's rows. Indexing happens on demand when a question is asked, not in a separate background worker.
4. BGE-M3 embeds the question. Parameterized SQL filters by case ID, revision and embedding-model digest, then ranks exact cosine distances with pgvector `<=>`. It returns at most three candidates, rejecting similarity below 0.45. This bounded implementation uses exact search, not an approximate HNSW index.
5. The app reloads authoritative case data and checks hashes. The vector database stores embeddings, identifiers and hashes; the case store supplies the source text. Missing relevant reviewed evidence produces an abstention without generation.
6. Qwen3:4b receives the question and up to three source texts through internal Ollama. Evidence is untrusted data in the prompt. The requested JSON contains at most three claims, each with a source number and exact supporting quote.
7. The app checks quote membership/source numbers and rechecks permissions/revision after generation. A concurrent edit returns HTTP 409. Missing/changed models, unavailable inference or invalid output return HTTP 503. No cloud fallback occurs.
8. The browser shows citations and a human-validation notice. An exact quote check confirms that the quote exists, not that the claim follows logically. Human review remains necessary.

Source corrections invalidate earlier evidence and draft approval through existing revision rules. The next question synchronizes the vector index. Stale vector rows never supply authorization or authoritative case content.

## 4. Configuration and offline model preparation

`deploy/openshift/local-rag/app-rag.yaml` configures `LAB_RETRIEVAL_BACKEND=pgvector`, `LAB_VECTOR_HOST=lab-vector`, `LAB_VECTOR_PORT=5432`, and `LAB_INFERENCE_URL=http://lab-inference:11434`. The vector password comes from a mounted server-side Secret file.

Model names are pinned to these SHA-256 manifest digests:

- `bge-m3:latest`: `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
- `qwen3:4b`: `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.

Setup downloads weights. The copy utility verifies every selected file and refuses to overwrite a differing existing file. Serving uses a read-only model volume, `OLLAMA_NO_CLOUD=1`, a 3 CPU/5 GiB limit, one loaded model, context 4,096 and `keep_alive=0`. Runtime requests have no remote fallback, and inference/vector network policies deny Internet egress.

The app patch sets `enableServiceLinks: false` and explicit port 5432 to prevent Kubernetes injecting `LAB_VECTOR_PORT=tcp://...` where an integer is expected.

## 5. Local accounts and sessions

`LAB_AUTH_MODE=local_sessions` activates fictional accounts `officer-a`, `officer-b` and `reviewer`. Random passwords are generated once, without defaults. An owner-only file outside Git retains passwords for local use. Never print it or include it in builds/reports.

The mounted `lab-local-credentials` Secret contains account settings and salted scrypt hashes only (N=131072, r=8, p=1, salt 16 bytes, digest 32 bytes). `identity.sqlite3` on the app data volume stores hashed random session tokens, expiry, CSRF state and credential fingerprint.

- `POST /api/auth/login` validates a password and returns the principal plus a CSRF (Cross-Site Request Forgery) token.
- The `lab_session` cookie is HttpOnly and SameSite Strict, and Secure on HTTPS. The HTTP lab uses loopback only.
- Writes need `X-Lab-CSRF`, with existing Host/Origin guards.
- Actor parameters cannot impersonate another account. Case rights and reviewer-only training/release/manual-monitor actions still apply.
- Sessions expire after one hour or 15 minutes idle. Logout deletes the session; password changes/account disablement invalidate prior sessions through checks on each request.
- The worker rechecks account availability before queued work. Case writes retain authorization/revision checks.

Administrator changes belong in the mounted credential Secret. There is no browser account-management screen. This is local authentication, not organizational OpenID Connect.

## 6. Classifier monitoring

`LAB_MONITORING=enabled` starts `QualityMonitor` inside the app. `LAB_MONITOR_INTERVAL=1800` evaluates every 30 minutes. A reviewer can call `POST /api/models/monitoring/run`. `GET /api/models/monitoring` returns the latest 20 retained reports.

The monitor verifies the promoted artifact and MLflow provenance before loading it, then predicts eight fixed fictional canary examples distinct from training and release holdout. A canary is a repeated small check for obvious regressions or a broken loading path.

MLflow experiment `synthetic-classifier-monitoring` retains accuracy, macro F1, sample count, prediction time, model hash and fixture hash. No case text or recordings enter this experiment. SQLite table `model_monitor_runs` retains local report history.

| State | Meaning |
| --- | --- |
| `healthy` | Accuracy and macro F1 both reach 0.85, with successful MLflow recording. |
| `alert` | A measured score falls below threshold. |
| `unavailable` | No active model, or provenance/evaluation/recording cannot be verified. |
| `stale` | The active release changed during evaluation. |

Monitoring never automatically retrains, promotes or rolls back. It covers this classifier only. Production-data drift, representative language evaluation and external alert delivery remain additional work.

## 7. Deployment sequence

Complete the [base local deployment](../../deploy/openshift/local.md) first. Use prepared Python dependencies and an authenticated `oc` session. Check `oc whoami --show-server` and `oc get pods,pvc -n multimodal-ai-lab` before mutations. Commands below run from the repo root. Replace angle-bracket placeholders with absolute private paths outside Git.

Prepare exact pinned host Ollama manifests/weights before copying. Tags may change: a digest mismatch must stop preparation rather than silently change the pin.

```powershell
python scripts/deploy_local_vector.py
oc apply -n multimodal-ai-lab -f deploy/openshift/local-rag/inference.yaml
python scripts/copy_inference_to_cluster.py --report <private-copy-report.json>
oc rollout restart deployment/lab-inference -n multimodal-ai-lab
oc rollout status deployment/lab-inference -n multimodal-ai-lab --timeout=240s
python scripts/build_local_app.py
oc apply -n multimodal-ai-lab -f deploy/openshift/local-rag/app-rag.yaml
oc patch deployment/lab-app -n multimodal-ai-lab --type=strategic --patch-file deploy/openshift/local-rag/app-rag-patch.yaml
$appImage = oc get istag lab-app:latest -n multimodal-ai-lab -o jsonpath='{.image.dockerImageReference}'
if ($LASTEXITCODE -ne 0 -or -not $appImage) { throw 'Cannot resolve app image' }
oc set image deployment/lab-app -n multimodal-ai-lab "app=$appImage"
oc rollout status deployment/lab-app -n multimodal-ai-lab --timeout=240s
python scripts/deploy_local_auth.py --private-output <private-credentials.json>
oc patch deployment/lab-app -n multimodal-ai-lab --type=strategic --patch-file deploy/openshift/local-rag/app-monitor-patch.yaml
oc rollout status deployment/lab-app -n multimodal-ai-lab --timeout=240s
```

The build utility stages runtime sources only, excluding models/data/secrets/virtual environments. Build completion and deployment are separate. Deploy an immutable ImageStream digest and verify rollout. The auth utility preserves existing Secret revocations. An existing Secret without its matching private file causes a stop, not account replacement.

Interactive access:

```powershell
oc port-forward service/lab-app -n multimodal-ai-lab 18770:8770 --address=127.0.0.1
```

Open `http://127.0.0.1:18770` and keep that terminal open. Log in, import fictional sources, review them, then ask a question. The acceptance scripts create their own temporary forwards.

## 8. Acceptance and limits

| Evidence | Verified result |
| --- | --- |
| Base deployment evidence at `docs/evidence/2026-10-09-openshift-local/attempt-2/README.md` | 27 workflow checks before replacement, eight afterwards, nine network paths in both phases. Includes actual speech, OCR and classifier release evidence. |
| Private authenticated RAG report | Eight checks, including actual cited generation, review/denial/abstention and vector/inference persistence after pod replacement. |
| Private account/session report | 13 checks: credential/principal/case/CSRF/role/logout behavior and persisted session after app replacement. |
| Private extended network report | 15 paths, including retained base paths, app-only vector/inference access and blocked Internet egress. |
| Private monitoring report | Five named checks: role denial, temporary interval, manual canary, distinct scheduled run and independent MLflow readback. Scheduled run: eight samples, accuracy 1.00, macro F1 1.00, FINISHED. |
| Local source checks | 104 Python tests plus 15 subtests, browser core/syntax checks. Distinct from deployment acceptance. |

The tiny synthetic monitoring score does not establish production quality. The accepted main app image is `sha256:2896f66810f524b07a5117b5589ceb4b71ee999b0e9eea21065e89c87700afb4`. A later source build needs new acceptance. The authenticated speech/OCR/release rerun subsequently passed the 16 actual checks described below.

```powershell
python scripts/cluster_auth_acceptance.py --credentials <private-credentials.json> --report <private-auth-report.json>
python scripts/cluster_rag_acceptance.py --credentials <private-credentials.json> --report <private-rag-report.json>
python scripts/cluster_rag_network_acceptance.py --report <private-network-report.json>
```

Tests create fictional records. Auth/RAG tests replace selected pods to check persistence, so avoid running them during another workflow. Use fresh private report paths. Scheduled monitoring acceptance temporarily uses 60 seconds and must restore 1,800 seconds even on failure:

```powershell
try {
    oc set env deployment/lab-app -n multimodal-ai-lab LAB_MONITOR_INTERVAL=60
    oc rollout status deployment/lab-app -n multimodal-ai-lab --timeout=240s
    if ($LASTEXITCODE -ne 0) { throw 'Test rollout failed' }
    python scripts/cluster_monitor_acceptance.py --credentials <private-credentials.json> --report <private-monitor-report.json>
    if ($LASTEXITCODE -ne 0) { throw 'Monitoring acceptance failed' }
} finally {
    oc set env deployment/lab-app -n multimodal-ai-lab LAB_MONITOR_INTERVAL=1800
    oc rollout status deployment/lab-app -n multimodal-ai-lab --timeout=240s
}
```

Finally verify the effective interval through the authenticated API. A failed probe command must not be counted as a successful network denial.

## 9. Code map and recovery

| Source | Responsibility |
| --- | --- |
| `app.py`, `static/app.js`, `static/core.js`, `static/index.html` | API/middleware and browser login, RAG and monitoring views. |
| `lab/store.py`, `lab/jobs.py` | Authoritative cases/revisions, durable jobs and authorization. |
| `lab/rag.py`, `lab/vector_retrieval.py` | Reviewed evidence, on-demand index, case-scoped search and final validation. |
| `lab/local_inference.py` | Local endpoint/model checks, requests and citation validation. |
| `lab/auth.py` | Hashing, persistent sessions, revocation and CSRF. |
| `lab/models.py`, `lab/monitoring.py` | Classifier training/release and recurring evaluation. |
| `deploy/openshift/local-rag/`, `scripts/deploy_local_*.py` | Cluster resources, credentials and feature activation. |
| `scripts/cluster_*_acceptance.py`, `tests/test_auth.py`, `tests/test_local_rag.py`, `tests/test_monitoring.py` | Actual cluster acceptance and focused unit checks. |

- Inference failure: inspect readiness, model names and pinned digests. After first weight copy, restart inference only. Do not enable cloud fallback or writable serving weights.
- HTTP 409: reload the changed case, review its current source revision and ask again.
- Session/CSRF denial: log in again and use the returned token with the same browser origin. Do not restore impersonation through actor selectors.
- Monitor unavailable: check active release, artifact identity and MLflow health. Train, verify and promote a classifier before evaluating it.
- MLflow memory pressure: the accepted base uses one worker and a 2 GiB limit. Diagnose before modifying resources, retaining tracking data.
- Pod replacement reuses PVCs. Cluster reset, PVC deletion or deleting runtime data are destructive, not routine recovery steps.

## 10. Bounded CPE acceptance and remaining scope

Two actual local CPEs, `lab-cpe-a` and `lab-cpe-b`, each pass **39 verification checks and 13 lifecycle checks** with the offline AI profile on immutable image `sha256:8cee7703d342f30dc04e730ff3b39dce13924551ee49dd005899f496d524a7c6`. Each also passes six actual Chromium browser checks, including login/case loading and absence of forbidden model-admin requests. Separate PostgreSQL databases/roles and scoped gateway credentials isolate vector access; BGE-M3 embeddings and Qwen3:4b generate cited answers from reviewed sources. Unsupported questions abstain, and corrections replace stale indexed evidence. Direct model access, other-project databases, audio and model-release actions remain denied. Both projects retain RAG after their application replacements and after the shared vector database pod replacement, with the original PVC identity preserved. The earlier lexical-only profile remains available as a baseline. Approval expires after 24 hours; the 1 GiB PVC quota controls admission on shared hostpath, not physical filesystem consumption. See [local CPE operation](local-cpe.md) and [curated extension acceptance](../evidence/2026-10-09-local-ai-extensions/README.md).

The authenticated main-lab integration passed 16 actual checks covering CPU Whisper transcription of a fictional recording, scanned-PDF Tesseract OCR, retained original text, reviewer approval, source-revision invalidation, denied case/release access, genuine classifier training/MLflow artifact publication/promotion/prediction and rollback to the original active model. Main lab remains on its accepted image6; the newly accepted CPE/gateway extension uses image12.

The private recovery exercise passed ten checks: consistent SQLite/file copies restored offline with matching hashes and integrity, `mlflow` and `lab_vectors` PostgreSQL dumps restored to isolated temporary databases with actual records, and unchanged original PVC identities. It excludes whole-cluster restoration, inference-weight backup and the MLflow artifact-server volume. Temporary restored databases were removed. Backup copies and credentials remain outside Git under owner-only file permissions.

Full Python suite: 104 tests and 15 subtests passed. This full suite includes the gateway thread-bound and CPE login changes. Unit checks remain distinct from runtime evidence. Acceptance scripts include `cluster_authenticated_workflow_acceptance.py`, `cluster_cpe_acceptance.py`, `cluster_cpe_lifecycle_acceptance.py`, `cluster_cpe_rag_acceptance.py`, `cluster_ai_platform_acceptance.py` and `cluster_backup_acceptance.py`. Their command-line help gives required private input/output paths. Use fictional recordings only; no recording is fabricated as evidence of real-world speech quality.

The [free local AI platform](local-ai-platform.md) passes 15 actual checks: Ready KServe predictor, credential denial, real classifier category and accepted artifact identity, completed CPU training, separate 24/8 training/holdout, independent MLflow readback, workbench access denial/authentication, saved native notebook, actual kernel start, private prediction from the workbench, Internet denial and notebook/PVC persistence after pod replacement. Training accuracy and macro F1 are 1.00 on tiny synthetic holdout data. This installs selected components, not the complete Red Hat OpenShift AI product. GPU execution, organizational identity, representative live-data drift and remote hosting remain additional work.

Other production work includes TLS ingress, administrator separation, representative evaluation data, physical storage quotas and complete disaster recovery. The optional CPE model profile requires the explicit model/network/resource acceptance described in its guide. Article and figure publication remain separately controlled.

## 11. Training exercise

Sign in as the fictional reviewer and train a classifier candidate in the Models view. Training fits TF-IDF (Term Frequency–Inverse Document Frequency) plus logistic regression to 24 fictional English examples. Inspect the eight-example holdout results, MLflow run, published artifact hash and quality gate. Accuracy and macro F1 must both reach 0.85 before a reviewer can promote the verified candidate. An officer cannot release it because that account lacks the model-release role.

Predict a new fictional document, inspect its proposed category and confidence, then run monitoring against eight additional canary examples. The automatic schedule repeats every 1,800 seconds and never promotes or retrains a model. Rollback restores a verified previous release. This exercise explains the training and release lifecycle using a real small classifier; it does not establish production accuracy. Whisper base, BGE-M3 and Qwen3:4b are pretrained inference models. No LLM fine-tuning occurred.
