# Offline RAG and local password sessions on OKD

For the step-by-step runtime, code map, model/training explanation, reproducible deployment, recovery and acceptance scope, read the [local implementation guide](local-implementation.md).

This optional extension runs on the existing single-node local OKD application. It keeps the original case, MLflow and speech volumes. It is a synthetic lab, not institutional identity integration, OpenShift AI installation or production acceptance.

## Runtime components

- A separate PostgreSQL 16.15 / pgvector 0.8.7 service owns vector data. The application uses a non-superuser `vector_client` account, not the bootstrap administrator.
- Ollama 0.32.13 serves pinned **BGE-M3** embeddings (1,024 dimensions) and **Qwen3 4B** generation inside OKD, on CPU. Model manifests and every copied weight file are verified by SHA-256. Existing host models are preserved.
- Reviewed source revisions are indexed per case. Permission checks run before retrieval and again after generation. Source edits invalidate old indexed evidence. Missing evidence leads to abstention; missing inference leads to an explicit error, never cloud fallback.
- Generated claims must cite a returned source and supply an exact source quote. This verifies citation syntax and quote membership; it does **not** prove factual entailment. Human review remains required.
- Optional local password accounts bind sessions to the existing fictional case actors. Caller-selected actors cannot override the logged-in account. Case ownership and reviewer-only actions still apply.
- Optional monitoring evaluates the promoted document classifier on eight fixed fictional canary examples, distinct from the training and release holdout. Numeric results and model/fixture hashes are recorded in MLflow. A failed threshold raises an alert; it does not change the active release. This is not production-data drift detection.

## Prepare and activate

Use an authenticated `oc` session for the **local** cluster. The application must already have passed its base deployment acceptance. Keep credentials and acceptance output outside the Git checkout.

1. Prepare the two pinned Ollama models on the host. Setup downloads model weights; recordings and case documents never enter remote inference.
2. Run `python scripts/deploy_local_vector.py` to create the separate vector service and client role. Existing secrets and data are retained.
3. Apply `deploy/openshift/local-rag/inference.yaml` and copy weights with `scripts/copy_inference_to_cluster.py --report <private-report>`. After the first copy, restart **only** `deployment/lab-inference` so it can discover the prepared manifests. Its model volume remains read-only.
4. Build using `python scripts/build_local_app.py`. The utility uploads only runtime sources; do not submit the entire checkout as a binary build context.
5. Apply `app-rag.yaml` and strategic patch `app-rag-patch.yaml` from the same directory to `deployment/lab-app` in `multimodal-ai-lab`. An explicit vector port and disabled Kubernetes service-link injection prevent conflicting environment variables.
6. Run `scripts/deploy_local_auth.py --private-output <owner-only-file-outside-repository>`. It creates random passwords once and stores only salted scrypt hashes in the mounted Secret. No default password exists. Rerunning does not overwrite existing accounts or undo administrator revocations.
7. Apply strategic patch `app-monitor-patch.yaml` to enable the 30-minute canary schedule. Monitoring is disabled unless explicitly configured.

All Secrets remain server-side. Passwords are salted with scrypt (N=131072, r=8, p=1); random server sessions are stored as hashes. Sessions expire after one hour or 15 minutes idle, whichever occurs first. Writes require the session's CSRF token. Logout, password changes and disabled accounts invalidate access; queued case work rechecks account availability. Browser cookies are HttpOnly and SameSite Strict, with Secure enabled for HTTPS. Keep the HTTP test service behind loopback port-forward; do not expose it through a public Route.

Administrator account changes belong in the mounted credential Secret, not the UI. New account management, organizational OpenID Connect and production identity assurance are separate work. The local reviewer is a fictional lab account.

## Actual acceptance scripts

- `cluster_rag_acceptance.py --credentials <private-file> --report <private-report>`: actual intake, review, cited generation, abstention, identity denial and vector/model persistence after pod replacement.
- `cluster_rag_network_acceptance.py --report <private-report>`: original nine network paths plus six vector/inference paths. Probe execution failures never count as successful denials.
- `cluster_auth_acceptance.py --credentials <private-file> --report <private-report>`: real credentials, account binding, CSRF, role denial, logout and session persistence after application replacement.
- `cluster_monitor_acceptance.py --credentials <private-file> --report <private-report>`: reviewer-triggered and actual scheduled canary runs, verified on the MLflow server. For this bounded test set the interval temporarily to 60 seconds, then restore 1,800 seconds and verify rollout.

Local unit tests do not replace these cluster checks. Keep raw reports private until publication of the verified implementation is explicitly authorized. CPE provisioning and activation remain separate from this extension.
