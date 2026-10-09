# Bounded local CPE operation

A Controlled Project Environment (CPE) is an operator-approved project boundary whose application stays inactive until its controls pass verification. Two local profiles use fictional documents, isolated namespaces and the actual FastAPI application. The baseline approves document extraction, human review and lexical retrieval. The optional AI profile adds **offline BGE-M3 embeddings and Qwen3:4b cited generation** through a controlled inference gateway. Both profiles deny speech, classifier training and model release; those remain in the main lab.

## Operator prerequisites

Use the existing local CRC administrator kubeconfig, never a credential copied into the application. The controller permits only `api.crc.testing`. Keep generated passwords and evidence in an owner-only private directory **outside the Git checkout**. Default developer credentials cannot provision namespaces. Do not print kubeconfig contents, passwords or Secret data.

## Provision, verify and activate

Build the runtime sources with `scripts/build_local_app.py` only after source changes. Existing accepted images do not need another build. The controller reads the local ImageStream and pins its immutable digest.

```powershell
python scripts/local_cpe.py provision --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir> --approval-reference <synthetic-owner-reference>
python scripts/local_cpe.py provision --project b --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir> --approval-reference <synthetic-owner-reference>
python scripts/cluster_cpe_acceptance.py --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/cluster_cpe_lifecycle_acceptance.py --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/cluster_cpe_acceptance.py --project b --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/cluster_cpe_lifecycle_acceptance.py --project b --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
```

Both projects must exist before cross-project verification. Verification admits only the reviewer while tenants remain inactive. The lifecycle exercise activates from fresh passing evidence, checks denial and persistence, then leaves the passing project active. Run it without another workflow in that project. A failure revokes the project and preserves storage/evidence.

For normal activation after a fresh passing verification:

```powershell
python scripts/local_cpe.py activate --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
```

Activation requires all checks passing, a matching profile and protected-resource fingerprint, verification less than ten minutes old and unexpired approval. It records the evidence file's SHA-256 in the read-only operator control. Credentials, approval and audit evidence stay private. Approval expires after 24 hours; this is a local exercise, not unattended permanent hosting.

## Actual boundaries and limits

| Control | Local implementation |
| --- | --- |
| Identity | Separate randomly generated project passwords, scrypt hashes in project Secret, persistent server sessions. Runtime service account token not mounted. |
| Network | Default-deny ingress and egress. Same-namespace application access on TCP 8770. The AI profile additionally permits only the scoped gateway, vector database and cluster DNS. No Route, Internet, other project's application or direct Ollama access. Port-forward is an operator path that bypasses NetworkPolicy. |
| Resources | Container requests 500 mCPU / 512 MiB, limits 1 CPU / 1 GiB. Namespace admission budgets constrain requests, limits, GPU allocation, PVC count and storage request. |
| Secrets | Read-only credentials mount. Runtime account cannot list Secret objects or change operator configuration. No Kubernetes credentials in the application. |
| Storage | A distinct 1 GiB PVC request per project. SQLite, originals and audit records persist across pod replacement. CRC hostpath shares physical disk capacity; this is **admission enforcement, not a physical 1 GiB filesystem quota**. |
| Models | Baseline: no approved endpoints. AI profile: exactly two pinned pretrained models, authenticated gateway, bounded requests and one shared inference slot. Audio, training, release, model downloads and arbitrary remote models remain denied. |
| Workload contract | Restricted pod security, immutable approved image, fixed service account, no mounted API token, no host network/PID/IPC, no init/ephemeral containers, allowed volume sources and read-only root filesystem. The temporary volume is operator-managed at 64 MiB and included in the verification fingerprint. |

The same application validates the current phase on each business request, queued job authorization and store actor check. Revoked, expired, malformed or missing controls return HTTP 503. Model actions outside the approved profile return HTTP 403. Case writes still check ownership and source revision. This local operator boundary is separate from organizational identity and production assurance.

Mounted ConfigMaps propagate through kubelet rather than a direct application API. The operator updates a scoped pod annotation to prompt refresh and acceptance waits for the actual observed application phase. This follows the [Kubernetes mounted ConfigMap update mechanism](https://kubernetes.io/docs/tasks/configure-pod-container/configure-pod-configmap/). Do not assume a ConfigMap write alone has already revoked a running session.

## Revocation, image changes and recovery

```powershell
python scripts/local_cpe.py revoke --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/local_cpe.py upgrade-image --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
```

Revocation retains every PVC and evidence file. An image upgrade revokes first, pins the new local digest, updates the scoped admission contract and deployment, then returns to provisioned. Fresh verification is mandatory. The controller refuses to reprovision an existing runtime, avoiding accidental credential or data replacement.

The acceptance scripts separately demonstrate pod-to-pod NetworkPolicy denial and operator port-forward access. A probe execution failure never counts as a successful denial. Lifecycle checks retain the same PVC UID, existing case and session after replacing only the project's application pod. Do not reset the cluster or remove namespaces/PVCs as a routine recovery action.

## Extend both projects to offline RAG

First prepare and accept the main lab's offline vector and inference services using [the RAG guide](local-rag.md). The extension reuses their pinned weights; it does not download models during requests. Build only changed runtime sources, then run:

```powershell
python scripts/extend_cpe_rag.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-cpe-dir>
python scripts/cluster_cpe_rag_acceptance.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-cpe-dir>
```

The operator revokes both projects before modifying protected controls. Each gets its own PostgreSQL database and non-administrative role (`cpe_vectors_a`/`cpe_vector_a`, or `cpe_vectors_b`/`cpe_vector_b`). Each role is denied connection to the other project, the main vector database and the administration database. The original case PVCs and accounts are retained. Read-only mounted credentials authenticate project-specific access to the shared gateway; no Kubernetes token is mounted.

The gateway permits only tags, embedding and generation operations for the approved model digests. It denies pull/download operations and cloud fallback, limits each project to 32 inference requests per minute, bounds inputs and serializes CPU inference with a five-second capacity wait. Shared inference does not imply shared case data. Reviewed sources, permission checks, current revisions and checked supporting quotes remain authoritative. Citation checks do not prove factual entailment; human review remains required.

On 9 October 2026, both AI profiles passed 39 verification checks, 13 lifecycle checks and six actual Chromium browser checks each. The wrapper also passed both application-replacement RAG checks and all three vector-restart/PVC checks. See the [curated accepted evidence](../evidence/2026-10-09-local-ai-extensions/README.md). The accepted VM uses six vCPUs and 24 GiB RAM; the preceding 16 GiB run stalled during inference and was not accepted.

Fresh verification checks real cited generation, unsupported-question abstention, replacement of stale indexed text after a source correction, cross-project database denial and all original isolation controls. The lifecycle exercise separately checks revocation, expiry and app replacement. Finally, the wrapper replaces only the vector database pod and checks both projects' answers and the original PVC identity. Actual reports remain private. Approval still expires after 24 hours; changing the approved profile requires fresh acceptance.

## Remaining production scope

Organizational OIDC, TLS ingress, administrator separation, retention/expiry automation and enforced physical storage quotas remain production work. [The lightweight AI platform](local-ai-platform.md) installs separate KServe/Jupyter/training components on this CPU-only OKD cluster; it is not the complete Red Hat OpenShift AI product. GPU execution is not installed.
