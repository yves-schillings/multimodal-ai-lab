# Controlled Project Environment configuration across platforms

This guide identifies what the operator configures, which repository files provide it, and what has actual deployment acceptance. It addresses application and platform engineers reproducing the fictional laboratory. A Controlled Project Environment (CPE) remains inactive until its required controls pass verification.

## Execution mode and hosting location

Direct Python execution and Docker Compose can run on the **same computer**. Python runs the application in a language environment. Compose runs the application and its services as containers. A private server is a hosting location that can use either execution mode.

Kubernetes orchestrates containers. **OKD** is the community Kubernetes distribution upstream of Red Hat OpenShift Container Platform. OpenShift AI adds tools for model development, workbenches, serving and pipelines to a supported OpenShift platform. Its enabled capabilities depend on the selected supported release and configuration.

The six architecture views describe execution modes, hosting platforms and external service extensions. They do not require six machines or a mandatory migration sequence. Secondary GPU hosting and an approved onward cloud route remain governed service extensions.

## Current scope and differences

| Environment | Verified or executable scope | Additional work |
| --- | --- | --- |
| Direct Python | The local application has workflow and model-gate evidence. | Shared operating controls and cluster isolation require separate acceptance. |
| Docker Compose | The application, MLflow, PostgreSQL metadata and persistent volumes passed workflow and restart checks. | The current Compose file uses simulated actors. Complete separate Docker CPEs require additional configuration and tests. |
| Kubernetes reference | The guide identifies native cluster policy mechanisms. | No separate vanilla Kubernetes deployment is accepted. OpenShift-specific manifests need adaptation. |
| Local OKD | Two CPEs passed isolation, lifecycle, browser and offline retrieval exercises. Selected free AI components have separate acceptance. | Full OpenShift AI, organizational identity, production ingress, GPU execution and distributed training remain outside this deployment. |
| OpenShift with OpenShift AI | The architecture identifies supported platform and AI components. | The complete product is a target, with installation and end-to-end acceptance still required. |

## Six controls in every environment

1. **Identities and access rights.** The identity service authenticates the user, and the application checks the permitted case/document action. Cluster Roles and RoleBindings govern platform-resource operations.
2. **Network policies.** The operator allows required application, database and gateway paths and blocks other paths. A Kubernetes network plugin must actually implement NetworkPolicy.
3. **Resource and consumption quotas.** Container limits bound execution, quotas bound admitted allocations, and the gateway bounds requests. Storage admission does not establish physical disk isolation.
4. **Project secrets.** The operator supplies distinct protected credentials to services, scopes administrative access and verifies rotation and revocation. Operational values stay outside Git.
5. **Controlled storage.** Separate stores and database roles protect project records. Application rights, current reviewed revisions and tested restoration remain necessary.
6. **Approved model endpoints.** The private gateway checks project credentials, model identity, approved operations and request limits before forwarding inference.

Audit, approval, activation, revocation and expiry apply across all six. A namespace alone does not establish the full boundary.

## Docker Compose

The executable baseline is [compose.yaml](../../deploy/docker/compose.yaml). It starts the application and worker, MLflow and PostgreSQL with three named volumes and loopback host ports. PostgreSQL stores MLflow metadata, while case records stay in SQLite on the application volume. This baseline retains simulated actors and a demonstration database credential.

The operator can validate the file without starting workloads:

```powershell
docker compose -f deploy/docker/compose.yaml config --quiet
```

For a scheduled reproduction, the existing acceptance procedure is:

```powershell
python scripts/prepare_speech.py
docker compose -f deploy/docker/compose.yaml up -d --build --wait
python scripts/demo_e2e.py --phase before --base http://127.0.0.1:8770
docker compose -f deploy/docker/compose.yaml restart app
python scripts/demo_e2e.py --phase after --base http://127.0.0.1:8770
```

Choose an approved available host port when another instance uses 8770. The recorded baseline passed 27 checks before restart and 8 afterwards. These results do not verify separate Docker CPE isolation.

For a full Docker CPE, the operator must add authenticated users, separate service networks, unique read-only credentials, project stores, container limits and a scoped gateway. Compose supports the required container configuration, but the current file does not implement this complete isolated deployment. The privileged daemon remains a shared host boundary.

## Kubernetes reference

The operator needs a cluster with a policy-enforcing network plugin and approved storage. The following configuration is a reference, not an accepted vanilla Kubernetes bundle:

| Control | Platform objects | Application responsibility |
| --- | --- | --- |
| Identity | ServiceAccount, Role and RoleBinding limit resource operations. | Named accounts and record/action grants govern data use. |
| Network | Default-deny NetworkPolicy permits selected endpoints and ports. | Service authorization still checks every allowed request. |
| Resources | ResourceQuota, LimitRange and container requests/limits constrain allocations. | File, request and storage-admission budgets bound work. |
| Secrets | Scoped Secret mounts supply protected credentials. | The service rejects revoked credentials. |
| Storage | PersistentVolumeClaim selects approved storage. | Database roles and record checks protect content. |
| Models | Private Services expose approved serving endpoints. | The gateway checks pinned model identity and allowed operations. |

The [OpenShift base](../../deploy/openshift/base/kustomization.yaml) is a source for review. Its DNS selectors and ports, security assumptions, storage classes and image-build path need adaptation to a different cluster. Actual pod-to-pod allow/deny tests must pass. An administrative port-forward does not prove network-policy enforcement.

Compared with Kubernetes core, OKD provides OpenShift project administration, Security Context Constraints and Routes. A team can assemble equivalent integrations separately, with its own operating responsibility and acceptance.

## Local OKD

[local-cpe.md](local-cpe.md) gives the executable procedure. [local_cpe.py](../../scripts/local_cpe.py) provisions protected project resources and implements the lifecycle gate. It permits only the approved local cluster and refuses to overwrite existing environments.

For **new** projects, the operator provisions both before testing cross-project denial:

```powershell
$taskKubeconfig = '<approved-local-admin-kubeconfig-path>'
$taskPrivateDir = '<protected-cpe-configuration-path>'
python scripts/local_cpe.py provision --project a --admin-kubeconfig $taskKubeconfig --private-dir $taskPrivateDir --approval-reference fictional-project-a
python scripts/local_cpe.py provision --project b --admin-kubeconfig $taskKubeconfig --private-dir $taskPrivateDir --approval-reference fictional-project-b
```

Existing environments follow renewal or profile-change procedures rather than reprovisioning. Renewal uses **`--approval-reference`** and optionally `--hours`. It preserves scope and data, but requires fresh verification and activation.

[extend_cpe_rag.py](../../scripts/extend_cpe_rag.py) revokes projects before changing protected profiles. It provisions separate PostgreSQL/pgvector roles and project credentials for the shared gateway. **BGE-M3** creates search vectors, and **Qwen3:4b** generates answers from supplied passages using pinned local weights.

```powershell
python scripts/extend_cpe_rag.py --admin-kubeconfig $taskKubeconfig --private-dir $taskPrivateDir
python scripts/cluster_cpe_rag_acceptance.py --admin-kubeconfig $taskKubeconfig --private-dir $taskPrivateDir
```

These acceptance commands mutate owned project state and replace selected workloads. Schedule them separately from user work and retain private evidence. Recorded acceptance includes 39 verification checks, 13 lifecycle checks and 6 browser checks per project. Approval is bounded and expires. A historical report does not prove that a project remains active indefinitely.

[local-ai-platform.md](local-ai-platform.md) describes separately accepted JupyterLab, private KServe classifier prediction, a native training Job and MLflow evidence. The complete OpenShift AI dashboard, pipelines, distributed training and GPU execution remain outside that acceptance.

## Red Hat OpenShift with OpenShift AI

**OpenShift hosts applications. OpenShift AI adds the selected AI tools.** The application still owns case and document permissions.

1. The platform administrator inventories the supported release combination, networking, storage and planned AI components.
2. The administrator installs the supported Operators and enables approved workbench, serving and pipeline components.
3. The identity administrator integrates organizational authentication and platform roles. The application owner separately configures record/action rights.
4. The operator configures project policies, quotas, workload limits, protected secrets and scoped storage. Model serving uses approved runtimes and pinned artifacts.
5. The model engineer uses workbenches and pipelines and retains evaluation evidence. An authorized reviewer controls model activation.
6. The responsible teams test allowed and denied access, revocation, capacity limits, persistence and recovery before activation.

The full product, shared production hosting and GPU execution remain **targets**. Operator installation or a dashboard does not itself establish a safe deployment.

## Authorized searches across projects

Some Officers may access several projects, with different combinations for each user. Others may access only one. A common portal derives its working view from the authenticated person's current grants. This view does not require a dedicated namespace or server per user.

The proposed coordinator must query approved source endpoints with verified user context. Each source must enforce record/action permissions before retrieval. Models receive permitted reviewed passages with project, case, document and revision citations. Direct arbitrary project-to-project connections remain denied.

The independent grant registry and delegated cross-project retrieval path are **not implemented or accepted** in the inspected lab. Existing CPE evidence verifies isolated operations, not this extension.

## Sources

- [Recorded Docker baseline](../evidence/2026-10-07-closing/README.md).
- [Recorded local AI extension acceptance](../evidence/2026-10-09-local-ai-extensions/README.md).
- [Docker Compose networks](https://docs.docker.com/reference/compose-file/networks/).
- [Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/).
- [Kubernetes multi-tenancy](https://kubernetes.io/docs/concepts/security/multi-tenancy/).
- [Kubernetes NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/).
- [OpenShift AI supported configurations](https://access.redhat.com/articles/rhoai-supported-configs).
- [OpenShift AI projects, workbenches and pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/getting_started_with_red_hat_openshift_ai_self-managed/overview-for-getting-started_get-started).
