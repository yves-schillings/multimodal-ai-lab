# Whole-system deployment and CPE configuration

Version 1.1. Baseline reviewed on 7 October 2026.

This guide covers the entire Multimodal AI Lab system, including vector retrieval. It accompanies the [enterprise architecture](../../docs/enterprise-implementation-architecture.md) and [implementation backlog](../../docs/system-implementation-backlog.md). The local Compose application is implemented. Shared identity, enterprise data services, gateway, vector retrieval, generative models, CPE provisioning and OpenShift AI serving still require implementation and acceptance.

## Deployment order and acceptance

| Stage | Deployment scope | Acceptance before proceeding |
| --- | --- | --- |
| **D0 Inventory and scope** | > Record installed cluster/Operator components<br>> Choose supported versions and required services<br>> Agree workload and quality targets | > Signed inventory and compatibility matrix<br>> Named owners and acceptance dataset |
| **D1 Local baseline** | > Build immutable images<br>> Prepare model weights separately<br>> Reproduce reviewed sources and model releases | > Image/weight manifest<br>> Full vulnerability report and accepted exceptions<br>> Dated end-to-end evidence |
| **D2 OpenShift Local** | > Run existing app/MLflow/PostgreSQL manifests<br>> Retain SQLite and one worker<br>> Use port-forward only | > Pods Ready under restricted execution<br>> PVC persistence after pod replacement<br>> Repeat workflow before/after restart |
| **D3 Shared foundation** | > Trusted identity and case authorization<br>> Case PostgreSQL, objects and durable jobs<br>> Network controls and audit | > Forged/revoked/cross-case requests denied<br>> Migration and transaction tests<br>> Secure shared ingress accepted |
| **D4 Vector retrieval** | > Separate pgvector database<br>> Approved source chunks and embedding model<br>> Authorised hybrid search | > No stale/unreviewed/unauthorised citations<br>> Held-out quality and latency comparison<br>> Reindex and rollback tests |
| **D5 AI services and lifecycle** | > Required OpenShift AI components<br>> Compatible private model endpoints<br>> Gateway and MLflow/pipeline integration | > Model parity and provenance<br>> Endpoint bypass denied<br>> Model release and rollback accepted |
| **D6 Controlled project environment** | > Six project-specific control families<br>> Live routing and audit/expiry<br>> Approved provisioning and activation | > All positive/negative tests pass<br>> Independent signed activation record |
| **D7 Shared acceptance and operations** | > Complete five business workflows<br>> Backup/restore, load, retention and monitoring<br>> Clean release reproduction | > Measured requirements met<br>> Recovery and operational handover<br>> Versioned evidence |
| **D8 Approved extensions** | > Private-host variant if selected<br>> Secondary AI service<br>> Independently approved onward processing | > Each destination and hop authorised<br>> No primary/cloud direct path<br>> Outage queues/refuses |

This is dependency order, not a promise of sequential calendar duration. Local vector work can proceed while the platform is prepared. A shared rollout cannot bypass identity or CPE acceptance by declaring an Operator installed. Installing OpenShift Local tests application portability. It does not by itself install or validate the complete OpenShift AI platform.

## Baseline reproduction

Existing repository commands, from the repository root, are:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node tests/browser_core.test.mjs
docker compose -f deploy/docker/compose.yaml ps
.\.venv\Scripts\python.exe scripts/demo_e2e.py --phase before --base http://127.0.0.1:8780
docker compose -f deploy/docker/compose.yaml restart app
.\.venv\Scripts\python.exe scripts/demo_e2e.py --phase after --base http://127.0.0.1:8780
```

The end-to-end script mutates a synthetic dossier and model runs. Run it in the agreed acceptance environment. Port 8780 is the observed local application port, configurable through `LAB_HOST_PORT`. Do not delete persistent volumes to obtain a clean test. Use a separate Compose project and retained volumes for an isolated acceptance run.

The dated local evidence has 63 Python tests plus 15 subtests and 27 checks before app restart plus 8 afterwards. Those counts describe that snapshot, not guaranteed future test counts. The image scan retains 77 HIGH/CRITICAL findings without available fixes in its recorded database snapshot. A filter that ignores unfixed findings is not a remediation. Synthetic French and Dutch speech errors remain high and require a separate human-recording evaluation.

## Cluster inventory before configuration

After account, cluster access and the CLI are established, collect a sanitised inventory. These are proposed read-only checks and were not executed against a cluster for this guide:

```bash
oc version
oc get clusterversion
oc get clusteroperators
oc get subscriptions.operators.coreos.com -A
oc get clusterserviceversions.operators.coreos.com -A
oc get storageclass
oc get nodes
oc api-resources
```

When the selected release supplies the resources, inspect the OpenShift AI `DataScienceCluster` and the actual serving runtime/hardware profile objects. Use `oc api-resources` to confirm their names for that release rather than copying API kinds from a different version. Record Operator channel/version, enabled component Ready conditions, registry, storage access modes, GPU allocatable capacity, trusted identity provider and certificate chain. Do not export secret contents, user tokens or private endpoints into public evidence.

The supported product pairing and enabled components must be selected in SYS-004 and observed in SYS-002/SYS-028. Installation instructions here refer to Red Hat product documentation as a capability reference, not an assertion that a specific version is installed. [OpenShift AI installation and component management](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html-single/installing_and_uninstalling_openshift_ai_self-managed/index).

## Adapt the prepared application manifests

The existing `deploy/openshift/base` is a portability blueprint for three services. Before applying it:

1. Set immutable pushed image references for the application, its model-check init container and MLflow.
2. Choose the tested storage class and adjust PVC sizes from measured needs. Seed prepared speech weights into the model PVC and record their hash. The present init container reports absent weights but does not make missing speech weights fatal.
3. Provision the PostgreSQL credential through the approved secret mechanism. The example Compose password is local-only.
4. Verify arbitrary-UID execution, probes and network rules on the actual cluster. Include successful DNS and app/MLflow/PostgreSQL connectivity and denied unrelated ingress/egress.
5. Keep one replica and `Recreate` while the case store is SQLite and the worker is single-process. Do not use file-based SQLite as a shared multi-replica database.
6. Use a private port-forward for the synthetic portability test. Do not activate the Route overlay while the API accepts simulated actors.
7. Record deployment versions and repeat end-to-end acceptance after pod replacement. Preserve previous data volumes and capture actual results.

The prepared base supplies PVCs, Services, application/MLflow Deployments, a PostgreSQL StatefulSet, ConfigMap and some NetworkPolicies. It does not include trusted API login, project ResourceQuota/LimitRange, GPU serving, CPE automation or a vector service. Its existing limits and network YAML require review rather than an assumption of operational acceptance.

## Data deployment

**D01 case database:** migrate SQLite transactions, row handling, placeholders, schema and revision invariants to PostgreSQL. Export/import in dependency order and verify cases, documents, segments, jobs, model decisions and audit. Test concurrency and rollback before scaling workers. Preserve independent model publication evidence.

**D02 objects:** provide controlled source/derivative storage, project credentials, encryption, immutable source checksums, retention and backup. A PVC declaration requests storage. It does not supply an object-storage service or establish case authorization.

**D03 vectors:** provision a separate PostgreSQL/pgvector database and volume with migration/runtime roles. Keep MLflow credentials and tables separate. Pin extension, PostgreSQL and embedding versions. Source revisions, project/case IDs, chunk boundaries and hashes must accompany vectors. The index is derived data and inherits the source restrictions. Exact scoped search is the first baseline. An approximate index requires measured filtered-recall acceptance. [pgvector documentation](https://github.com/pgvector/pgvector).

Index updates follow an authoritative transactional outbox. Review, correction and access revocation produce idempotent indexing events. Revalidate source eligibility against D01 before returning passages even if D03 is stale or unavailable. An embedding model change creates a compatible new index generation, with a measured switch and rollback. Include deletion from indexes and backups in the retention policy.

PostgreSQL row-level security can supplement application filters when designed correctly. Runtime roles must not own protected tables or have superuser/BYPASSRLS rights. Shared service credentials and a compromised service still require a broader threat model. [PostgreSQL row security documentation](https://www.postgresql.org/docs/16/ddl-rowsecurity.html).

**D04 artifacts:** configure MLflow tracking, registry metadata and artifact access with separate privileges. Preserve run linkage and exact checksums. Model serving must consume the accepted pinned artifact rather than an unqualified latest version. Rollback changes both the application release decision and the actual endpoint revision, then verifies parity.

## AI deployment

Enable only the OpenShift AI components required by the selected deployment. Configure project storage/credentials, runtime compatibility and hardware before serving models. A pretrained model, serving container, Python library and GPU profile are distinct inventory records. Speech, embeddings and classification may use CPU where evaluation supports it. A generative model's memory and throughput requirements are established from its chosen weights/runtime, not inferred from a generic GPU label.

Select pretrained speech/embedding/language weights with licence, immutable revision, tokenizer and preparation manifest. The existing classifier is trained by the application. Prepare/download weights outside request processing and make the accepted artifacts available read-only. Deploy private endpoints and test local/served output parity, timeouts, concurrency, readiness, cancellation and rejected unauthorised requests. [Red Hat model-serving prerequisites](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploying_models/deploying_models).

The C05 gateway checks verified identity, case rights, catalogue/model revision, permitted destination/geography, resource availability and consumption quotas. Direct endpoint access must not bypass those checks. Platform model serving supplies execution. Business authorization, bounded tools, supported citations and independent human approval remain custom integration work.

## Controlled Project Environment configuration for Docker and OpenShift

A Controlled Project Environment (CPE) is an approved project perimeter with enforced isolation, attributable access and retained acceptance evidence. Docker supplies a local integration environment. OpenShift supplies the shared platform enforcement primitives. Neither an installed platform nor a running application is sufficient to declare a CPE accepted.

This chapter replaces the earlier six-item implementation grouping. Keep six control families: namespace or project perimeter, network policy, identity and policy, storage scope, audit and evidence, resource and compute. Secrets belong under identity and policy. Approved model endpoints belong under network, identity, storage integrity and resource controls. Approval, expiry and promotion govern the whole environment.

### Configuration contract before provisioning

The approved request records project identifier, owner, purpose, permitted users and services, case and dataset scope, permitted models and artifact revisions, processing location, CPU/GPU/storage needs, consumption limits, retention, expiry and recovery objectives. Each value has a named approver. Use synthetic dossiers for the initial implementation test.

Keep the approved request, configuration revision, image and model digests, test report and activation decision together. A provisioning failure leaves the CPE inactive. A material scope or policy change requires re-verification. Shared services must identify the calling project and enforce its scope, even if the service is hosted outside the project perimeter.

### The six controls on both platforms

| Control | Docker configuration | OpenShift configuration | Required evidence |
| --- | --- | --- | --- |
| **1 Project perimeter** | > Separate Compose project name, networks, volumes and ports<br>> Dedicated host or VM where the risk requires a stronger boundary | > One namespace/project per approved CPE<br>> Restricted workload admission and scoped administrative delegation | > Inventory matches the approved scope<br>> Project A cannot access project B resources<br>> Runtime cannot modify its isolation controls |
| **2 Network policy** | > Explicit service networks and internal back-end networks<br>> Loopback entry only for the local demonstration<br>> Host/VM firewall or controlled proxy for required egress | > Default-deny ingress and egress for every pod<br>> Explicit service/port allowlist and approved ingress<br>> Controlled external destinations | > Approved paths work<br>> Cross-project, internet and model-gateway bypass paths fail |
| **3 Identity and policy** | > Trusted authentication before shared access<br>> Application case/action authorization<br>> Per-service credentials and protected secret files | > Identity federation, scoped ServiceAccounts and RBAC<br>> Case/action policy enforced in the API and workers<br>> Restricted Secrets and credential rotation | > Forged, expired, revoked and wrong-project identities fail<br>> No self-approval or unauthorized role/secret changes |
| **4 Storage scope** | > Separate persistent project volumes and data credentials<br>> Prepared weights mounted read-only<br>> Protected backups with project lineage | > Scoped PVCs, database roles and object buckets<br>> Separate vector/index and artifact scopes<br>> Retention, encryption and backup policies | > Raw sources, vectors and backups are inaccessible across projects<br>> Restore preserves source, review and approval revisions |
| **5 Audit and evidence** | > Application decisions forwarded to a separately controlled collector<br>> Bounded container logs for operations | > Application decision events plus cluster API audit<br>> Restricted central collection, retention and correlation | > Actor, project, case, decision and revision reconstruct a run<br>> Runtime cannot delete the authoritative audit record |
| **6 Resource and compute** | > Per-container CPU, memory and PID limits<br>> Host disk monitoring and service consumption accounting | > ResourceQuota and LimitRange<br>> Approved CPU/GPU scheduling and storage requests<br>> Application duration/request/token accounting | > Limits are enforced under concurrent load<br>> Exhaustion queues or refuses work<br>> No unauthorized capacity fallback |

### Docker configuration procedure

#### 1 Establish two separate test projects

Use projects `cpe-a` and `cpe-b` with synthetic data and different loopback ports. The existing named volumes acquire the Compose project prefix when `-p` is used. Do not add an explicit shared volume name or an `external` network that reconnects the two projects unintentionally. Do not reuse production or current demonstration volumes.

The current Compose file fixes MLflow's host port at 5000. Starting two copies without changing that publication causes a port conflict. In the revised CPE composition, keep MLflow internal and access it only through an authorized administration path. Set the application port separately for each test project. A Compose project name organizes resources but does not provide namespace RBAC or exclude the Docker host administrator.

#### 2 Define the allowed service flows

For the present three-service application, allow browser to application, application to MLflow, and MLflow to its PostgreSQL metadata database. The application case store remains SQLite on its own volume. PostgreSQL in this composition is not the case database or vector store.

Replace the implicit all-service network with separate application-to-MLflow and MLflow-to-PostgreSQL networks. Use internal networks for the back-end paths and publish no database ports. Prepare weights during a separate controlled preparation step. Do not permit runtime internet access simply to download missing weights.

Docker networks are not a complete directional policy engine. Services on the same network can generally connect to each other. A multi-homed service, published host port or host administrator can create other paths. Prove outbound and cross-project refusal using actual probes, and add host/VM firewall controls or a controlled proxy where required. Use a dedicated VM/host when a shared daemon does not meet the approved boundary.

#### 3 Implement trusted identity and secrets

The existing persona selector is a simulated identity mechanism. Keep the present application on loopback until trusted authentication and case/action checks are implemented. A reverse proxy login alone does not prevent a caller from selecting another application actor or bypassing the proxy. The API must validate trusted identity, derive the actor from it and reject untrusted actor overrides. Workers revalidate authorization before delayed jobs run.

Replace the demonstration database password default and command-line credential construction with unique secret-file delivery and a server-side loader. PostgreSQL can consume its supported password-file mechanism. MLflow needs an entrypoint or other approved configuration that reads the secret without logging it. Declare each Compose secret only for the services that need it. Protect its source file with host permissions and keep it outside version control. Compose file-backed secrets are mounted files, not an encrypted enterprise secret-management system.

Test rotation, rejection of the old credential, denied project-B secret access and redacted operational logs. Do not mount the Docker socket into application containers.

#### 4 Separate data and restrict the runtime

Use project-specific writable data and artifact volumes, read-only prepared model weights, non-root execution, dropped capabilities, no privilege escalation and the default supported seccomp profile. After testing all required writable paths, make the root filesystem read-only and provide bounded temporary storage for decoding and OCR. Never add privileged mode or host filesystem mounts to resolve a permissions failure.

Pin release images by digest after building and scanning them. Record accepted findings and compensating controls. The dated container scan has unresolved findings. A filter that ignores unfixed findings is not a risk acceptance.

Test both direct filesystem access and application access. A distinct volume does not replace case rights. Backups, exports, derived chunks and any later vector service must retain the same project boundary. Restore into an isolated test project and verify reviewed revisions and source hashes before reopening access.

#### 5 Add audit and actual limits

Set and inspect CPU, memory and PID limits for each container. Allocate measured capacity rather than copying an arbitrary number from a sample. These are container limits, not a project-wide GPU, disk or inference-token budget. Add disk capacity monitoring and transactional application admission for audio duration, job count, requests and model tokens where relevant.

Forward project decision events to an independently controlled sink. Container stdout and a mutable SQLite audit table are useful local records but are not tamper-resistant central evidence. Record identifiers and decisions without copying raw statements, prompts or credentials into routine logs. Test collector failure under an agreed fail-closed or protected-buffer policy.

#### 6 Accept the local CPE design

Run the two-project negative tests in the acceptance matrix below. Repeat the business workflow and model release before and after application restart. Verify persistence, quota exhaustion, secret rotation and expiry. Record what Docker cannot prove, including OpenShift RBAC, SCC admission and the selected CNI's network enforcement.

### OpenShift and OpenShift AI configuration procedure

#### 1 Inventory the actual platform

Record cluster release, network implementation, storage classes, identity provider, audit configuration and available node/GPU capacity. Separately record the OpenShift AI Operator, enabled components, supported model-serving runtimes and hardware profiles. Select documentation for those exact supported versions. No installed cluster or OpenShift AI inventory has been verified for this machine.

OpenShift AI projects provide a namespace-based workspace with permissions and connections. They do not automatically enforce the custom case API's dossier permissions, model release decision or complete CPE approval lifecycle.

#### 2 Create the project perimeter before workloads

Create `cpe-a` and `cpe-b` through the authorized provisioner. Attach owner, scope, policy revision and expiry metadata. Create separate application, worker, model-serving and provisioning service accounts where the implementation separates these responsibilities. Disable automatic service-account token mounting when a workload does not call the Kubernetes API. Use audience-bound, time-limited tokens where an API token is needed.

Apply the supported restricted security context constraints, non-root execution, dropped capabilities and seccomp. Prevent runtime and ordinary project users from editing NetworkPolicies, RoleBindings, quotas, admission settings and protected labels. A broad project-admin grant can invalidate the isolation design. Cluster/host administration remains a privileged boundary outside the application tenant.

#### 3 Apply default deny and then the flow allowlist

Start with a policy selecting all pods and both ingress and egress. Add only the approved workload-to-workload flows, required DNS, identity and telemetry paths. Confirm DNS ports and selectors for the actual network deployment. Standard NetworkPolicy does not authorize a model version or generally provide dynamic domain-name egress controls. Combine supported platform egress facilities and application policy where necessary.

Initially use authorized port-forward for the local application test. NetworkPolicy acceptance must use pod-to-pod probes as well. A successful port-forward is not proof that the network policy permits or denies ordinary traffic. Publish a TLS Route only after trusted API authentication, case authorization and ingress tests pass.

For private model serving, allow only the approved gateway/workload identities and paths. Shared endpoints outside the CPE namespace need their own access controls. Test a direct bypass attempt, an unapproved model revision and an unavailable approved destination.

#### 4 Bind scoped storage and credentials

Use an accepted StorageClass and access mode. Keep the current SQLite application at one replica with Recreate strategy for the portability test. Migrating the case store to PostgreSQL and scaling workers are separate changes with transaction and concurrency tests.

For the shared system, scope case PostgreSQL roles, source buckets, pgvector indexes and model artifacts per project. Enforce the boundary in the actual database/object service, not solely by namespace placement. Use credentials scoped to the project and operation. Configure secret-store protection, certificate trust, rotation, backups, retention and restore tests. Read-only weights must match the approved artifact digest.

#### 5 Configure audit, quotas and approved compute

Retain cluster API audit and application decision audit as separate evidence streams. Cluster audit records resource actions, but not every dossier read, search or model approval. Send those application decisions to the restricted central sink, with correlation IDs and defined retention.

Apply ResourceQuota and LimitRange, including applicable CPU, memory, GPU and storage resources. Assign approved node pools through administrator-controlled scheduling and hardware profiles. A GPU resource count is not, by itself, proof of GPU memory isolation between projects. Accept the selected dedicated, partitioned or shared hardware configuration through its own tests. Prevent a tenant from changing scheduling labels or tolerations to reach another pool.

Implement audio-duration, request and inference-token accounting in the API/gateway. Kubernetes compute quotas cannot replace these consumption rules. Quota failures must preserve policy and refuse or queue work rather than select an unauthorized pool.

#### 6 Make activation an evidenced decision

Provisioning, model preparation and acceptance are distinct steps. A missing required model artifact must fail the relevant service readiness or activation gate. Passing `/health/live` or obtaining Ready pods is not sufficient.

Approve the exact configuration and evidence only after all mandatory tests pass. Enable approved ingress and jobs, record the activation decision and monitor ongoing drift. Expiry first disables access and new jobs, revokes credentials and applies the approved retention or legal-hold procedure. Do not automatically delete persistent volumes. Promotion to a shared AI capability requires a separate release and ownership decision.

### Gaps found in the present implementation

| Inspected baseline | Required change or verification |
| --- | --- |
| **Docker default network** | > Segment actual service paths<br>> Verify egress and cross-project refusal |
| **Docker password and MLflow host port** | > Replace the demonstration password/command-line construction<br>> Avoid a shared fixed administration port in multiple CPE copies |
| **Simulated actors** | > Implement trusted API identity and case/action policy<br>> Keep shared/public ingress disabled until accepted |
| **OpenShift namespace policies** | > Extend the current ingress denial and service-specific egress to all-pod default-deny ingress and egress<br>> Protect policy administration |
| **OpenShift service accounts and quotas** | > Add explicit scoped workload accounts/RBAC and namespace quota/LimitRange<br>> Disable unused API tokens |
| **Model initialization** | > The existing weight check reports absence without failing<br>> Add an explicit required-service preparation/readiness gate |
| **Audit and recovery** | > Add independent central decision evidence<br>> Prove rotation, retention, restore and expiry |
| **OpenShift AI and vectors** | > Install/configure the selected serving components and supported models<br>> Implement the separate reviewed-source pgvector index and its access/invalidation tests |

The existing OpenShift files are a portability starting point, not this complete CPE configuration. They have not passed cluster acceptance. The original architecture diagrams remain preserved. Their six-item setup grouping must be read alongside this six-control mapping, not treated as exact control-family equivalence.

### Acceptance matrix

| Test | Expected result and evidence |
| --- | --- |
| **Approved project-A workflow** | > Authentication, bounded upload, review, permitted search and separate approval succeed<br>> Correlated project and revision evidence retained |
| **Project-B identity/data/network attempt** | > API, storage and network denial<br>> No source, vector, artifact or secret disclosed |
| **Runtime changes its boundary** | > Role, policy, protected-label and quota mutation denied |
| **Unapproved endpoint or direct bypass** | > Invocation refused<br>> Permitted pinned model still works |
| **Quota exhausted under concurrent jobs** | > New work queues or refuses without overspending or unauthorized fallback |
| **Credential rotation or revoked membership** | > Old authority rejected<br>> Delayed work revalidates access |
| **Source edited or review revoked** | > Old citations and draft approval become ineligible<br>> Index lag cannot disclose revoked material |
| **Restart and isolated restore** | > Sources, reviewed versions and accepted model selection persist consistently |
| **Audit sink interrupted** | > Agreed refusal or protected buffering operates<br>> No silently unaudited accepted action |
| **Project expired or suspended** | > Access and new work disabled<br>> Retention/hold policy respected and closure recorded |

Use the same semantic tests for Docker and OpenShift, plus platform-specific resource and network probes. Save configuration and image/model digests with each run. Acceptance means the agreed controls passed on the named environment. It does not mean that a sample manifest is universally perfect.

### References

- [Docker Compose networks](https://docs.docker.com/reference/compose-file/networks/)
- [Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/)
- [Docker Compose service configuration](https://docs.docker.com/reference/compose-file/services/)
- [OpenShift project quotas](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/building_applications/quotas)
- [OpenShift network security](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/observability/network_security/index)
- [OpenShift audit policy](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/security_and_compliance/audit-log-policy-config)
- [OpenShift AI project access](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_on_projects/managing-access-to-projects_projects)
- [OpenShift AI connections](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_on_projects/using-connections_projects)

These release-specific references support the design. Select the matching documentation after the installed platform inventory is agreed.


## Rollback, recovery and final acceptance

- Record approved commit, image digests, model/index revisions, migrations, policy bundle, cluster versions and acceptance evidence in one release manifest.
- Disable new ingress/jobs before an emergency data/schema rollback. Restore to an isolated namespace and verify revision/approval lineage before reopening access.
- Retain previous model and compatible index generations for controlled rollback. Stop failed indexing consumption and replay idempotent events after recovery.
- Restore cases, originals, registry metadata, artifacts and audit at a documented consistent point. Rebuild vectors from eligible sources when necessary.
- Verify the five business workflows, negative access tests, unsupported-answer abstention, pod/worker failures, permission changes, quota exhaustion, source edits, model rollback, retention and expiry.
- Agree and measure recovery/availability requirements. Obtain operational ownership and a second engineer's clean-release reproduction before claiming the shared implementation accepted.

External extensions remain a separately approved deployment. The only permitted chain is Primary AI Lab to Secondary GPU Data Center, followed by independently approved Secondary GPU Data Center to Sovereign Cloud. Keep direct primary/cloud routes absent in gateway, network and tests.
