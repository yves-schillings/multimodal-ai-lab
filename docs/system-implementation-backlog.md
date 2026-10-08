# Multimodal AI Lab implementation backlog

Version 1.3. Planning and control mapping reviewed 2026-10-08.

The 60 work items cover the five use cases, eight responsibility layers and all 28 application, AI service, data, governance and hosting components. Existing local features remain the baseline; the work items specify the remaining shared-system implementation and its acceptance.

The tracking workbook is Multimodal AI Lab - System Implementation Backlog - v 1.3.xlsx. This text reproduces its work-item names, draft effort, prerequisites, role allocation, architecture decision references and acceptance. Named assignments and confirmed capacity remain open in SYS-060. Draft effort is not an agreed delivery schedule.

[Enterprise architecture and installed-component inventory](enterprise-implementation-architecture.md) explain products and the observed baseline. The [deployment and Controlled Project Environment (CPE) configuration guide](../deploy/system/README.md) defines Docker and Red Hat OpenShift configuration and negative acceptance tests.

The Summary worksheet counts the 60 work items, local proof records, priority gates, dependencies and draft effort. It also names the eleven items on the longest activation dependency chain. Planning now allocates draft effort by phase and primary profile. Enter confirmed availability only in the five existing profile capacity inputs; phase estimates link to those inputs. Blank capacity remains unconfirmed, and zero capacity cannot produce a finish date. The same availability is assumed across phases; elapsed weeks exclude waits and other dependency branches.

## Reading the identifiers

SYS identifies an implementation work item, followed by its descriptive name. ADR identifies an Architecture Decision Record from the article. Registry families are C (application component), M (AI model or service), D (data store), A (access and governance control), and H (hosting environment). L1–L8 identify responsibility layers. These codes are references rather than product names.

## Phases, priorities and effort

A phase P0–P5 is a delivery stage. Priority P0 means one of the 15 first-activation control gates; priority P1 includes their mandatory prerequisites and other shared delivery work; priority P2 is separately approved extension work. Lower priority does not remove a required control. The first CPE activation has 38 mandatory dependencies.

Draft size assumptions are S: 1–2, M: 3–5 and L: 6–12 person-days. One person-day is one person working one day. Parallel work and available profiles determine elapsed duration; procurement, access waits and dataset collection lead time are excluded. No calendar finish date is committed.

| Phase | Scope | Items | Draft person-days | Exit condition |
| --- | --- | ---: | ---: | --- |
| P0 | Scope and architecture | 6 | 18–30 | Scope, installed inventory, compatibility and thresholds approved |
| P1 | Trusted foundation | 9 | 39–73 | Trusted identities, workloads, secrets, images and network paths accepted |
| P2 | Shared core and vector retrieval | 13 | 63–121 | Case/data migration, durable jobs and authorised vector retrieval accepted |
| P3 | AI services and CPE | 20 | 90–170 | Pinned private serving, release gate and complete CPE acceptance pass |
| P4 | Acceptance and operations | 8 | 33–61 | Five workflows, restore, load, retention and independent handover pass |
| P5 | Approved extensions | 4 | 21–41 | Each secondary/onward hosting extension separately approved and accepted |

**Total draft effort: 264–496 person-days.** This range covers all six phases; it is not the duration of the first CPE activation.

## First CPE activation dependency path

The preliminary longest dependency chain uses each item’s midpoint estimate and has 74 person-days of sequential work. It is not a resource-levelled calendar and is not the complete activation scope. All 38 dependencies, including other branches, must pass.

| Order | Work item | Draft person-days |
| ---: | --- | ---: |
| 1 | [SYS-001: Confirm the system scope](#sys-001-confirm-the-system-scope) | 3–5 |
| 2 | [SYS-002: Record installed platform capabilities](#sys-002-record-installed-platform-capabilities) | 3–5 |
| 3 | [SYS-004: Select supported platform and runtimes](#sys-004-select-supported-platform-and-runtimes) | 3–5 |
| 4 | [SYS-006: Prepare the cluster and storage prerequisites](#sys-006-prepare-the-cluster-and-storage-prerequisites) | 6–12 |
| 5 | [SYS-010: Create scoped workload identities](#sys-010-create-scoped-workload-identities) | 3–5 |
| 6 | [SYS-011: Enforce case and action authorization](#sys-011-enforce-case-and-action-authorization) | 6–12 |
| 7 | [SYS-017: Implement durable concurrent job execution](#sys-017-implement-durable-concurrent-job-execution) | 6–12 |
| 8 | [SYS-022: Index only approved source revisions](#sys-022-index-only-approved-source-revisions) | 6–12 |
| 9 | [SYS-043: Configure controlled project storage](#sys-043-configure-controlled-project-storage) | 6–12 |
| 10 | [SYS-045: Enforce fail-closed routing across the CPE](#sys-045-enforce-fail-closed-routing-across-the-cpe) | 3–5 |
| 11 | [SYS-047: Verify all controls before activation](#sys-047-verify-all-controls-before-activation) | 6–12 |

The workbook’s Activation path worksheet retains the complete dependency closure. Changing sizes or prerequisites requires recalculating the path.

## Profiles and capacity

| Primary profile | Draft person-days | Capacity |
| --- | ---: | --- |
| Architecture and security | 24–40 | Unconfirmed |
| Application engineering | 81–157 | Unconfirmed |
| Platform and data | 90–168 | Unconfirmed |
| AI and MLOps | 45–87 | Unconfirmed |
| Quality assurance | 24–44 | Unconfirmed |

Every item is counted once under its primary profile. Specialist review still needs a named allocation. Architecture leadership and delivery coordination share the same person’s availability.

## Controlled Project Environment control mapping

The six control families are Namespace, Network policy, Identity & policy, Storage scope, Audit & evidence, and Resource & compute. Secrets, rotation and revocation sit under identity and policy; private pinned model endpoints span network, identity, artifact integrity and compute. Enforced routing and approved activation/expiry apply across all six families. These controls do not imply that a CPE has been deployed or accepted.

## Completion and retained evidence

The seven local proof records in the Local evidence worksheet are Done. They do not close broader shared-system acceptance. The implementation list has 59 Open items and SYS-060 In progress: draft planning exists, but people, capacity and planning approval are not confirmed. An item becomes Done only when all of its acceptance requirements have retained evidence.

Dated [fresh-checkout evidence](evidence/2026-10-07-fresh-clone/README.md) covers the same workstation with warm caches and a separate Compose project on fresh volumes. [Closing evidence](evidence/2026-10-07-closing/README.md) includes synthetic speech measurements and scanned-PDF extraction. Neither establishes a clean-machine release, hosted CI, representative human speech quality or OpenShift acceptance.

## Phase P0: Scope and architecture

### SYS-001: Confirm the system scope

- **Components and layers:**
  - C01-C05 M01-M08 D01-D04 A01-A05 H01-H06.
  - L1-L8.
- **Observed baseline:** Five use cases and eight responsibility layers documented.
- **Implementation:** Confirm actors, sensitive-data classes, volumes, languages, latency, availability, retention and rollout environments.
- **Owner role:** Lead Enterprise and AI Architect.
- **Primary profile:** Architecture and security.
- **Prerequisites:** None.
- **Acceptance:**
  - Every use case has an owner and measurable acceptance thresholds.
  - Local, shared and extension scopes are distinct.
- **Expected evidence:** Signed scope and non-functional requirements.
- **Planning:** Priority P0; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-002: Record installed platform capabilities

- **Components and layers:**
  - H01-H06.
  - L6.
- **Observed baseline:**
  - Docker stack active.
  - No configured Kubernetes context.
  - OpenShift not confirmed.
- **Implementation:** Inventory cluster version, Operators, enabled AI components, storage, GPU nodes, runtimes, identity and subscriptions when cluster access exists.
- **Owner role:** Platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-001.
- **Acceptance:** Each platform entry has a dated observation, version and Ready condition or an explicit absent/unverified state.
- **Expected evidence:** Sanitised inventory with image digests and Operator conditions.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-003: Approve component contracts and data flows

- **Components and layers:**
  - C01-C05 M01-M08 D01-D04 A01-A05.
  - L1-L8.
- **Observed baseline:**
  - Local registry exists.
  - Some registry descriptions lag newer OCR and speech evidence.
- **Implementation:**
  - Define interface contracts, trust boundaries, revision lineage and accountable owners.
  - Preserve existing diagrams.
- **Owner role:** Enterprise and AI architect.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-001.
- **Acceptance:** All 28 registry components mapped to business capability, product, environment, owner and remaining work.
- **Expected evidence:** Component and interface register.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-004: Select supported platform and runtimes

- **Components and layers:**
  - H03 H04 M01 M04 M06.
  - L4 L6.
- **Observed baseline:**
  - OpenShift manifests prepared.
  - No platform/model version selected.
- **Implementation:** Confirm a supported OpenShift/OpenShift AI pairing, Operator channels, CPU/GPU compatibility, model licences and deployment mode.
- **Owner role:** Platform engineer and AI model engineer.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-002, SYS-003.
- **Acceptance:**
  - Compatibility matrix cites the chosen release documentation.
  - Optional components are explicitly enabled or excluded.
- **Expected evidence:** Version and compatibility decision.
- **Planning:** Priority P0; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-005: Define release and quality gates

- **Components and layers:**
  - A02 A03 A04 M01 M03 M04 M05 M06 M08.
  - L4 L7 L8.
- **Observed baseline:** Local tests, synthetic speech and Compose evidence exist.
- **Implementation:** Define acceptance datasets, leakage/abstention checks, security exception process, performance targets and evidence retention.
- **Owner role:** QA lead and security architect.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-001, SYS-003.
- **Acceptance:**
  - Thresholds agreed before evaluation.
  - Synthetic evidence is labelled separately from representative quality.
- **Expected evidence:** Acceptance catalogue and evidence template.
- **Planning:** Priority P0; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-060: Record decisions and size the remaining work

- **Components and layers:**
  - C01-C05 M01-M08 D01-D04 A01-A05 H01-H06.
  - L1-L8.
- **Observed baseline:**
  - Draft S/M/L effort ranges and role allocation prepared.
  - Named people, capacity and planning approval not confirmed.
- **Implementation:** Resolve baseline vs optional broker/RAG/pipeline choices, assign people, refine size into effort and plan parallel dependencies.
- **Owner role:** Lead Enterprise and AI Architect.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-001, SYS-003, SYS-005.
- **Acceptance:**
  - Named owners, bounded work estimates, capacity and dependencies agreed.
  - No unsupported calendar or budget promise.
- **Expected evidence:** Implementation schedule and decision log.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** In progress. Named assignment: unconfirmed.

## Phase P1: Trusted foundation

### SYS-006: Prepare the cluster and storage prerequisites

- **Components and layers:**
  - H04.
  - L6.
- **Observed baseline:**
  - OpenShift Local prerequisites documented.
  - Not installed.
- **Implementation:** Obtain account/pull secret, establish cluster access, registry, storage class, DNS, certificates, time sync and backup destination.
- **Owner role:** Platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-004.
- **Acceptance:**
  - Cluster/API reachable and healthy.
  - PVC, registry pull and certificate checks pass.
  - Record resource capacity.
- **Expected evidence:** Cluster bootstrap report.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-007: Build reproducible deployable images

- **Components and layers:**
  - H02 H04 M01 M03 M08.
  - L4 L6 L8.
- **Observed baseline:** Docker build and arbitrary-UID portability checks exist.
- **Implementation:** Pin image bases and model revisions, produce SBOM, scan both images, publish immutable digests to the approved registry.
- **Owner role:** DevOps engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-004, SYS-005.
- **Acceptance:**
  - Both images support restricted execution.
  - Full vulnerability report retained.
  - Exceptions reviewed rather than silently hidden.
- **Expected evidence:** Build manifest, SBOM and scan reports.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01, ADR-05.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-008: Run hosted integration and release checks

- **Components and layers:**
  - H02 H04 M08.
  - L8.
- **Observed baseline:** Workflow prepared, hosted run not evidenced.
- **Implementation:** Validate workflow on approved publication, run all suites, publish evidence and create a versioned release after approval.
- **Owner role:** DevOps engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-007.
- **Acceptance:**
  - Hosted workflow result is tied to the release commit and image digests.
  - Filtered scan is distinguished from the full findings.
- **Expected evidence:** CI run and release manifest.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01, ADR-05.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-009: Replace simulated users with trusted identity

- **Components and layers:**
  - A01 C01 C02.
  - L1 L2 L3 L7.
- **Observed baseline:** Actor selector is a simulation.
- **Implementation:** Implement OIDC login, token issuer/audience/expiry verification, logout, secure session handling and user mapping.
- **Owner role:** Backend and identity engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-004, SYS-005.
- **Acceptance:**
  - Forged, expired and wrong-audience tokens fail.
  - Caller-supplied actor IDs cannot impersonate another user.
- **Expected evidence:** Identity integration and negative tests.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-010: Create scoped workload identities

- **Components and layers:**
  - A01 C03 C05 M08.
  - L3 L4 L7.
- **Observed baseline:** No scoped production workload identities.
- **Implementation:** Define service accounts, restricted Roles/RoleBindings and least-privilege credentials for jobs, serving and data services.
- **Owner role:** Platform and security engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-006, SYS-009.
- **Acceptance:**
  - Every workload has its own identity.
  - Unauthorised namespace/data/model actions fail.
  - No routine cluster-admin identity.
- **Expected evidence:** RBAC matrix and denied-action tests.
- **Planning:** Priority P0; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-011: Enforce case and action authorization

- **Components and layers:**
  - A02 C02 C03 C05 D01 D03.
  - L3 L4 L5 L7.
- **Observed baseline:** Case access checked locally with simulated actors.
- **Implementation:** Bind existing rules to trusted identity and authorise every read, upload, review, search, export and asynchronous job.
- **Owner role:** Backend and security engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-009, SYS-010.
- **Acceptance:**
  - Cross-case and direct-endpoint attempts fail before data retrieval or inference.
  - Worker rechecks revoked permissions.
- **Expected evidence:** Authorization matrix and API tests.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-04, ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-012: Preserve revision and separate approval rules

- **Components and layers:**
  - A03 C01 C02 D01.
  - L1 L3 L5 L7.
- **Observed baseline:** Review, separate approval and invalidation implemented locally.
- **Implementation:**
  - Carry revision checks and reviewer separation into shared APIs and concurrency.
  - Add permission-revocation and stale-write tests.
- **Owner role:** Backend engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-011.
- **Acceptance:**
  - Source edits invalidate dependent approval.
  - Self-approval, stale writes and revoked reviewer access fail.
- **Expected evidence:** Revision/approval integration report.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-04, ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-013: Configure project credentials and rotation

- **Components and layers:**
  - A01 H04 D01 D02 D03 M08.
  - L5 L6 L7.
- **Observed baseline:** Prepared PostgreSQL Secret references, no rotation workflow.
- **Implementation:** Provision secret store/namespace Secrets, mounted credentials, certificate trust, rotation and workload restart procedures.
- **Owner role:** Platform and security engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-006, SYS-010.
- **Acceptance:**
  - No credentials in images, ConfigMaps or logs.
  - Rotation succeeds and old credentials cease to work.
- **Expected evidence:** Rotation and secret-access report.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-014: Enforce approved network paths

- **Components and layers:**
  - H04 C02 C03 C05 D01-D04.
  - L3 L5 L6 L7.
- **Observed baseline:** Three-service NetworkPolicies prepared, not cluster-tested.
- **Implementation:** Create default-deny ingress/egress and narrow allow rules for DNS, identity, data, MLflow, model endpoints and telemetry.
- **Owner role:** Platform and security engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-006, SYS-010, SYS-013.
- **Acceptance:**
  - Positive and negative traffic tests pass using the actual CNI and ports.
  - Shared entry points remain closed until identity passes.
- **Expected evidence:** Network flow matrix and probes.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

## Phase P2: Shared core and vector retrieval

### SYS-015: Migrate case metadata from SQLite to PostgreSQL

- **Components and layers:**
  - D01 C02 C03 A03 A04.
  - L3 L5 L7.
- **Observed baseline:**
  - D01 uses SQLite.
  - Compose PostgreSQL holds MLflow metadata only.
- **Implementation:** Implement database adapter/migrations, transaction semantics, row access, revision consistency and data migration with rollback.
- **Owner role:** Backend and database engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-003, SYS-005.
- **Acceptance:**
  - Existing cases/revisions/jobs/audit migrate without loss.
  - Both-store contract tests and concurrent edits pass.
- **Expected evidence:** Migration, parity and rollback report.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-02.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-016: Provide governed source object storage

- **Components and layers:**
  - D02 C02 C03.
  - L3 L5.
- **Observed baseline:** Originals and derivatives stored on local volume.
- **Implementation:** Implement approved object store, project-scoped access, immutable original hashes, encryption, retention and deletion.
- **Owner role:** Backend and storage engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-011, SYS-013.
- **Acceptance:**
  - Originals remain traceable.
  - Other projects cannot access objects.
  - Deletion respects retention and all derivatives.
- **Expected evidence:** Object-store integration and retention tests.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-04, ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-017: Implement durable concurrent job execution

- **Components and layers:**
  - C03 D01.
  - L4 L5.
- **Observed baseline:** One SQLite-backed worker with local recovery.
- **Implementation:**
  - Add PostgreSQL outbox/job claiming, idempotency, leases, retries, cancellation and dead-letter handling.
  - Introduce a broker only if measured need justifies it.
- **Owner role:** Backend engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-015, SYS-011.
- **Acceptance:**
  - Duplicate delivery produces one business result.
  - Worker crash and permission revocation recover safely.
- **Expected evidence:** Concurrency and recovery tests.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-02.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-018: Deploy the secure shared interface and API

- **Components and layers:**
  - C01 C02 H04.
  - L1 L2 L3 L6.
- **Observed baseline:** Local UI/API implemented, enterprise interface pending.
- **Implementation:** Integrate authenticated UI, bounded uploads, accessible errors, health endpoints and secure shared ingress.
- **Owner role:** Frontend and backend engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-009, SYS-011, SYS-012, SYS-014, SYS-015, SYS-016.
- **Acceptance:**
  - Named users complete all five workflows.
  - Unauthenticated mutation APIs unavailable.
  - Large/bad uploads remain bounded.
- **Expected evidence:** UI/API acceptance and Route checks.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-01, ADR-02, ADR-04.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-019: Extend application audit and retention

- **Components and layers:**
  - A04 D01 C02 C03 C05.
  - L3 L5 L7.
- **Observed baseline:** Local actor/case/action/decision/version audit.
- **Implementation:** Propagate correlation IDs, record access and release decisions, restrict audit readers and implement retention/export without raw case content.
- **Owner role:** Backend and security engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-011, SYS-015, SYS-017.
- **Acceptance:**
  - Denied access, review, release and routing events correlate.
  - Raw source text and secrets absent.
  - Audit failure handling defined.
- **Expected evidence:** Audit lineage and redaction tests.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-02, ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-020: Select and pin multilingual embeddings

- **Components and layers:**
  - M05 D03.
  - L4 L5.
- **Observed baseline:** Lexical TF-IDF retrieval only, no embedding weights.
- **Implementation:**
  - Benchmark pretrained multilingual candidates on FR/NL/EN against lexical search.
  - Pin model, tokenizer, dimensions, licence and local preparation.
- **Owner role:** AI model engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-005.
- **Acceptance:**
  - Recorded candidate comparison and model manifest.
  - No request-time downloads.
  - A library/runtime is not labelled a model.
- **Expected evidence:** Embedding decision and benchmark.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-03.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-021: Deploy a separate vector database

- **Components and layers:**
  - D03 H02 H04.
  - L5 L6.
- **Observed baseline:** No vector store installed.
- **Implementation:**
  - Prepare separate PostgreSQL/pgvector database, migrations, volume, credentials and service.
  - Do not reuse MLflow database or credentials.
- **Owner role:** Database and platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-020, SYS-013.
- **Acceptance:**
  - Extension/version and restore tested.
  - Runtime role cannot change schema.
  - No public database port.
- **Expected evidence:** Vector database and restore report.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-03.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-022: Index only approved source revisions

- **Components and layers:**
  - M05 D03 A03 C03.
  - L4 L5 L7.
- **Observed baseline:** Reviewed-only eligibility exists in lexical code.
- **Implementation:** Create token-bounded chunks with timestamps/pages, source hashes, case/project IDs, model revision and immutable index generations.
- **Owner role:** AI and backend engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-012, SYS-017, SYS-020, SYS-021.
- **Acceptance:**
  - Every index row traces to an approved revision.
  - Corrections/revocation remove eligibility immediately.
  - Outbox retries are idempotent.
- **Expected evidence:** Index manifest and edit/revoke tests.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-03, ADR-04.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-023: Implement authorised hybrid retrieval

- **Components and layers:**
  - M05 D03 A02 C05.
  - L3 L4 L5 L7.
- **Observed baseline:** Current search uses a case-scoped lexical corpus.
- **Implementation:** Authorise before embedding/search, filter candidates by permitted case and project, revalidate authoritative revision, compare hybrid ranking and exact search.
- **Owner role:** AI and backend engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-011, SYS-022.
- **Acceptance:**
  - Zero unreviewed, stale or unauthorised citations.
  - Unsupported questions abstain.
  - Failure never broadens the retrieval scope.
- **Expected evidence:** Retrieval quality and isolation report.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-03.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-024: Accept vector quality and reindex recovery

- **Components and layers:**
  - M05 D03.
  - L4 L5 L8.
- **Observed baseline:** No vector evaluation or reindex process.
- **Implementation:**
  - Measure Recall@k, citation precision, unsupported-answer rate, latency and indexing lag.
  - Exercise model change, database outage and rollback.
- **Owner role:** QA and AI model engineer.
- **Primary profile:** Quality assurance.
- **Prerequisites:** SYS-023.
- **Acceptance:**
  - Agreed thresholds met on held-out questions.
  - Known jacket false match included.
  - Rollback restores compatible model/index pair.
- **Expected evidence:** Comparison report and reindex runbook.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-03, ADR-04.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-025: Measure speech on human recordings

- **Components and layers:**
  - M01.
  - L4 L8.
- **Observed baseline:** Synthetic FR/NL WER approximately 67% and 71%, poor on those fixtures.
- **Implementation:** Use consented fictional human recordings with reference transcripts, acoustic variation, language and hardware profiles.
- **Owner role:** AI model engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-005.
- **Acceptance:**
  - WER, timestamps, latency and failures recorded by language.
  - Release choice justified against pre-agreed thresholds.
- **Expected evidence:** Human speech evaluation.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-026: Evaluate extraction and OCR coverage

- **Components and layers:**
  - M02 M03.
  - L4 L8.
- **Observed baseline:** Image and scanned-PDF OCR implemented and bounded locally.
- **Implementation:**
  - Evaluate mixed PDFs, scans, layouts, languages and hostile inputs.
  - Preserve page lineage and extraction warnings.
- **Owner role:** AI and QA engineer.
- **Primary profile:** Quality assurance.
- **Prerequisites:** SYS-005.
- **Acceptance:**
  - Native text preferred and bounded OCR fallback retained.
  - Unsupported/encrypted or pathological input fails safely.
- **Expected evidence:** Document extraction quality report.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-027: Extend and evaluate the classifier

- **Components and layers:**
  - M04 M08.
  - L4 L8.
- **Observed baseline:** TF-IDF/logistic regression trained on 24 synthetic examples and 8 held-out examples.
- **Implementation:** Curate labelled representative data, class balance, split/leakage policy, confidence and unknown-category handling.
- **Owner role:** AI model engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-005.
- **Acceptance:**
  - Class-wise metrics and calibration reported.
  - Receipt/witness misclassification tested.
  - Human override preserved.
- **Expected evidence:** Dataset/model cards and evaluation.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

## Phase P3: AI services and CPE

### SYS-028: Enable the selected OpenShift AI services

- **Components and layers:**
  - H04 M01 M04 M05 M06 M08.
  - L4 L6 L8.
- **Observed baseline:** No OpenShift AI installation confirmed.
- **Implementation:**
  - Install supported Operator and enable required dashboard, serving, workbenches or pipelines.
  - Inventory actual Ready conditions.
- **Owner role:** Platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-006, SYS-007, SYS-004.
- **Acceptance:**
  - Enabled services have Ready conditions and version evidence.
  - Disabled/optional services not presented as installed.
- **Expected evidence:** Operator/component inventory.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-029: Configure CPU and GPU hardware profiles

- **Components and layers:**
  - H04 M01 M04 M05 M06.
  - L4 L6.
- **Observed baseline:** No shared GPU inventory confirmed.
- **Implementation:** Inventory CPU/GPU capacity, select compatible NVIDIA profile if needed, configure scheduling, quotas and runtime compatibility.
- **Owner role:** Platform and AI model engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-028.
- **Acceptance:**
  - Scheduled workload uses approved hardware.
  - CPU models do not require an invented GPU dependency.
  - Capacity exhaustion handled.
- **Expected evidence:** Hardware allocation and load test.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-030: Serve versioned speech, classifier and embeddings

- **Components and layers:**
  - M01 M04 M05 D04 H04.
  - L4 L5 L6 L8.
- **Observed baseline:** Adapters execute in the local app process.
- **Implementation:** Package each selected model with supported or custom runtime, private endpoint, health checks, limits and pinned artifact.
- **Owner role:** AI platform engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-025, SYS-027, SYS-020, SYS-028, SYS-029, SYS-013, SYS-014.
- **Acceptance:**
  - Endpoint outputs match accepted local model.
  - Missing weights or wrong artifact prevents readiness.
  - No automatic download during requests.
- **Expected evidence:** Endpoint parity and startup evidence.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-031: Implement the inference gateway

- **Components and layers:**
  - C05 C04 A01 A02.
  - L3 L7.
- **Observed baseline:** C04 tests planning decisions only, no C05 service.
- **Implementation:** Implement private gateway enforcing permitted models/versions, caller policy, geography, quotas, cancellation and timeouts before dispatch.
- **Owner role:** Backend and security engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-011, SYS-019, SYS-030.
- **Acceptance:**
  - Direct model endpoint bypass denied.
  - Ineligible request refused/queued.
  - Model and policy decision logged without source content.
- **Expected evidence:** Gateway contract and bypass tests.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-06, ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-032: Select and evaluate a pretrained language model

- **Components and layers:**
  - M06.
  - L4 L8.
- **Observed baseline:** No generative model installed.
- **Implementation:**
  - Choose licensed pretrained model and supported runtime using quality, language, privacy, context and hardware benchmarks.
  - Train only if measured need justifies it.
- **Owner role:** AI model engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-005, SYS-028, SYS-029.
- **Acceptance:**
  - Model/weight/runtime versions and licence recorded.
  - Hallucination and prompt-injection evaluation complete.
- **Expected evidence:** Language-model selection report.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-03.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-033: Add bounded evidence-based generation

- **Components and layers:**
  - M06 M07 M05 A03.
  - L4 L7.
- **Observed baseline:** Deterministic extractive draft and bounded workflow implemented.
- **Implementation:** Use approved retrieved passages only, explicit tool allowlist, step/token bounds, citation checks and separate human acceptance.
- **Owner role:** Backend and AI model engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-024, SYS-031, SYS-032, SYS-012.
- **Acceptance:**
  - Unsupported facts cannot be auto-approved.
  - Source edit invalidates draft.
  - Injection cannot grant tools, access or external egress.
- **Expected evidence:** Generation safety and citation tests.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-03, ADR-04.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-034: Retain verified MLflow publication and rollback

- **Components and layers:**
  - M08 D04.
  - L5 L8.
- **Observed baseline:** Local MLflow registration, checksum release gate and rollback implemented.
- **Implementation:** Deploy authenticated tracking/artifact storage, preserve application release authority and bind serving revision to verified registry evidence.
- **Owner role:** MLOps engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-007, SYS-013, SYS-019, SYS-030.
- **Acceptance:**
  - Ready registry version, successful run and matching artifact hashes required.
  - Wrong checksum or unavailable registry blocks promotion.
- **Expected evidence:** Shared release and rollback evidence.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-035: Automate training and model evaluation pipelines

- **Components and layers:**
  - M04 M08 D04 H04.
  - L4 L8.
- **Observed baseline:** Manual local training and prepared CI.
- **Implementation:** Create reproducible training/evaluation/publishing pipeline with immutable datasets and manual promotion separation.
- **Owner role:** MLOps engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-027, SYS-028, SYS-034, SYS-008.
- **Acceptance:**
  - Pipeline can be replayed from dataset and commit.
  - Training service cannot grant itself production promotion.
- **Expected evidence:** Pipeline and separation report.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-05.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-036: Operate monitoring and drift controls

- **Components and layers:**
  - M01 M03 M04 M05 M06 M08.
  - L4 L8.
- **Observed baseline:** No shared model monitoring deployment.
- **Implementation:**
  - Collect quality, latency, errors, cost and drift indicators without raw prompts/documents.
  - Set alert owners and rollback criteria.
- **Owner role:** MLOps and operations engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-024, SYS-026, SYS-030, SYS-033, SYS-034.
- **Acceptance:**
  - Alert and rollback drill succeeds.
  - Monitoring data respects project access and retention.
- **Expected evidence:** Dashboard, alerts and drill.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-03, ADR-05.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-037: Define environment request and approval

- **Components and layers:**
  - A05 C04.
  - L6 L7.
- **Observed baseline:** Slide 10 defines five-step lifecycle, no provisioning automation.
- **Implementation:** Implement request record with owner, case/data scope, users, models, permitted geography, retention, expiry and approved resources.
- **Owner role:** Security architect and project owner.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-001, SYS-003, SYS-005.
- **Acceptance:**
  - Approval references explicit scope.
  - No self-activation.
  - Unsupported isolation requirements rejected.
- **Expected evidence:** CPE request and approval schema.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-038: Create the project provisioning workflow

- **Components and layers:**
  - A05 H04.
  - L6 L7.
- **Observed baseline:** CPE diagram only, no provisioner.
- **Implementation:** Generate project namespace and versioned policy bundle from approved request, use bounded provisioner identity and idempotent state machine.
- **Owner role:** Platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-037, SYS-006, SYS-010.
- **Acceptance:**
  - Request moves through approved/provisioning/verifying/active states.
  - Partial failure never exposes a usable environment.
- **Expected evidence:** Provisioner and failed-creation tests.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-039: Configure scoped users and service accounts

- **Components and layers:**
  - A01 A05.
  - L7.
- **Observed baseline:** Simulated users only.
- **Implementation:**
  - Bind approved user groups and project workload accounts.
  - Separate project owner, reviewer, runtime and provisioner rights.
- **Owner role:** Identity and platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-038, SYS-009, SYS-010.
- **Acceptance:**
  - Users from another project cannot access workloads or data.
  - Removing membership revokes access.
- **Expected evidence:** CPE element 1 identity tests.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-040: Configure project network isolation

- **Components and layers:**
  - H04 A05 C05.
  - L6 L7.
- **Observed baseline:** Shared stack policy manifests only.
- **Implementation:**
  - Apply project-specific deny policies and explicit gateway/data/identity paths.
  - Prevent direct endpoint and onward cloud bypass.
- **Owner role:** Platform and security engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-038, SYS-014, SYS-031.
- **Acceptance:**
  - Only approved flows pass from each project.
  - Forbidden external and cross-project routes denied.
- **Expected evidence:** CPE element 2 traffic tests.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-06, ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-041: Configure resources and consumption limits

- **Components and layers:**
  - H04 A05 C05.
  - L6 L7.
- **Observed baseline:** Pod requests/limits prepared, no project ResourceQuota or consumption enforcement.
- **Implementation:**
  - Apply ResourceQuota, LimitRange and CPU/GPU/PVC limits.
  - Add application admission for audio duration, request rate and model tokens.
- **Owner role:** Platform and backend engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-038, SYS-029, SYS-031.
- **Acceptance:**
  - Quota breach is refused/queued.
  - Runtime accounts cannot raise limits.
  - Concurrent requests cannot bypass consumption accounting.
- **Expected evidence:** CPE element 3 quota tests.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-042: Configure project-scoped secrets

- **Components and layers:**
  - H04 A05 A01.
  - L6 L7.
- **Observed baseline:** Generic database Secret reference only.
- **Implementation:** Provision per-project credentials and rotation without copying shared secrets or exposing model-store administration.
- **Owner role:** Platform and security engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-038, SYS-013.
- **Acceptance:**
  - Only approved service accounts read secrets.
  - Rotation/revocation verified.
  - Images and logs remain clean.
- **Expected evidence:** CPE element 4 secret tests.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-043: Configure controlled project storage

- **Components and layers:**
  - D01 D02 D03 D04 A05.
  - L5 L7.
- **Observed baseline:** Local volumes and prepared PVCs, no project isolation or retention implementation.
- **Implementation:**
  - Select per-project database/index and object-store boundary or stronger dedicated service.
  - Apply classification, encryption, retention and derived-data deletion.
- **Owner role:** Database and storage engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-038, SYS-015, SYS-016, SYS-021, SYS-022.
- **Acceptance:**
  - Source, chunk, vector, model and backup access tested.
  - Another project cannot read data.
  - Revocation excludes derivatives immediately.
- **Expected evidence:** CPE element 5 data isolation and deletion tests.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-04, ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-044: Configure approved model endpoints

- **Components and layers:**
  - C05 D04 A05.
  - L3 L5 L7.
- **Observed baseline:** Local in-process adapters, no CPE endpoint catalogue.
- **Implementation:**
  - Bind permitted model IDs/revisions to private serving endpoints and scoped tokens.
  - Pin artifact identity and reject unapproved versions.
- **Owner role:** AI platform and security engineer.
- **Primary profile:** AI and MLOps.
- **Prerequisites:** SYS-038, SYS-030, SYS-031, SYS-034.
- **Acceptance:**
  - Only catalogue-approved revisions reachable.
  - Direct use, wrong project and stale model identity refused.
- **Expected evidence:** CPE element 6 model access tests.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-05, ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-045: Enforce fail-closed routing across the CPE

- **Components and layers:**
  - C04 C05 A05.
  - L3 L6 L7.
- **Observed baseline:** C04 planning policy unit tests implemented.
- **Implementation:** Re-evaluate approved destination, data class, geography, CPE requirement, capacity and quota on every live dispatch.
- **Owner role:** Backend and security engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-040, SYS-041, SYS-043, SYS-044.
- **Acceptance:**
  - Revoked approvals and absent capacity refuse/queue.
  - No shared-service fallback for restricted data.
- **Expected evidence:** CPE transverse routing tests.
- **Planning:** Priority P0; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-06, ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-046: Implement project audit and expiry

- **Components and layers:**
  - A04 A05.
  - L7.
- **Observed baseline:** Local audit exists, no active CPE lifecycle.
- **Implementation:**
  - Record provisioning, access and model/routing changes.
  - Add expiry suspension, retention-aware closure and alerting.
- **Owner role:** Backend and operations engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-019, SYS-038, SYS-042.
- **Acceptance:**
  - Audit correlates approvals and actions without raw data.
  - Expired CPE denies new work and revokes credentials.
- **Expected evidence:** CPE audit/expiry test.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. Mandatory dependency of activation gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-047: Verify all controls before activation

- **Components and layers:**
  - A05 C04 C05 H04.
  - L6 L7.
- **Observed baseline:** No CPE activated or tested.
- **Implementation:** Execute all six setup controls plus routing and audit acceptance, sign result, then allow activation only for that policy version.
- **Owner role:** QA lead and security architect.
- **Primary profile:** Quality assurance.
- **Prerequisites:** SYS-039, SYS-040, SYS-041, SYS-042, SYS-043, SYS-044, SYS-045, SYS-046.
- **Acceptance:**
  - Failure of any control keeps environment inactive.
  - Independent reviewer approves the recorded policy/evidence version.
- **Expected evidence:** CPE activation report.
- **Planning:** Priority P0; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Required. First activation control gate.
- **Remaining work:** Open. Named assignment: unconfirmed.

## Phase P4: Acceptance and operations

### SYS-048: Accept the existing stack on OpenShift Local

- **Components and layers:**
  - H04 H02.
  - L6.
- **Observed baseline:** Images checked locally, manifests not applied.
- **Implementation:** Replace placeholder image references, seed model PVC and deploy synthetic single-replica app/MLflow/PostgreSQL through port-forward.
- **Owner role:** Platform and QA engineer.
- **Primary profile:** Quality assurance.
- **Prerequisites:** SYS-006, SYS-007.
- **Acceptance:**
  - 63-test baseline adapted as required, end-to-end checks pass before/after pod replacement.
  - Record actual results, no fixed claimed future count.
- **Expected evidence:** OpenShift Local acceptance report.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01, ADR-02.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-049: Accept the complete shared deployment

- **Components and layers:**
  - H04 C01-C05 M01-M08 D01-D04 A01-A05.
  - L1-L8.
- **Observed baseline:** No shared system deployment.
- **Implementation:**
  - Deploy a versioned overlay with identity, data, jobs, gateway, models, review, audit and CPE.
  - Test all five use cases.
- **Owner role:** Platform and QA lead.
- **Primary profile:** Quality assurance.
- **Prerequisites:** SYS-018, SYS-024, SYS-026, SYS-033, SYS-035, SYS-047, SYS-048.
- **Acceptance:**
  - Authenticated users complete workflows and negative cases.
  - All services Ready and evidence tied to actual versions.
- **Expected evidence:** Shared acceptance and deployment manifest.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-050: Verify backup, restore and service recovery

- **Components and layers:**
  - D01-D04 A04 H04.
  - L5 L6 L7.
- **Observed baseline:** Persistence tested across local app restart, no full restore drill.
- **Implementation:**
  - Back up cases, sources, vectors, registry and audit with coordinated revision watermark.
  - Restore to isolated environment and reindex if needed.
- **Owner role:** Database and operations engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-049.
- **Acceptance:**
  - Agreed recovery objectives measured.
  - Restored citations/model releases match checksums.
  - No expired access revived.
- **Expected evidence:** Restore drill and recovery runbook.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-051: Measure capacity and availability

- **Components and layers:**
  - C02 C03 C05 H04.
  - L3 L4 L6.
- **Observed baseline:** No shared capacity benchmark.
- **Implementation:**
  - Load-test concurrent uploads, review, retrieval, training and inference.
  - Tune replicas, storage and queues after measurement.
- **Owner role:** Performance and platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-049.
- **Acceptance:**
  - Latency, throughput, exhaustion and recovery meet SYS-001.
  - No multi-replica SQLite deployment.
- **Expected evidence:** Capacity report and resource settings.
- **Planning:** Priority P1; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-02.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-052: Test retention and decommissioning

- **Components and layers:**
  - D01-D04 A04 A05.
  - L5 L7.
- **Observed baseline:** Original retention/deletion not operationally accepted.
- **Implementation:** Implement legal-hold-aware expiry, derived-data deletion, backup expiry and safe project closure.
- **Owner role:** Security and data engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-047, SYS-050.
- **Acceptance:**
  - Required data retained and expired eligible data removed.
  - Closure revokes entry and model access before deletion.
- **Expected evidence:** Retention and closure evidence.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-053: Publish operator and support runbooks

- **Components and layers:**
  - H04 C03 C05 M08 A05.
  - L6 L8.
- **Observed baseline:** Local deployment and reproduction guides exist.
- **Implementation:** Write incident, failed job, model rollback, restore, secret rotation, index rebuild and CPE approval/closure procedures.
- **Owner role:** Operations and platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-050, SYS-051, SYS-052, SYS-036.
- **Acceptance:** Another engineer reproduces deployment and recovery with approved inputs, without private local paths.
- **Expected evidence:** Independent runbook walkthrough.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-07.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-054: Reproduce the release on a clean environment

- **Components and layers:**
  - H02 H04.
  - L6 L8.
- **Observed baseline:**
  - Fresh checkout and new virtual environment passed locally on 7 October 2026 with warm caches.
  - Separate Compose project on fresh volumes passed 27 + 8 checks.
  - Clean-machine release and cluster reproduction remain unverified.
- **Implementation:** Install from release assets, restore no private credentials, prepare licensed weights and execute local/cluster acceptance.
- **Owner role:** QA engineer.
- **Primary profile:** Quality assurance.
- **Prerequisites:** SYS-008, SYS-049, SYS-053.
- **Acceptance:**
  - Fresh environment reproduction passes with exact release references.
  - All missing prerequisites documented.
- **Expected evidence:** Clean-environment report.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01, ADR-05.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-059: Reconcile evidence and public descriptions

- **Components and layers:**
  - C01-C05 M01-M08 D01-D04 A01-A05 H01-H06.
  - L1-L8.
- **Observed baseline:** Article and deck revisions under local review.
- **Implementation:**
  - Update component facts, run evidence, deployment labels, article and repository after acceptance.
  - Retain model without ownership assertion.
- **Owner role:** Enterprise architect and technical writer.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-049, SYS-054.
- **Acceptance:**
  - Every deployment claim has dated evidence.
  - Main figures preserved unless separately approved.
  - Exact Word approval before publication.
- **Expected evidence:** Evidence-to-claim register.
- **Planning:** Priority P1; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-01.
- **First CPE activation:** Later scope. Shared delivery or operating acceptance.
- **Remaining work:** Open. Named assignment: unconfirmed.

## Phase P5: Approved extensions

### SYS-055: Map an on-site private-host variant

- **Components and layers:**
  - H03.
  - L6.
- **Observed baseline:** Diagram only, no private site deployment.
- **Implementation:** Define VM/workers/private data and GPU services with the same identities, interfaces, controls and recovery contract.
- **Owner role:** Infrastructure architect.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-004, SYS-049, SYS-053.
- **Acceptance:** Variant states which components change and repeats authorization, network and recovery acceptance.
- **Expected evidence:** Private-host deployment overlay and report.
- **Planning:** Priority P2; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-06.
- **First CPE activation:** Later scope. Separate extension approval.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-056: Approve the secondary service contract

- **Components and layers:**
  - H05 C04 C05.
  - L3 L6 L7.
- **Observed baseline:** Routing planning only, no service connection.
- **Implementation:** Confirm private interconnect, destination, compatible model, licence, region, classification, quotas and operational responsibilities.
- **Owner role:** Enterprise architect and security owner.
- **Primary profile:** Architecture and security.
- **Prerequisites:** SYS-045, SYS-049.
- **Acceptance:**
  - Each transfer has explicit approval and technical rejection cases.
  - Actual GPU inventory recorded.
- **Expected evidence:** Secondary service contract.
- **Planning:** Priority P2; size M; draft effort 3–5 person-days.
- **Architecture decisions:** ADR-06.
- **First CPE activation:** Later scope. Separate extension approval.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-057: Implement the secondary service connector

- **Components and layers:**
  - H05 C05 A04.
  - L3 L6 L7.
- **Observed baseline:** No live remote connector.
- **Implementation:** Implement authenticated approved service calls, retries, idempotency, cancellation, accounting and restricted payloads.
- **Owner role:** Backend and network engineer.
- **Primary profile:** Application engineering.
- **Prerequisites:** SYS-056, SYS-053.
- **Acceptance:**
  - Secondary failure queues/refuses.
  - No direct cloud fallback.
  - Denial and transfer decisions audited.
- **Expected evidence:** Secondary connector acceptance.
- **Planning:** Priority P2; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-06.
- **First CPE activation:** Later scope. Separate extension approval.
- **Remaining work:** Open. Named assignment: unconfirmed.

### SYS-058: Govern onward sovereign-cloud processing

- **Components and layers:**
  - H06 H05 C04 A04.
  - L6 L7.
- **Observed baseline:** No cloud resources or approved onward path.
- **Implementation:** Specify and implement independently approved secondary-to-cloud path, geography, encryption, retention, exit and revocation.
- **Owner role:** Security and platform engineer.
- **Primary profile:** Platform and data.
- **Prerequisites:** SYS-057.
- **Acceptance:**
  - Primary cannot call cloud directly.
  - Onward approval is checked at secondary hop.
  - Missing approval denies transfer.
- **Expected evidence:** Onward transfer acceptance.
- **Planning:** Priority P2; size L; draft effort 6–12 person-days.
- **Architecture decisions:** ADR-06.
- **First CPE activation:** Later scope. Separate extension approval.
- **Remaining work:** Open. Named assignment: unconfirmed.
