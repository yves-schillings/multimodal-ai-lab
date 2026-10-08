# Deployment stages

Status vocabulary used in this document and in the architecture deck: **Implemented** (runs and is
tested), **Prepared** (code or configuration exists, not verified), **Target** (design only).

## Latest retained verification

The [7 October 2026 closing package](evidence/2026-10-07-closing/README.md) records
63 Python tests and 15 subtests, 27 Compose checks before restart and 8 afterwards,
bounded image/scanned-PDF OCR and model publication with checksum-verified release.
The [fresh-checkout package](evidence/2026-10-07-fresh-clone/README.md) confirms these
checks on a separate local checkout and fresh volumes, with warm caches on the same
workstation. Neither package is hosted CI, a release tag or cluster acceptance.

The 53-test/25-check and 58-test records below describe earlier implementation
snapshots. They are preserved as dated history rather than the latest suite count.
The original image report used a gate that rejected all HIGH/CRITICAL findings.
The prepared workflow now blocks fixable HIGH/CRITICAL findings while retaining the
full report; this policy change does not approve the remaining findings or establish
a green hosted run. See the [security disposition](evidence/2026-10-07-security-review/README.md).

## Running today

The standalone synthetic browser exercise is separate from the local Python deployment. It executes in the browser and has no operational case connection, file intake, cloud inference or MLflow server. Its simulated personas are teaching aids.

The Python application runs locally at http://127.0.0.1:8770. Use `Start.cmd` on Windows. Runtime databases, model weights, audio and logs are ignored by Git.

## Docker Compose on a local host (Implemented, verified on 7 October 2026)

`deploy/docker/compose.yaml` starts three services with persistent named volumes:

| Service | Image | Persistent volume | Role |
| --- | --- | --- | --- |
| `app` | built from the repository `Dockerfile` | `lab-data` (SQLite case store, uploads, model pickles); `../../models` mounted read-only for the prepared speech weights | Case API, browser workspace, durable job runner, in-process adapters |
| `mlflow` | built from `deploy/docker/mlflow.Dockerfile` (MLflow 3.16.1 + psycopg2) | `mlflow-artifacts` | Tracking server; the application logs runs to it when `MLFLOW_TRACKING_URI` is set |
| `postgres` | `postgres:16-alpine` | `postgres-data` | MLflow backend store (experiments, runs, metrics, params) |

Network behaviour is fail-closed. The image defaults to `LAB_BIND_HOST=127.0.0.1` and
`LAB_NETWORK_MODE=loopback`; Compose sets `0.0.0.0` and `container` explicitly. In `container` mode
the application accepts clients from the container network but still refuses any `Host` header that
is not loopback or listed in `LAB_ALLOWED_HOSTS`. Ports are published on the host loopback only.
Actors remain simulated identities: nothing here adds authentication.

### Commands

```bash
# from the repository root; host port 8780 leaves 8770 to the Start.cmd instance
LAB_HOST_PORT=8780 docker compose -f deploy/docker/compose.yaml up -d --build
python scripts/demo_e2e.py --phase before --base http://127.0.0.1:8780 --state data/reports/compose-state.json
docker compose -f deploy/docker/compose.yaml down          # containers removed, volumes kept
LAB_HOST_PORT=8780 docker compose -f deploy/docker/compose.yaml up -d
python scripts/demo_e2e.py --phase after  --base http://127.0.0.1:8780 --state data/reports/compose-state.json
docker compose -f deploy/docker/compose.yaml down -v       # only when the volumes must be discarded
```

On PowerShell, set the variable first: `$env:LAB_HOST_PORT = "8780"`. The MLflow user interface is
at http://127.0.0.1:5000 while the stack runs. Copy `deploy/docker/.env.example` to
`deploy/docker/.env` and change the PostgreSQL password before any use beyond a local demonstration.

### Verified results (development machine, Docker Engine 28.2.2, 7 October 2026)

`scripts/demo_e2e.py --phase before` against the container, 25 checks passed, 0 failed:

- synthetic dossier created by officer A; `samples/evidence-inventory.txt` uploaded, extracted through the durable job runner; classification is unavailable until an active model exists;
- source reviewed by officer A; draft prepared from reviewed sources only; officer A refused approval (403); the separate reviewer approved; a supported question returned one cited source; an unsupported question abstained;
- officer B refused on officer A's dossier (403) for read and for question; officer A refused on `demo-case-b` (403); unknown actor refused (403);
- two training jobs completed with metrics recorded on the MLflow server (`recorded_on_server`); officer promotion refused (403); unknown candidate refused (422); reviewer promoted candidate 1, then candidate 2; rollback restored candidate 1; the active model classifies and flags `requires_human_validation`.

`docker compose down` then `up -d` (containers and network recreated, volumes kept), then
`scripts/demo_e2e.py --phase after`, 8 checks passed, 0 failed: the case, its reviewed document,
the draft and its approval, the denial, the four model versions and the active model all survived.
The two MLflow runs were still listed by the server and counted by `select count(*) from runs` in
PostgreSQL.

Unit tests at the same commit: 53 passed and 15 subtests passed with pytest (49 before this change
plus 4 network-mode tests). The browser check passed.

### Remaining limits of the Compose deployment

- The case store stays in SQLite on the `lab-data` volume; PostgreSQL serves MLflow only. See the migration checklist below.
- One application replica only: the job runner is a single in-process worker and SQLite is file-based.
- Speech works only if `scripts/prepare_speech.py` was run on the host before `up`; the container never downloads weights.
- Image OCR runs through Tesseract in the Docker image. A synthetic English scan passed extraction with an arbitrary non-root UID; a later closing run also verifies scanned-PDF OCR, while multilingual accuracy remains separate acceptance work.
- Simulated actors, no TLS inside the stack, loopback-only publishing: a demonstration, not a shared service.
- MLflow 3 rejects unknown `Host` headers; the server is started with `--allowed-hosts` for its service names. Add any other name there before reusing the stack elsewhere.

## Moving the case store from SQLite to PostgreSQL (checklist, Target)

`lab/store.py` and `lab/backend_store.py` use the `sqlite3` module directly. A migration without a
general rewrite needs the following, in this order:

1. **Connection factory.** Replace `sqlite3.connect` in `StatementStore._connection` by a factory selected from `LAB_DATABASE_URL`; keep the context-manager contract (`write=True` opens a transaction, commit on success, rollback on error).
2. **Placeholders.** Every statement uses `?`; PostgreSQL drivers use `%s`. Route statements through one helper that rewrites placeholders, or adopt a thin adapter with named parameters.
3. **Schema statements.** `executescript` with several statements, `INTEGER PRIMARY KEY AUTOINCREMENT` (audit table) and `PRAGMA foreign_keys=ON` are SQLite-specific. Use `GENERATED ALWAYS AS IDENTITY`, split the scripts, and drop the pragma.
4. **Row access.** Code reads rows as `sqlite3.Row` (by name and by `dict(row)`); use a dictionary row factory on the PostgreSQL side.
5. **Upserts and JSON.** `INSERT ... ON CONFLICT(key) DO UPDATE` is valid in both; JSON stays in `TEXT` columns, or moves to `JSONB` later.
6. **Concurrency.** `BEGIN IMMEDIATE` serialises SQLite writers. With PostgreSQL, keep one application replica until the job runner claims jobs with `SELECT ... FOR UPDATE SKIP LOCKED`.
7. **Data move.** Export cases, transcripts, segments, documents, jobs, settings, model versions and audit rows from `statements.sqlite3`; load them in dependency order; keep uploads and model pickles on object storage or a shared volume.
8. **Tests.** Run the existing suite against both engines (the tests create the store on a temporary directory; add a PostgreSQL fixture gated by an environment variable).

## Red Hat OpenShift manifests (Prepared, not tested)

`deploy/openshift/base` contains the Kustomize equivalent of the Compose stack: ConfigMap with the
same environment, PostgreSQL StatefulSet, MLflow Deployment, application Deployment with two
PersistentVolumeClaims, Services, and NetworkPolicies (default-deny ingress, egress limited to
MLflow and DNS). No Route is in the base; `overlays/exposed-route` documents the shape of a Route
and must not be applied while actors are simulated. The manifests have not been applied to any
cluster; the acceptance steps are listed in `deploy/openshift/README.md`.

## Primary AI Lab target

Red Hat OpenShift AI is the shared deployment target. Before activating application routes, replace simulated identity with trusted OpenID Connect, implement project/case authorization at every service boundary, provision controlled storage and model-serving identities, and validate supported platform/model versions. Deploy model services and pipelines with resource limits, health checks, controlled egress, artifact provenance and measured release gates.

The current application is not suitable for exposure through an OpenShift Route. Merely changing the bind address does not establish authentication or isolation; the `container` network mode exists for port-forward and in-cluster access only.

## Future resource pools

Use only the neutral names **Primary AI Lab**, **Secondary GPU Data Center** and **Sovereign Cloud**. A policy-approved private interconnect or VPN can extend capacity. The extension is a service chain: the primary lab requests approved AI services from the Secondary GPU Data Center, which may burst to the Sovereign Cloud only with explicit onward approval. There is no direct route from the primary lab to the cloud. Cloud bursting selects a compatible pool only after classification, permitted processing geography, destination approval, retention, CPE requirements and consumption quotas pass. Each onward transfer is independently governed.

Case/source classification is distinct from geography. No actual hosting region, GPU capacity, sovereign certification or commercial rate is assumed. Unavailable eligible capacity queues or rejects work. Access tokens and inference consumption tokens serve different purposes.

## Future CPE lifecycle

A Controlled Project Environment is activated only for cases requiring enhanced isolation. Establish the owner, case policy, approvals and prerequisites; provision namespace, identities, storage and network policy; configure services and quotas; verify permitted and forbidden routes; activate and monitor. Approved shared speech processing can operate without a dedicated CPE while retaining confidentiality controls.

## Later Azure preparation

Use synthetic data for an initial Azure variant. Real source data and derived transcripts require explicit authorisation for the chosen service, region, identity, access, retention and transfer path. No Azure resources have been provisioned by this release.

## Model release evidence — 7 October 2026

MLflow now stores the loadable sklearn model, the exact local inference artifact and registered
model versions. The PostgreSQL backend stores registry metadata as well as experiments and runs.
Application promotion and rollback require successful recording, a READY version linked to the
finished training run, and matching SHA-256 checksums of local and downloaded model artifacts.
The release decision is application-managed; no shared production model-serving deployment is claimed.

Fresh checks on this local revision:

- 58 Python tests and 15 subtests passed; browser core checks passed.
- Compose replay: 27 checks before application restart and 8 afterwards, all passed.
- Two newly trained candidates were published and registered on the Compose MLflow server;
  reviewer promotion, second promotion, rollback and persistent inference passed.
- An isolated Docker run without network access, using UID 1000780000 and group 0,
  passed application data-store creation, classifier imports and synthetic image OCR.
- Older data volumes were retained; no runtime database or model directory was erased.

OpenShift preparation now includes group-writable runtime directories, arbitrary UID compatibility,
explicit probe Host headers and restricted application/MLflow/PostgreSQL network policies.
These image checks do not substitute for actual cluster storage, security-policy or Operator acceptance.
See [local OpenShift preparation](../deploy/openshift/local.md).

Historical Compose evidence earlier in this document (25 checks before container recreation and 8
afterwards) remains a separate run. The new 27-check replay adds model artifact publication checks.

## Closing snapshot verification — 7 October 2026

The rebuilt application passed 63 Python tests plus 15 subtests, browser core checks, 27 Compose checks before an actual application restart and 8 after it. Registry versions 5 and 6 were published. A real image-only PDF passed the Tesseract fallback. The application image also passed image and scanned-PDF OCR as UID 1000780000 with network disabled, capabilities dropped and no-new-privileges.

[The dated evidence package](evidence/2026-10-07-closing/README.md) retains reports and tested-source hashes. Its image audit records 76 HIGH and 1 CRITICAL package findings, with no fixed versions listed in that database snapshot. Python requirement auditing returned no known vulnerabilities. The prepared CI gate remains strict and has not run on GitHub. These checks do not prove a cluster deployment, a clean-machine reproduction or French and Dutch quality.

Earlier 25/8, 53-test and 58-test results above describe prior snapshots. They are retained as history rather than substituted for the closing reports.

## Fresh local clone and independent volumes — 7 October 2026

A separate clone of the unpublished local snapshot was installed in a new virtual environment and run in a distinct Compose project with initially empty volumes. All 63 Python tests and 15 subtests, browser core checks, 27 pre-restart checks and 8 post-restart checks passed. The cloned image also passed image and scanned-PDF OCR under an arbitrary non-root UID with networking disabled. See the [timed reports and exact source hashes](evidence/2026-10-07-fresh-clone/README.md). The existing demonstration volumes were not reused.

This narrows the local-reproduction gap; it does not establish cold-cache or other-machine reproduction. The [security disposition](evidence/2026-10-07-security-review/README.md) separately explains the current fixable-only automation gate and the retained unapproved findings. Neither a local test pass nor zero reported fixable findings substitutes for a hosted CI run or shared-platform acceptance.
