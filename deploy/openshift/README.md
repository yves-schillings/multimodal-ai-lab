# OpenShift deployment preparation

Status: **Accepted locally on CRC with the OKD preset on 9 October 2026**. The
application, MLflow and PostgreSQL passed the local cluster workflow, restricted
UID, NetworkPolicy and restart-persistence checks. See the
[successful acceptance run](../../docs/evidence/2026-10-09-openshift-local/attempt-2/README.md).
Shared deployments remain prepared; OpenShift AI, GPU serving and trusted identity
are not installed or accepted by this run. No application Route is exposed.

For a desktop first step, see [local.md](local.md) and run `Check-LocalHost.ps1`.
The application and MLflow images support arbitrary non-root UIDs with the root group;
this was verified on Docker and the local OKD cluster, not an OpenShift AI installation.
Health probes supply the allowed loopback Host header without weakening the application guard.

The same three services as Docker Compose run in one namespace:

| Service | Manifest | Storage | Notes |
| --- | --- | --- | --- |
| Application (C01 Browser workspace, C02 Case API, C03 Job runner, adapters) | `app.yaml` | PVC `lab-data` (SQLite, uploads, models), PVC `lab-models` (prepared speech weights) | One replica only: the job runner is a single durable local worker and SQLite is file-based |
| MLflow tracking server (M08) | `mlflow.yaml` | PVC `mlflow-artifacts` | Backend store on PostgreSQL |
| PostgreSQL (MLflow backend store) | `postgres.yaml` | PVC `postgres-data` | Password from the Secret |

The application keeps its case data in SQLite on a persistent volume. Moving the case store to
PostgreSQL is a separate change; the checklist is in `docs/deployment.md`.

## What is deliberately not included

- **No Route in the base.** The application still uses simulated actors. Exposing it through a
  Route would publish an unauthenticated mutation API. `overlays/exposed-route/` exists only to
  document the shape of a Route and must not be applied before trusted OpenID Connect identity
  and case authorization at the boundary are implemented.
- **No GPU scheduling, no model serving, no workbenches.** Speech runs on CPU inside the application
  container, as in Docker Compose. OpenShift AI model serving remains a target.
- **Image build:** build the application with the repository `Dockerfile`, and MLflow with
  `deploy/docker/mlflow.Dockerfile`. Push them to the chosen cluster registry; set both the
  application/init-container image and MLflow image references before deployment.

## Commands for a separately reviewed deployment

```bash
oc new-project multimodal-ai-lab
oc create secret generic lab-postgres --from-literal=POSTGRES_PASSWORD='choose-a-strong-value' -n multimodal-ai-lab
oc apply -k deploy/openshift/base -n multimodal-ai-lab
oc rollout status deployment/lab-app -n multimodal-ai-lab
oc port-forward deployment/lab-app 8770:8770 -n multimodal-ai-lab
python scripts/demo_e2e.py --phase before --base http://127.0.0.1:8770
```

The port-forward keeps the Host header on `127.0.0.1`, which the application accepts in
`container` network mode without any Route.

## Acceptance before anyone calls this "deployed"

1. `oc get pods` shows the three pods Ready; `/health/ready` returns `network_mode: container`.
2. `scripts/demo_e2e.py --phase before` passes through the port-forward.
3. Delete the application pod; `scripts/demo_e2e.py --phase after` passes (data on PVCs survived).
4. The NetworkPolicy denies any ingress to the namespace other than the port-forward path and
   limits egress to PostgreSQL, MLflow and DNS.
5. Record platform version, storage class, resource limits and the image digest in `docs/deployment.md`.

## Prepared local speech weights

After building the images and binding the models PVC, copy the already prepared
Whisper base files with no model download from the running application:

```powershell
crc oc-env --shell powershell | Invoke-Expression
python scripts/copy_speech_to_cluster.py --report '<private-evidence-folder>/speech-weights.json'
```

The utility uses a temporary restricted pod and checks SHA-256 digests. It verifies
existing files instead of overwriting them, then removes the temporary pod. The
application retains its read-only models mount. Synthetic local CPU transcription,
image OCR, scanned-PDF OCR and saved speech state after pod replacement were also
verified; see the speech and OCR reports beside the successful acceptance run.
