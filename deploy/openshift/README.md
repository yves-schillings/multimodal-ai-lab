# OpenShift manifests: prepared configuration, not a tested deployment

Status: **Prepared**. These manifests were written from the Docker Compose deployment that was
verified locally (see `docs/deployment.md`). They have **not** been applied to any OpenShift
cluster by this repository's authors. No cluster, GPU pool, route host or image registry was used.
Treat every file here as a blueprint to review with the platform team before a first `oc apply`.

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
- **No image build.** Build the image with the repository `Dockerfile` and push it to the cluster
  registry, or create a BuildConfig; then set the image name in `app.yaml`.

## Expected commands (to run by the platform team, not yet executed)

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
