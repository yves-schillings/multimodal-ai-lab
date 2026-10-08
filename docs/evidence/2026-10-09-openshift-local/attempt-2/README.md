# OpenShift Local acceptance run - 2026-10-09

Result: **application accepted on OpenShift Local (OKD preset, the community distribution of OpenShift bundled with CRC; not the Red Hat OpenShift Container Platform bundle)**.

Scope: application, MLflow tracking server and PostgreSQL from `deploy/openshift/overlays/openshift-local` on a single-node
OpenShift Local cluster with synthetic data only. Access through `oc port-forward`; no Route; simulated identities.
**OpenShift AI and KServe are not installed and not tested by this run. Their acceptance is a separate scope.**

## Platform

- Login: `developer`
- CRC: `CRC version: 2.64.0+458092`
- CRC preset: `preset : okd` (OKD preset, the community distribution of OpenShift bundled with CRC; not the Red Hat OpenShift Container Platform bundle)
- oc/server version: see `preflight.json`
- Storage classes: recorded

## Images built in the cluster registry

- `lab-app`: `sha256:d55ab0432f36e0a0dc523f54bcfdc60b9a2d501e51e54fb4aa165dc3bb799131`
- `lab-mlflow`: `sha256:23bc9ea5ff89c7069508ba43eca3eb0feb1cd613cdc73c67c9284470c02b8dee`

## Checks before restart

| Check | Result | Detail |
| --- | --- | --- |
| three pods Running and Ready | PASS | lab-app, lab-mlflow, lab-postgres |
| persistent volume claims Bound | PASS | {"lab-data": "Bound", "lab-models": "Bound", "mlflow-artifacts": "Bound", "postgres-data": "Bound"} |
| containers run as non-root UID | PASS | {"app": "1000650000", "mlflow": "1000650000", "postgres": "1000650000"} |
| app UID is the arbitrary OpenShift UID, not the image default 10001 | PASS | 1000650000 |
| data volume writable by the arbitrary UID | PASS | writable |
| network matrix: every allowed path open and every denied path closed | PASS | 9/9 |
| /health/ready through port-forward | PASS | {"status": "ready", "mode": "synthetic_container", "network_mode": "container"} |
| synthetic end-to-end scenario (before) | PASS | 27 passed, 0 failed |

Network matrix:

| Path | Expected | Observed | Result |
| --- | --- | --- | --- |
| app -> mlflow:5000 /health | allowed | open | PASS |
| app -> postgres:5432 | denied | closed | PASS |
| app -> internet 1.1.1.1:443 | denied | closed | PASS |
| mlflow -> postgres:5432 | allowed | open | PASS |
| mlflow -> app:8770 | denied | closed | PASS |
| postgres -> mlflow:5000 | denied | closed | PASS |
| probe pod -> app:8770 (ingress default deny) | denied | closed | PASS |
| probe pod -> mlflow:5000 (ingress only from app) | denied | closed | PASS |
| probe pod -> postgres:5432 (ingress only from mlflow) | denied | closed | PASS |

## Checks after restart

| Check | Result | Detail |
| --- | --- | --- |
| three pods Running and Ready | PASS | lab-app, lab-mlflow, lab-postgres |
| persistent volume claims Bound | PASS | {"lab-data": "Bound", "lab-models": "Bound", "mlflow-artifacts": "Bound", "postgres-data": "Bound"} |
| containers run as non-root UID | PASS | {"app": "1000650000", "mlflow": "1000650000", "postgres": "1000650000"} |
| app UID is the arbitrary OpenShift UID, not the image default 10001 | PASS | 1000650000 |
| data volume writable by the arbitrary UID | PASS | writable |
| network matrix: every allowed path open and every denied path closed | PASS | 9/9 |
| /health/ready through port-forward | PASS | {"status": "ready", "mode": "synthetic_container", "network_mode": "container"} |
| synthetic end-to-end scenario (after) | PASS | 8 passed, 0 failed |

Network matrix:

| Path | Expected | Observed | Result |
| --- | --- | --- | --- |
| app -> mlflow:5000 /health | allowed | open | PASS |
| app -> postgres:5432 | denied | closed | PASS |
| app -> internet 1.1.1.1:443 | denied | closed | PASS |
| mlflow -> postgres:5432 | allowed | open | PASS |
| mlflow -> app:8770 | denied | closed | PASS |
| postgres -> mlflow:5000 | denied | closed | PASS |
| probe pod -> app:8770 (ingress default deny) | denied | closed | PASS |
| probe pod -> mlflow:5000 (ingress only from app) | denied | closed | PASS |
| probe pod -> postgres:5432 (ingress only from mlflow) | denied | closed | PASS |

## Restart

- All three pods deleted at 2026-10-09T00:47:32; replacements Ready: {'lab-app': 'lab-app-6b5549cc4b-56w4f', 'lab-mlflow': 'lab-mlflow-58c9d979cc-bxqkr', 'lab-postgres': 'lab-postgres-0'}
- Persistent volumes unchanged: True

## Limits

- Single-node developer cluster on one workstation; not a sizing or performance result.
- Prepared local Whisper base weights were copied and separately verified. Speech and OCR results are recorded in the [supplemental checks](../README.md), outside this generated core workflow report.
- Port-forward bypasses NetworkPolicy by design (kubelet path); the ingress default-deny was verified from an in-namespace probe pod.
- No Route, no trusted identity, no GPU, no model serving: unchanged target scope.
