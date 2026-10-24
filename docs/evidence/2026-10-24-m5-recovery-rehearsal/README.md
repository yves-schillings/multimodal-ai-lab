# M5 full recovery rehearsal - 24 October 2026

Status: **not executed.** M5 rebuilds the laboratory on a fresh OKD cluster. On this workstation, OpenShift Local runs one cluster only, so a fresh cluster means deleting the current one with `crc delete`, including the CPE namespaces and the local AI platform that the rehearsal does not restore. That irreversible step waits for the owner's explicit approval. No recovery time is claimed.

The procedure, backup set, timing points and acceptance criteria are fixed in [`PROTOCOL.md`](PROTOCOL.md).

## What already exists

- The 9 October recovery exercise restored SQLite and file copies offline and the `mlflow` and `lab_vectors` PostgreSQL dumps into temporary databases, with ten passing checks. It did not rebuild a cluster, back up the inference weights or the MLflow artifact volume, or measure a recovery time. See [the local implementation guide](../../deployment/local-implementation.md), section 10.

## Observed on 24 October while restarting the existing cluster

These observations come from bringing the stopped cluster back after 15 days. They are not the M5 measurement, but they belong in any recovery runbook.

- The Windows scheduled task that starts the CRC daemon could not be read through PowerShell (`Get-ScheduledTask` error 0x80041318), so `crc status` and `crc start` refused to run. Starting the existing task with `schtasks /run /tn crcDaemon` and skipping the two daemon-task preflight checks for that start worked, as on 9 October; the checks were restored afterwards.
- `crc start` stopped after 11 min 40 s while renewing the expired kubelet client certificate ("No Pending CSR with signerName kubernetes.io/kubelet-serving"). The cluster itself was healthy: the node was `Ready` and all cluster operators available.
- `lab-app` and `lab-mlflow` stayed in `ImagePullBackOff`: the namespace's internal-registry pull token had expired on 9 October. Once the OpenShift controller manager took its lease, it refreshed the token; replacing the two pods then started them on their existing volumes.
- From the virtual machine start (09:59:19) to the replaced `lab-app` container running (10:14:54) took 15 min 35 s, including about 4 minutes of manual diagnosis. Most of it went to the certificate renewal timeout and the expired pull token.
