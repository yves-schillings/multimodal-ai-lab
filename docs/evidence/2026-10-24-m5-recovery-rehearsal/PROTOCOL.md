# M5 full recovery rehearsal - protocol

The rehearsal rebuilds the main laboratory (`multimodal-ai-lab` namespace) on a **fresh** OKD cluster from backups only, replays the acceptance scripts on the restored service and measures the recovery time.

## Scope

- In scope: the six main workloads (`lab-app`, `lab-mlflow`, `lab-postgres`, `lab-vector`, `lab-inference`, `lab-inference-gateway`), their six persistent volumes, five Secrets, network policies and the two image streams.
- Out of scope: the CPE namespaces `lab-cpe-a` and `lab-cpe-b`, the `lab-ai-platform` and `kserve` namespaces, and any shared OpenShift AI platform. A fresh cluster loses them; their reconstruction is a separate exercise.

## Backup set, taken before the rebuild

| Backup | Method | Restored how |
| --- | --- | --- |
| Application data: `statements.sqlite3`, `identity.sqlite3`, uploads, classifier releases | SQLite online backup API inside the app pod, then a tar archive with a SHA-256 manifest ([`cluster_backup_acceptance.py`](../../../scripts/cluster_backup_acceptance.py)) | Copied into the new `lab-data` volume before the app starts |
| MLflow tracking database | `pg_dump -Fc` of `mlflow` | `pg_restore` into the new `lab-postgres` |
| Vector database | `pg_dump -Fc` of `lab_vectors` | `pg_restore` into the new `lab-vector`; the service would also rebuild it on demand from reviewed sources |
| MLflow artifact volume | tar of `/mlflow/artifacts` inside the MLflow pod | Copied into the new `mlflow-artifacts` volume |
| Secrets: local credentials, PostgreSQL, vector admin and client, inference gateway | `oc get secret -o json`, stripped of cluster metadata, owner-only private file | `oc create -f` before the workloads start |
| Whisper base weights | Already prepared outside the cluster (`models/whisper-base`, public weights with recorded hashes) | Copied to the new `lab-models` volume |
| BGE-M3 and Qwen3:4b weights | Host Ollama store, pinned manifest digests | [`copy_inference_to_cluster.py`](../../../scripts/copy_inference_to_cluster.py) verifies every file |
| Source and manifests | Git commit recorded in the report | Images rebuilt in the new cluster registry |

Every backup file stays in an owner-only private folder outside Git. The report publishes their SHA-256 digests and sizes only.

## Timeline and recovery time

1. Record the backup set and stop the old cluster. Export the old virtual machine so it can be imported back if the rehearsal fails.
2. **T0:** start of the rebuild: `crc delete`, then `crc start` of a new OKD 4.22 cluster.
3. Create the namespace, restore the Secrets, build both images, deploy PostgreSQL, MLflow, pgvector, Ollama, the gateway and the app with the documented manifests and patches.
4. Restore the data, copy the weights, roll out the app.
5. **T1:** the first accepted case: an authenticated officer opens a case that existed before the backup and receives a cited answer from its restored reviewed sources.
6. Recovery time = T1 − T0, with the duration of each step recorded.

## Acceptance on the restored service

- Restored counts of cases, documents, jobs, model versions, monitoring runs, MLflow runs and vector rows equal the backup manifest.
- The existing scripts pass on the restored service: `cluster_auth_acceptance.py`, `cluster_rag_acceptance.py`, `cluster_rag_network_acceptance.py` and `cluster_monitor_acceptance.py` (60-second interval, restored to 1,800 seconds afterwards).
- Threshold: every check passes. The recovery time becomes the measured basis of the recovery time objective.
