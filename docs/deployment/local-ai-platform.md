# Free local AI workbench, training and serving

This CPU-only OKD extension combines **JupyterLab, a native Kubernetes training Job, MLflow and KServe Standard serving**. It supplies these specific capabilities, rather than installing the complete Red Hat OpenShift AI product or Open Data Hub dashboard. The existing lab, accounts, model release and RAG remain separate.

The accepted local cluster has six vCPUs and 24 GiB RAM. The [Open Data Hub installation prerequisites](https://opendatahub.io/docs/installing-open-data-hub/) require at least 16 CPUs and 32 GiB worker memory for the full installation. This smaller exercise uses [KServe Standard mode](https://kserve.github.io/website/docs/admin-guide/kubernetes-deployment), without Knative or Istio, and the [official Jupyter Docker image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html). No GPU passthrough or paid service is used.

## Prepare official controller releases

Use an existing CRC administrator kubeconfig. The scripts permit only `api.crc.testing`. Keep manifests, generated credentials and acceptance reports in an owner-only directory outside Git. Never copy kubeconfig or Secret data into source control.

Download these official release files into that directory before installation:

| Private filename | Official release source | Required SHA-256 |
| --- | --- | --- |
| `cert-manager-v1.21.2.yaml` | [cert-manager 1.21.2](https://github.com/cert-manager/cert-manager/releases/download/v1.21.2/cert-manager.yaml) | `e03b668ec8675214af6b0a671699d088f2601fa3878e0dbe1b41d3feafd1879f` |
| `kserve-v0.20.0.yaml` | [KServe 0.20.0](https://github.com/kserve/kserve/releases/download/v0.20.0/kserve.yaml) | `6e6c1a5ceeb889e6ac41d68a551fa8c7d1fc1abcba10284154a0872bcc0d3fb8` |

The installer refuses changed manifest hashes. It installs cert-manager and the predictive KServe controller, adapts containers to restricted non-root security and disables public ingress creation. LLM/local-model controllers and their workloads are omitted. Controller installation alone does not establish successful inference.

```powershell
python scripts/install_local_ai_controllers.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-platform-dir>
python scripts/deploy_local_ai_workloads.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-platform-dir>
python scripts/cluster_ai_platform_acceptance.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-platform-dir>
```

The existing main application must already have an accepted active classifier and working MLflow. The workload script verifies and copies that classifier into a read-only serving snapshot; it never promotes a new main-lab release. It pins the current local app image by digest and uses the fixed Jupyter image `quay.io/jupyter/minimal-notebook@sha256:d65a4aef5c169c199e91232b47932c283fda2dc56ba157967def4d05f7b4e408`.

## Actual workloads

| Component | What it does | Access and storage |
| --- | --- | --- |
| JupyterLab workbench | Python notebook editing and kernel execution; private classifier calls | Token-required contents API; loopback operator port-forward; its own persistent work directory mounted at `/work` for arbitrary OKD UIDs |
| Native CPU training Job | Fits TF-IDF (Term Frequency–Inverse Document Frequency) plus logistic regression on 24 fictional examples, evaluates eight separate examples and logs metrics/model artifacts to MLflow | A separate training store and registry name; private persistent `training.json`; no automatic production promotion |
| KServe `document-classifier` | Runs the accepted classifier through a custom V2 prediction API, checking the artifact SHA-256 before loading | Internal Service, bearer credential, one static replica; at most eight bounded texts per request |
| Existing MLflow | Stores the actual independent training run and artifact evidence | Only labelled training pods gain the additional network path; existing main-lab data is retained |

The namespace is `lab-ai-platform`. Workloads have restricted security, read-only root filesystems, no mounted Kubernetes credential and bounded temporary storage. Default-deny network policies permit private predictor access and the scoped training-to-MLflow path. Internet egress is denied. Two separate 1 GiB PVC requests retain notebook and training results; CRC hostpath remains a shared physical filesystem, not a hard per-PVC consumption quota.

The namespace admission budget is 800 mCPU requested, three CPUs limited, 2 GiB memory requested and 4 GiB limited, with four pods and no GPU, NodePort or load balancer allocation. Workbench and training share scarce CPU/memory capacity: the deployment script scales this owned workbench to zero while training, then restores it. It does not stop unrelated workloads. Rerunning the script intentionally creates another training Job; do not repeat it merely to inspect an existing accepted run.

## Use the workbench

```powershell
oc --kubeconfig <local-admin-kubeconfig> port-forward service/lab-workbench -n lab-ai-platform 18888:8888 --address=127.0.0.1
```

Open `http://127.0.0.1:18888/lab` and use the private Jupyter token from `workbench-access.json`. Keep it private. The acceptance exercise saves `Local-model-training.ipynb`, explaining the real training result and how to call the private classifier. `/results/training.json` is mounted read-only. Notebook cells are editable; an acceptance-created notebook is not itself proof that every cell was executed.

For an operator prediction check, forward `service/document-classifier-predictor` on its port 80, then call `/v2/models/document-classifier/infer` with the private bearer token. Do not expose the token in command history. There is no public Route; KServe's computed example-domain URL is not a deployed public endpoint.

## Acceptance and limits

The acceptance script checks actual model output and artifact identity, rejection without credentials, completed training, independent MLflow run/metric readback, authenticated notebook APIs, a real kernel start, a private workbench-to-predictor request and Internet denial. It then replaces only the workbench pod and reads back the saved notebook while confirming the same PVC UID.

This is genuine small classifier training, **not LLM fine-tuning**. Its synthetic scores do not establish operational quality. Whisper base, BGE-M3 and Qwen3:4b remain pretrained models. The installation excludes the full OpenShift AI/ODH dashboard, pipeline operator, distributed training, production identity, GPU serving and automatic model drift remediation. KServe uses one static predictive replica; CPU autoscaling is not claimed.

Inspect logs and retained failed Job evidence before retrying a failed exercise. Preserve every existing volume. Do not reset CRC or delete namespaces/PVCs for routine diagnosis. Apply [CPE model isolation](local-cpe.md) separately: this workbench is an operator training environment, not an automatically approved tenant endpoint.
