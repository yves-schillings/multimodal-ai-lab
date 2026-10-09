# Accepted local AI extensions

Actual execution on 9 October 2026: CRC 2.64.0 / OKD 4.22.0, six logical CPUs and 24 GiB RAM. Models run on CPU with prepared offline weights. All records and training samples are fictional. This is a bounded local exercise, not production acceptance or the complete Red Hat OpenShift AI product.

## Verified scope

| Exercise | Actual result |
| --- | --- |
| Each CPE, A and B | 39 verification checks and 13 lifecycle checks passed. Includes cited RAG, abstention, corrected-source indexing, case/session rules, network/database denials, activation, expiry and application persistence. |
| Each CPE browser | Six actual Chromium checks passed: authenticated cases load, workspace is visible, model administration and audio input are hidden, no forbidden model-admin request is issued and no JavaScript error occurs. |
| Application replacement | Both CPEs still generated cited answers from their retained reviewed sources after replacing their application pods. |
| Vector database replacement | Both CPE RAG answers passed afterwards and the original vector PVC identity was retained: three checks. |
| Free local AI platform | 15 checks passed after the CRC restart: actual KServe prediction/credential denial/artifact identity, native training Job, independent MLflow run readback, Jupyter authentication/kernel/private prediction, Internet denial and notebook/PVC persistence. |
| CRC memory restart | All 11 existing PVC identities were retained. Only CRC was stopped/restarted; Windows and other applications remained open. |
| Main lab after extension | Four checks passed: authenticated pgvector/generation configuration, actual cited RAG answer after CRC/vector restart, monitoring enabled at 1,800 seconds with retained records and retained active classifier release. |
| Source checks | 104 Python tests plus 15 subtests, browser core/syntax checks and dependency audit passed. Source checks are distinct from the deployment exercises above. |

The preceding 16 GiB run stalled during inference and did not complete acceptance. The accepted 24 GiB run used a two-thread inference bound in the CPE gateway. Failed exploratory reports remain private; they are not counted as successful tests.

## Models and training

- BGE-M3 embeddings: `bge-m3:latest`, digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
- Qwen3:4b generation: digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
- Native CPU training: TF-IDF (Term Frequency–Inverse Document Frequency) plus logistic regression; 24 fictional training examples and eight separate held-out examples. Accuracy and macro F1 were 1.00 on that tiny set. This does not establish production accuracy.
- No LLM fine-tuning, GPU execution or automatic promotion occurred. The KServe predictor serves a verified copy of the accepted classifier, independent of the new training run.

The accepted CPE/gateway image digest is `8cee7703d342f30dc04e730ff3b39dce13924551ee49dd005899f496d524a7c6`. The main lab retains its accepted image6; the AI platform retains its accepted image10. Official controller hashes and the pinned Jupyter image appear in the [platform guide](../../deployment/local-ai-platform.md).

## Reproduce and interpret

Follow the [CPE guide](../../deployment/local-cpe.md), [platform guide](../../deployment/local-ai-platform.md) and [implementation guide](../../deployment/local-implementation.md). Use private generated credentials, an existing CRC administrator kubeconfig and fictional files. The cluster acceptance scripts perform actual requests and selected pod replacements; avoid running them during another workflow. Do not rebuild accepted images unless sources change, or remove PVCs as a recovery shortcut.

[acceptance.json](acceptance.json) contains curated check names and results. Original reports, source records, credentials and kubeconfig remain outside Git. Approval expires after 24 hours and changes require fresh verification. CRC hostpath PVC admission budgets are not hard filesystem consumption quotas. Organizational identity, TLS ingress, representative drift evaluation, full disaster recovery and GPU/full-platform installation remain separate work.
