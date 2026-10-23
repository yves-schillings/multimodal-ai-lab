# Reference release article-v1.85

This tag freezes the repository state that the article [Multimodal AI Lab — Architecture for On-Premises Deployment on Red Hat OpenShift AI](https://secloudis.com/ai-lab-multimodal-case-processing/) describes, so that an independent reviewer can reproduce its results. All data, accounts and documents are fictional.

## What this release contains

- The application, its tests and the acceptance scripts of the local laboratory.
- The hosted pipeline with three jobs: Python and browser tests with a dependency audit, the application container with an OCR acceptance and a container scan, and the separate MLflow tracking-server image with a version check, a health check and a container scan.
- The dated evidence packages under `docs/evidence`.

## Verified results

| Capability | Environment | Commit | Result |
| --- | --- | --- | --- |
| Review workflow | Docker Compose and local OKD | [0825cea](https://github.com/yves-schillings/multimodal-ai-lab/commit/0825cea), [06a0355](https://github.com/yves-schillings/multimodal-ai-lab/commit/06a0355) | 27 checks before a restart and 8 after it |
| Access and identity | Local OKD | [22bf1e5](https://github.com/yves-schillings/multimodal-ai-lab/commit/22bf1e5) | 16 authenticated workflow checks, denied cross-case reads |
| Project isolation (CPE) | Local OKD, two projects | [22bf1e5](https://github.com/yves-schillings/multimodal-ai-lab/commit/22bf1e5) | 39 verification, 13 lifecycle and 6 browser checks per project |
| Speech recognition | Workstation CPU | [61a005f](https://github.com/yves-schillings/multimodal-ai-lab/commit/61a005f) | Synthetic French and Dutch WER 67,11 % and 71,01 %, a failure result |
| Document extraction and OCR | Container without network | [61a005f](https://github.com/yves-schillings/multimodal-ai-lab/commit/61a005f) | Text, PDF and scanned-PDF extraction pass |
| Retrieval and cited generation | Local OKD | [22bf1e5](https://github.com/yves-schillings/multimodal-ai-lab/commit/22bf1e5) | 8 authenticated RAG checks: citations, quotations, abstention, refused cross-case access |
| Classifier lifecycle | Local OKD with MLflow | [6402428](https://github.com/yves-schillings/multimodal-ai-lab/commit/6402428), [22bf1e5](https://github.com/yves-schillings/multimodal-ai-lab/commit/22bf1e5) | Promotion and rollback with digest checks, scheduled canary run |
| LLM fine-tuning | Workstation | [98c17b7](https://github.com/yves-schillings/multimodal-ai-lab/commit/98c17b7) | LoRA adapter 18 of 24 correct, 0 of 4 malicious-source tests passed, not promoted |
| AI platform components | Local OKD | [22bf1e5](https://github.com/yves-schillings/multimodal-ai-lab/commit/22bf1e5) | JupyterLab, KServe prediction service and CPU training Job |
| Backup and restore | Local OKD | [22bf1e5](https://github.com/yves-schillings/multimodal-ai-lab/commit/22bf1e5), [98c17b7](https://github.com/yves-schillings/multimodal-ai-lab/commit/98c17b7) | 10 restore checks, 3 archives with matching digests, 11 restored-serving checks |
| Delivery pipeline | GitHub Actions | [d1f10fb](https://github.com/yves-schillings/multimodal-ai-lab/commit/d1f10fb) | [Run 40035603702](https://github.com/yves-schillings/multimodal-ai-lab/actions/runs/40035603702): 111 tests, 15 subtests, browser checks, dependency audit, container gate |

## Pinned components

- Accepted application image on local OKD: `sha256:2896f66810f524b07a5117b5589ceb4b71ee999b0e9eea21065e89c87700afb4`.
- Accepted CPE and gateway image: `sha256:8cee7703d342f30dc04e730ff3b39dce13924551ee49dd005899f496d524a7c6`.
- BGE-M3 embedding model, Ollama tag `bge-m3:latest`: digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`.
- Qwen3:4b generator, Ollama tag `qwen3:4b`, the Q4_K_M quantization of Qwen3-4B-Thinking-2507: digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
- MLflow 3.17.2 in the application and in the tracking-server image. Pillow 13.0.0, pypdf 6.21.2, FastAPI 0.142.2 and faster-whisper 1.2.1.

## Not proven by this release

- Representative human French and Dutch speech, retrieval relevance, prompt-injection resistance, latency and a full cluster rebuild. The article plans these as measurements M1 to M5.
- Quality on representative and difficult scanned documents.
- Organisational identity, production HTTPS, alert delivery and the complete Red Hat OpenShift AI product on a shared platform.
- A risk decision for the container findings that have no fixed version.

## Reproduce

```bash
python -m pytest tests -q
node tests/browser_core.test.mjs
docker compose -f deploy/docker/compose.yaml up -d --build --wait
python scripts/demo_e2e.py --phase before --base http://127.0.0.1:8770
```

The cluster acceptance scripts and their private inputs are described in [the local implementation guide](../deployment/local-implementation.md).
