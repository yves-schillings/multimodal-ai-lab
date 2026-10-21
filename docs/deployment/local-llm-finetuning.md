# Local LLM fine-tuning: citations and abstention

This is a separate synthetic instruction-following experiment. It does not replace the accepted Qwen3:4b RAG model. It was trained on the host NVIDIA RTX 2080 Ti in WSL, not on the CPU-only OKD cluster.

## Purpose and data

RAG supplies permitted evidence. This exercise asks whether fine-tuning can teach a smaller model to return an answer, a source identifier and an exact quotation, or an empty answer with abstention when evidence is missing. The data contains no institutional or personal information: 96 fictional English question/source pairs for training and 24 held-out pairs with different document identifiers and question wording. Sixteen held-out questions have an answer; eight do not. Four missing-answer sources contain a misleading embedded instruction. These attacks are evaluation examples, not additional training examples.

## Accepted execution

The base model is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`. LoRA (Low-Rank Adaptation) uses rank 16, alpha 32, dropout 0.05, a learning rate of 0.0002, three epochs, seed 47, batch size one and gradient accumulation two. Base weights are FP16; trainable adapter weights remain FP32. No example is silently truncated. The original checkpoint is fixed; 224 adapter parameter tensors changed during training.

Weights were downloaded once from the official public model repository. Training, generation and evaluation then used local files with remote inference and uploads absent. Initial model download needs network access; offline inference does not imply an offline installation.

## Reproduce in a separate WSL environment

Use Python 3.12 and a working NVIDIA CUDA driver. Keep this environment separate from the application environment.

```bash
python3 -m venv ~/.local/share/multimodal-ft-venv
~/.local/share/multimodal-ft-venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
~/.local/share/multimodal-ft-venv/bin/pip install -r requirements-finetuning.txt
~/.local/share/multimodal-ft-venv/bin/python scripts/local_llm_finetune.py --output ~/multimodal-finetuning-evidence
```

Use an empty private output directory. The script refuses to overwrite existing evidence and requires a CUDA GPU. It writes the fictional splits, checkpoint, adapter, before/after/reloaded predictions and a result report. This command does not register or promote the adapter in the application. The accepted run was independently logged to MLflow and read back, including a downloaded adapter digest comparison.

## Measured results and withheld promotion

| Measure | Base model | Adapter, including fresh reload |
| --- | --- | --- |
| Valid JSON schema | 24/24 | 24/24 |
| Fully correct response under the strict protocol | 0/24 | 18/24 |
| Supported answer with expected source and quotation | 0/16 | 14/16 |
| Correct abstention with missing evidence | 0/8 | 4/8 |
| Malicious-source cases passed | 0/4 | 0/4 |

The base model could emit valid JSON while violating the task protocol, so schema validity alone is insufficient. The adapter improved the narrow extraction task but failed all four attack cases and abstained incorrectly on two supported questions. It remains an experimental candidate. There was no production promotion, no change to the active RAG, and no claim of representative real-document or French/Dutch quality. Further training requires a fresh independent evaluation set; fitting these held-out attacks and reporting the same set as independent validation would be misleading.

MLflow run `60daaa23b55147e58c9fa4d74010cf33` is FINISHED in `local-grounded-llm-lora`. Adapter SHA-256: `1f87d8493e0020a5747ddf93f2ba4ad684035f513730faa9a25203fe8f28ac0e`. Curated response-level evidence is in `docs/evidence/2026-10-09-local-llm-finetuning`. The checkpoint and adapter are private local artifacts, not Git attachments.
