# RAG search strategies under the same guards (28 October 2026)

This package compares four search strategies on the same synthetic cases, rights, models, prompts, limits and machine:

| Variant | What changes |
|---|---|
| `fixed` | One search with the question: the three nearest passages at or above 0.45 cosine similarity (the `VectorIndex` rule), then one Qwen3:4b answer |
| `fixed_reranked` | The ten nearest passages, reordered by BAAI/bge-reranker-v2-m3; the three best are kept; then one answer |
| `agentic` | The unchanged `AgenticSearch` with the `fixed` retriever: Qwen3:4b may search again up to three times |
| `identifier_linked` | `fixed_reranked` plus the identifier link of `lab/rag_variants.py`: no passage that names another identifier reaches the generator, and a missing or ambiguous link withholds the answer |

Everything is synthetic or fictional. Nothing here is a quality, security or production acceptance.

## Files

| File | Content |
|---|---|
| [`eval_set.json`](eval_set.json) | 15 cases, 108 questions of eleven types and 3 access probes, with expected, forbidden, stale, injected and unauthorised values, written before any run |
| [`protocol.json`](protocol.json) | Questions, variants, controlled conditions, known differences, split and freeze rules, metrics and limits, written before any run |
| [`dev-reranker-threshold-analysis.json`](dev-reranker-threshold-analysis.json) | Development only: reranker probabilities of the expected passages, used to drop the probability threshold |
| [`dev-run-1-stopped-gpu-fa.json`](dev-run-1-stopped-gpu-fa.json) | First development run, stopped after 66 trials when stray characters appeared in generated quotes |
| [`dev-run-2.json`](dev-run-2.json), [`dev-run-2-summary.json`](dev-run-2-summary.json) | Development run of the four variants on C01-C05 (148 trials, GPU, about 31 minutes) and its summary |
| [`dev-run-3-identifier-linked-cpu.json`](dev-run-3-identifier-linked-cpu.json) | The identifier-linked variant after its two development corrections, CPU only |
| [`development-changes.json`](development-changes.json) | Every change made after a development run, with its reason, and the settings a future test run would freeze |

## Status

**Development results only.** The five development cases (37 questions) served to find bugs and tune the variants, so their numbers are optimistic. **The held-out test run on the ten other cases (71 questions) was not done**: the workstation GPU became unstable under load, and the article was closed on the results already acquired. The held-out questions have never been run.

## Development results (dev-run-2, one run, GPU)

| Variant | Correct (of 37) | Wrong or unsupported | Wrong abstentions | Withheld by the quote check | Right object cited | Abstained on unanswerable (of 9) | Model calls | Median time |
|---|---|---|---|---|---|---|---|---|
| fixed | 20 | 13 | 4 | 5 | 9 of 13 | 2 | 0.97 | 12.9 s |
| fixed_reranked | 17 | 14 | 6 | 6 | 8 of 12 | 0 | 1.00 | 10.0 s |
| agentic | 22 | 10 | 5 | 4 | 8 of 11 | 4 | 2.05 | 13.0 s |
| identifier_linked | 19 | 10 | 8 | 10 | 9 of 10 | 4 | 1.76 | 12.8 s |

- With 37 questions the 95 % Wilson intervals of the correct share overlap widely (about ±15 points): no variant is shown better overall.
- The identifier link cited the right object most often and followed none of the three injected instructions (the others followed one or two).
- Every variant often answered unanswerable questions with a related fact instead of abstaining.
- Qwen3:4b sometimes damaged a long quote under the JSON schema (for example `"Rue des Tanne: 17:40"`). The quote check withheld these answers. The raw outputs of withheld trials are stored in the run record.
- After two development corrections (see `development-changes.json`), the identifier-linked variant gave 20 correct of 37 instead of 19 (dev-run-3, CPU). The corrected questions passed, three others got worse and one injected instruction was followed. The gain was measured on the questions used for the corrections, on another device, so it does not show that the correction generalises.

## Reproduction

```powershell
# 1. In the WSL PyTorch environment: the reranker on the CPU, with the GPU hidden
wsl -d Ubuntu -e bash -c "cd /mnt/e/<path>/multimodal-ai-lab && CUDA_VISIBLE_DEVICES= <python> -B scripts/serve_reranker.py --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --port 8790 --cpu"
# 2. In the repository environment on Windows (Ollama with the pinned qwen3:4b and bge-m3:latest)
$rag = "docs\evidence\ai-engineering-review-20261028\rag"
.\.venv\Scripts\python.exe -B scripts\benchmark_rag_variants.py run --eval-set "$rag\eval_set.json" --protocol "$rag\protocol.json" --split dev --min-probability 0.0 --cpu-only --out "$env:TEMP\dev-run.json"
.\.venv\Scripts\python.exe -B scripts\benchmark_rag_variants.py summarize --run "$env:TEMP\dev-run.json" --out "$env:TEMP\dev-summary.json"
```

Dev-run-2 used the GPU (no `--cpu-only`); dev-run-3 used `--cpu-only --variants identifier_linked`. Dev-run-1 was stopped and is kept only as a record. A generation-integrity check on three device settings was started and then stopped; its result was not saved.

The reranker weights are BAAI/bge-reranker-v2-m3 at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (weights SHA-256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`), downloaded once into the local Hugging Face cache. The Ollama models are pinned by digest in the script. A run refuses an existing output file.

## Code

- [`lab/rag_variants.py`](../../../../lab/rag_variants.py): the exact in-memory retriever, the reranker client and retriever, the fixed and identifier-linked controllers.
- [`lab/rag_benchmark.py`](../../../../lab/rag_benchmark.py): the judge and the summary statistics.
- [`scripts/benchmark_rag_variants.py`](../../../../scripts/benchmark_rag_variants.py): builds the store, runs the variants, writes the run record and the summary.
- [`scripts/serve_reranker.py`](../../../../scripts/serve_reranker.py): the loopback reranker service, run in the WSL PyTorch environment.
- [`scripts/check_generation_integrity.py`](../../../../scripts/check_generation_integrity.py): the generation integrity check.
- [`tests/test_rag_variants.py`](../../../../tests/test_rag_variants.py): scripted tests of the controllers and the judge.
