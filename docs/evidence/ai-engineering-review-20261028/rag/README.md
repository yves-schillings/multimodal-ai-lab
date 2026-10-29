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
| [`development-changes.json`](development-changes.json) | Every change made after a development run, with its reason, and the settings the test run froze |
| [`test-freeze.json`](test-freeze.json), [`test-analysis-plan.json`](test-analysis-plan.json) | The frozen inputs and code hashes, and the analysis plan, written before the held-out run |
| [`test-run-cpu-restart-2-20261029.json`](test-run-cpu-restart-2-20261029.json) | The complete held-out run: 876 trials, CPU only |
| [`test-run-incidents.json`](test-run-incidents.json) and the three partial `test-run-*` files | The interrupted attempts and why they stopped; never pooled |
| [`test-comparison.json`](test-comparison.json), [`test-repeatability.json`](test-repeatability.json) | The paired case-clustered comparison and the repeatability check |
| [`test-review-sheet.json`](test-review-sheet.json), [`test-review-sheet-key.json`](test-review-sheet-key.json) | The blinded sample for an independent reviewer and its key |

## Status

**Held-out test run complete (29 October 2026).** The ten held-out cases (71 scored questions and 2 access probes) were frozen in [`test-freeze.json`](test-freeze.json), with the analysis fixed in [`test-analysis-plan.json`](test-analysis-plan.json), before any of them was run. The complete run is [`test-run-cpu-restart-2-20261029.json`](test-run-cpu-restart-2-20261029.json): 876 trials, 3 repeats, CPU only, hashes matching the freeze. Three earlier attempts were interrupted and are kept unpooled ([`test-run-incidents.json`](test-run-incidents.json)). The development results below remain optimistic and are kept for the record.

## Held-out results (test-comparison.json)

Strict success: a correct answer supported by a cited quote and naming no other object, or a deliberate abstention where none was possible. An answer withheld by the quote check, a deadline or an invalid plan is a failure. Each question weighs the same; the interval resamples the ten cases.

| Strategy | Strict success | Answers correct | Withheld | Answered the unanswerable | Wrong object | Injected instruction followed (of 12) | Model calls | Median time | 95th percentile |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 45.1 % | 96 | 39 | 39 | 3 | 9 | 1.00 | 17.6 s | 29.9 s |
| fixed_reranked | 50.7 % | 108 | 39 | 39 | 12 | 12 | 1.00 | 20.3 s | 38.1 s |
| agentic | 46.5 % | 99 | 45 | 33 | 3 | 9 | 2.13 | 29.4 s | 47.2 s |
| identifier_linked | 53.5 % | 114 | 45 | 36 | 0 | 12 | 1.73 | 30.0 s | 46.6 s |

Counts are trials (213 per strategy, 71 questions x 3 repeats). No trial reached the 90 s deadline and all 24 access probes were refused.

| Contrast (plan) | Difference | 95 % case bootstrap | Sign-flip check |
|---|---|---|---|
| **Primary:** agentic minus fixed_reranked | -4.2 points | -14.3 to +5.7 | 0.63 |
| identifier_linked minus fixed_reranked | +2.8 points | -2.8 to +10.0 | 0.75 |
| fixed_reranked minus fixed | +5.6 points | -4.2 to +15.5 | 0.44 |

- **The primary criterion is not met.** The plan required a gain of at least 10 points with a lower bound above 0; the agentic strategy was 4.2 points below the reranked one. Leaving out any one case keeps the difference negative.
- **No contrast is established.** Every interval contains 0 and every Holm-adjusted sign-flip value of the six exploratory pairs is 1.0.
- **The identifier link** cited no other object in 54 answers and won 3 questions against 1 for fixed_reranked, two of them in case C14.
- **Common failures:** every strategy answered most unanswerable questions with a related fact (0 to 3 correct abstentions of 33), and multi-step questions stayed weak (3 to 15 of 36).
- **Repeatability:** the three repeats of the complete run gave identical verdicts, but an interrupted attempt on the same files differed on 16 of 479 trials ([`test-repeatability.json`](test-repeatability.json)). Differences of a few questions can come from service restarts alone.
- **Judgements are automatic.** The blinded sheet [`test-review-sheet.json`](test-review-sheet.json) (74 items, key in [`test-review-sheet-key.json`](test-review-sheet-key.json)) awaits an independent reviewer. It covers the fixed sample of the plan and every one of the 159 trials the judge flagged in the three repeats. Repeats with an identical answer, quotes and verdict share one item, and the key lists the repeats each item covers. The first export covered the flagged trials of repeat 1 only (106 flagged trials of repeats 2 and 3 were missing, all identical to repeat 1); it was replaced before any review.

Rerun the analysis:

```powershell
$rag = "docs\evidence\ai-engineering-review-20261028\rag"
.\.venv\Scripts\python.exe -B scripts\compare_rag_strategies.py --run "$rag\test-run-cpu-restart-2-20261029.json" --freeze "$rag\test-freeze.json" --plan "$rag\test-analysis-plan.json" --eval-set "$rag\eval_set.json" --out "$env:TEMP\test-comparison.json" --review-out "$env:TEMP\test-review-sheet.json"
```

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
- [`lab/rag_paired_analysis.py`](../../../../lab/rag_paired_analysis.py) and [`scripts/compare_rag_strategies.py`](../../../../scripts/compare_rag_strategies.py): the paired analysis of the plan.
- [`tests/test_rag_variants.py`](../../../../tests/test_rag_variants.py): scripted tests of the controllers and the judge.
