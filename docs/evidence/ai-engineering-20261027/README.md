# AI Engineering evidence — 27 October 2026

This package contains selected, sanitized records from local synthetic experiments. It distinguishes measured execution, failed criteria and acceptance still required. Private raw diagnostics and workstation paths are excluded. No production release or institutional deployment is claimed.

| Record | Meaning | Limits |
| --- | --- | --- |
| [`agentic-search-v3.json`](agentic-search-v3.json) | Actual pinned local-model search with direct, paraphrased and unsupported questions | Reviewed-only lexical retrieval, CPU, simulated case actors, three questions; no pgvector or general quality acceptance |
| [`explainability-comparison.json`](explainability-comparison.json) | Actual logistic-regression/tree comparison and numerical explanation checks | One artificial split; positive-case LIME fidelity fails; no causal, legal or production claim |
| [`vision-micro-earlier-run.json`](vision-micro-earlier-run.json) | Earlier trained tiny-CNN synthetic quality, cache and sequential timing results | Historical micro-benchmark; separate from the larger open-loop load probe |
| [`vision-resnet-open-loop.json`](vision-resnet-open-loop.json) | Actual ResNet-shaped open-loop queue, batching, precision and worker measurements | Random weights and synthetic inputs; no semantic model quality or production release eligibility |
| [`candidate-acceptance.json`](candidate-acceptance.json) | Pure evaluation of supplied evidence and content hashes | A technical evidence gate, never a human approval or registry transition |

The micro-benchmark and the completed ResNet-shaped load probe are separate records. Browser and API receipts, when available, are separate evidence from these command-line measurements.

## Search observations

The final run used real `qwen3:4b` CPU generation with its pinned digest. It produced a supported location answer in one search and answered a paraphrase after three searches: the model chose `location of parked cycle`, then `parked bicycle location`. Final generation still received the original question.

Two of three predefined question criteria passed. The unsupported registration-number question returned no answer after a repeated query was detected. This is safe withholding, but it did not meet the stricter expected model-abstention criterion, so the failed criterion is retained. Earlier grammar-compatibility and repeated-query failures remain in the private engineering record. They are not silently converted into successful runs.

The measured 37.046-second and 58.484-second durations concern a local CPU execution with a 90-second bound. They do not establish a low-latency service or accept the portal's separate 60-second profile. Exact supporting quotes establish a citation check, not factual entailment or the truth of a statement.

Behavior verification recorded 36 focused agent/inference tests. A broader agent, inference, authentication and case-workflow run recorded 57 tests and 11 subtests. Those fake-client control tests are separate from the real-model questions. The evidence-gate tests exercise artificial report fixtures and do not manufacture measured model outcomes.

## Explanation observations

The comparison uses the same synthetic dataset, training split and held-out rows for both models. All seven comparison checks pass numerically. The positive-case local surrogate fails: its raw output is approximately 1.425 rather than a valid probability, about 0.427 from the classifier output. The failure remains visible alongside successful SHAP reconstruction. Numerical additivity does not establish causality, fairness or a useful explanation for a domain expert.

## Earlier vision micro-benchmark

The earlier 1,362-parameter classifier was trained on 256 fictional stripe images and evaluated on 64 separate synthetic images, with zero image-hash overlap. CPU and CUDA each recorded 64 correct predictions. The run measured batch sizes 1 and 8, cache invalidation and sequential localhost HTTP. It did not execute YOLO, TensorRT, Triton or a cluster GPU deployment. A tiny synthetic classifier's accuracy and speed cannot substitute for representative object-detection quality or concurrent service capacity.

## Gate and provenance

The final recorded gate evaluates the current search run, the random-weight ResNet probe, and the single linear/tree comparison explanation record. Its 18 checks return `blocked`: the search's expected-abstention criterion, untrained vision weights, absent vision quality evaluation and the comparison positive-case LIME fidelity fail. This preserves successful performance/accounting checks without converting them into a valid model release. Human/domain and operational acceptance remain pending regardless of the technical checks. The API uses these same three input files and the same 18-check manifest. Since 28 October 2026, the gate also requires a passing two-step search case (`rag.multi_step_case`), so the same three files now return 19 checks and stay `blocked`. See the [search comparison of 28 October](../agentic-retrieval-20261028/README.md).

## Open-loop vision protocol

Five configurations each received 1,000 offered synthetic RGB requests over 10 seconds. The CPU configuration returned 177 successes and 823 HTTP 429 rejections. The four GPU configurations each returned 1,000 successes with no errors or client safety drops. FP32 with one/two workers recorded successful HTTP p95 32.99/17.96 ms; FP16 recorded 28.37/29.93 ms. The report includes p99, queue waits, scheduled-arrival latency, batch occupancy and complete offered/submitted/completed/error/drop accounting. All owned servers and batching workers were closed afterwards.

The 11,689,512-parameter network has random weights. Finite logits or a class index do not prove semantic labels, detection quality or model suitability. The GPU timing probe therefore remains release-ineligible. This is a short host/WSL experiment, without production image decoding, authenticated external requests or cluster execution. The process allocator peak is not the physical free memory of the GPU; unreliable WSL free/total readings are excluded from this export. GPU execution tests recorded 8 passed and 1 skipped; the native CPU environment recorded 6 passed and 3 skipped. Skipped CUDA tests on a CPU interpreter are not GPU acceptance.

Content hashes are computed over the supplied canonical JSON reports. Sanitized exports and raw reports can have different hashes because private or bulky fields are removed. A hash proves content identity only; these files are not signed attestations or proof of evidence completeness. No live cluster, two-project network, spend/token quota, production capacity or human release approval is inferred.

See the [engineering guide](../../engineering/README.md) for the code map and exact reproduction commands.
