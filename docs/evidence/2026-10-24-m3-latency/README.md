# M3 latency and throughput per stage - 24 October 2026

Status: **executed.** M3 has no pass or fail threshold: it records the CPU baseline that the service level objectives, the other measurements and the GPU sizing of the secondary data center will use. It also exposed a functional defect of the deployed generator, reported below.

## Runtime

- OpenShift Local (CRC) 2.64.0, OKD `4.22.0-okd-scos.10`, single node with 6 vCPUs and 24 GiB RAM on an Intel i9-9900K host. No GPU reaches the cluster.
- `lab-app` image `sha256:8cee7703d342f30dc04e730ff3b39dce13924551ee49dd005899f496d524a7c6`, limited to 2 CPUs and 3 GiB.
- `lab-inference`: Ollama 0.32.13, limited to 3 CPUs and 5 GiB, with `OLLAMA_KEEP_ALIVE=0`, `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_MAX_LOADED_MODELS=1`. BGE-M3 and Qwen3:4b (Q4_K_M, 4.0 B parameters) are pinned to the digests in [the local implementation guide](../../deployment/local-implementation.md).
- `lab-vector`: PostgreSQL 16 with pgvector, limited to 1 CPU and 768 MiB.
- Repository commit `153e10b`. The measured modules (`lab/speech.py`, `documents.py`, `rag.py`, `vector_retrieval.py`, `local_inference.py`, `retrieval.py`) are byte-identical in the running image and in the repository; full identities are in [`runtime-context.json`](runtime-context.json).

## Method

- [`scripts/measure_stage_latency.py`](../../../scripts/measure_stage_latency.py) runs each stage **inside the deployed `lab-app` pod**, with the service functions and configuration, 30 times. The 95th percentile is the nearest-rank value: the 29th of 30 sorted times.
- Transcription: the 10 French and the 10 Dutch fictional eSpeak NG clips of the [7 October evidence](../2026-10-07-closing/speech-audio/), each transcribed three times, with Whisper base on CPU int8 after one model load.
- OCR: one fictional scanned PDF page (300 dpi image, OCR path) and one fictional PNG page (200 dpi) from the [OCR measurement](../2026-10-24-ocr-quality/).
- Embedding, retrieval and generation: 30 questions of the [shared corpus](../2026-10-24-measurement-corpus/) (10 per language) on the 60-passage case of officer-a. Every retrieval returned three passages, so every generation received three sources.
- Ollama's own timing fields are recorded beside each wall-clock time, so model loading can be separated from computation.
- The cited-answer path runs through the authenticated HTTP API over a loopback port-forward. Each simulated user has its own session; users 1 and 3 are officer-a, users 2 and 4 officer-b on an identical corpus case. One level sends 30 questions in total (32 with four users).

## Results per stage

| Stage | n | Median | 95th percentile | Detail |
| --- | --- | --- | --- | --- |
| Transcription, French clip (Whisper base) | 30 | 1.80 s | 17.41 s | Real-time factor: median 0.34, p95 3.06, aggregate 0.67. Model load 0.73 s. |
| Transcription, Dutch clip (Whisper base) | 30 | 1.79 s | 10.29 s | Real-time factor: median 0.28, p95 1.90, aggregate 0.59. Model load 0.79 s. |
| OCR, scanned PDF page | 30 | 0.68 s | 1.36 s | pypdf finds no text, PDFium renders the page, Tesseract reads it. |
| OCR, PNG page | 30 | 0.59 s | 0.77 s | Tesseract only. |
| Question embedding (BGE-M3) | 30 | 2.15 s | 2.21 s | Model loading takes 2.02 s of it: `keep_alive=0` reloads the model for every call. |
| Vector search, SQL only (pgvector) | 30 | 0.01 s | 0.02 s | Exact cosine search over 60 rows, `LIMIT 3`. |
| Retrieval call (embedding, search, checks) | 30 | 2.18 s | 2.40 s | The full service call, dominated by the embedding. |
| Answer generation (Qwen3:4b) | 30 | 58.89 s | 77.66 s | Model load median 4.34 s, 6.6 output tokens per second. Only **4 of 30** answers passed the citation check. |

Indexing: one BGE-M3 batch over the 60 passages (12,815 characters), which the service embeds again after every revision change of the case, took a median of 17.01 s and at most 20.46 s over 5 repetitions ([`indexing-raw.json`](indexing-raw.json)).

## Speech real-time factor against 0.46 and 0.42

The 7 October values (0.46 French, 0.42 Dutch) were aggregates over the same 20 clips in a host container without a CPU limit. In the cluster pod, limited to 2 CPUs, the **aggregate** rises to **0.67 French and 0.59 Dutch**, while the **median clip** is faster (0.34 and 0.28). The aggregate is dominated by a few clips on which faster-whisper fails its confidence checks and decodes again at higher temperatures: clip `fr05` (5.7 s of audio) took 16 to 20 s each time, and `nl03`, `nl04` and `nl07` took 6 to 11 s. The same retries make repeated transcriptions of one clip differ. As a consistency check, the first pass over the 20 clips gives a word error rate of 81.2% in French and 63.0% in Dutch, against 67.11% and 71.01% on 7 October: on these synthetic clips Whisper base remains unreliable, and the human recordings of M4 are still needed.

## Generation defect found by M3

26 of the 30 generations were withheld (`InferenceUnavailable`, HTTP 503 through the API). In all 26 the model did not answer: it **reproduced the prompt JSON** (`{"question": ..., "evidence": [...]}`) instead of returning `{"abstained": ..., "claims": [...]}`. In 9 of them the copy was cut off before it closed, so the output was not even valid JSON. The citation check withheld every one of them, so no unchecked text reached the user: the failure is safe but makes the cited-answer path unusable for most questions with three sources. With a single short source, the original acceptance example still answers correctly. The likely cause is the combination of a JSON-only output constraint, a prompt that is itself a JSON document and a thinking model run with `think=false`. A corrected prompt is a separate, tested change of the service; M1 and M3 must then be repeated.

## Cited-answer path with 1, 2 and 4 simultaneous users

| Simultaneous users | Requests | Cited answers (HTTP 200) | Withheld (HTTP 503) | Median latency, all requests | 95th percentile, all requests | Median, cited answers | Requests per minute |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 4 | 26 | 63.9 s | 83.2 s | 48.8 s | 0.93 |
| 2 | 30 | 3 | 27 | 124.2 s | 211.3 s | 50.2 s | 0.92 |
| 4 | 32 | 0 | 32 | 185.4 s | 275.4 s | - | 1.29 |

- **Throughput does not grow with users.** Ollama runs one request at a time (`OLLAMA_NUM_PARALLEL=1`), keeps one model loaded and unloads it after each call, so every question loads BGE-M3 and then Qwen3:4b again. The lab completes about one cited-answer request per minute whatever the number of users; latency grows with the queue instead.
- **Four users exceed the inference timeout.** The service waits at most 180 s for Ollama. At 4 users, 23 of the 32 requests reached that limit: the Ollama request log records 23 generation calls cancelled at exactly 3 min 0 s, and none at 1 or 2 users. The other 9 completed but failed the citation check. The higher request rate at 4 users comes from requests abandoned at the timeout, not from more work done.
- At 1 and 2 users, every withheld answer is the prompt-echo defect described above.
- The warm-up question of each case, recorded apart, also ended in HTTP 503 after 50.2 s (officer-a) and 64.1 s (officer-b, including the first indexing of its case).
- **Service levels:** on this CPU baseline the cited-answer path serves one user with a median of about 64 s and a 95th percentile of about 83 s, but only for questions whose answer passes the citation check. It cannot serve 4 simultaneous users within its own 180 s limit.

## Files

| File | Content |
| --- | --- |
| `runtime-context.json` | Cluster, images, limits, Ollama settings, model digests and source hashes of the running pod |
| `stages-raw.json` | Every repetition: times, Ollama timing fields, transcripts of the synthetic clips, generation outcome and the first characters of each withheld response |
| `stages-summary.json` | Medians, 95th percentiles and derived rates per stage |
| `concurrency-raw.json` | Every request of the 1, 2 and 4 user levels with its status, abstention, claim count and latency |
| `indexing-raw.json` | Five timings of the 60-passage BGE-M3 indexing batch |
| `ollama-generate-log.txt` | The Ollama request log lines (UTC time, status, duration) of the 124 generation calls of the stage and concurrency runs, used to count the timeouts; the client address is replaced by "app pod" |
| `SHA256SUMS` | Digests of every file in this folder |
