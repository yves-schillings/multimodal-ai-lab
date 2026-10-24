# M3 after the generator fix - 24 October 2026

Status: **executed.** This repeats the generation stage and the cited-answer path of [the first M3 run](../2026-10-24-m3-latency/) after the generator fix of commit `b138b95`. Same scripts, corpus, questions and service settings; the first-run folder is unchanged. M3 has no pass or fail threshold.

## What changed, and what did not

- Commit `b138b95` gives the evidence to Qwen3:4b as plain numbered text instead of a JSON document, and constrains the output with an Ollama JSON schema (`ANSWER_SCHEMA`) instead of the free `json` format, in `lab/local_inference.py` and in the CPE gateway `lab/inference_gateway.py`. The [generator fix diagnostic](../2026-10-24-generator-fix-diagnostic/) explains the choice.
- Build `lab-app-13` produced image `sha256:497322ea83d7980a43ac311dc80e79bcb2dd4c03d95270b29e4e6085f4bcb6df`, replacing `sha256:8cee7703d342f30dc04e730ff3b39dce13924551ee49dd005899f496d524a7c6`. It runs in `lab-app` (both containers) and in `lab-inference-gateway`, deployed by digest. In both running pods the SHA-256 of every `lab/*.py` file and of `app.py` equals the source commit. Details in [`deployment.json`](deployment.json).
- The rebuilt image also carries changes already committed before the fix and absent from the old image: Pillow 13.0.0, pypdf 6.21.2, the MLflow 3.17.2 client, comment-only changes in `lab/cpe.py` and browser files. None is on the generation path.
- Unchanged: Ollama 0.32.13 and its settings (`keep_alive=0`, one parallel request, one loaded model, 3 CPUs, 5 GiB), pgvector, the service `LIMIT 3`, the 0.45 similarity threshold, the 180 s inference timeout and the app limits (2 CPUs, 3 GiB). The two CPE application deployments keep the old image; only their gateway changed, and they were not measured.
- Passage `P-EN-06`, corrected by the first M1 currency check, was reviewed back to its original text before the rerun (case revision 121 to 122), so the corpus is the authored one.
- The workstation froze at about 15:08 during the first attempt at the 4-user level. After the cluster restart, the deployed digests and source hashes were checked again and all three user levels were run again from the start. The generation stage had completed before the freeze and is kept.

## Generation stage, 30 questions

| Run | Answers passing the citation check | Withheld | Median | 95th percentile | Output tokens (median) | Output tokens per second | Model load (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Before the fix | 4 of 30 | 26: prompt copied back (17 complete, 9 cut off) | 58.89 s | 77.66 s | 272 | 6.6 | 4.34 s |
| After the fix | **25 of 30** | 5: a quote that is not an exact copy of the cited evidence | 55.81 s | 95.57 s | 135 | 3.8 | 5.49 s |

- The prompt echo is gone: every response is an answer object.
- The 5 withheld answers (`Q-EN-04`, `Q-EN-05`, `Q-FR-02`, `Q-FR-03`, `Q-NL-05`) quoted words that do not appear exactly in the cited passage; the citation check rejected them, as designed.
- Answers are about half as long, but decoding is slower: 3.8 output tokens per second against 6.6. Schema-constrained sampling is the likely cause; it was not isolated. The median time barely moves; the 95th percentile rises because five generations took 82 to 101 s.
- The stage counts outcomes only; answer correctness is measured by the repeated M1 run.

## Cited-answer path with 1, 2 and 4 simultaneous users

| Users | Run | Cited answers (HTTP 200) | Withheld (HTTP 503) | Median, all requests | 95th percentile, all requests | Median, cited answers | Requests per minute | Inference timeouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Before | 4 of 30 | 26 | 63.9 s | 83.2 s | 48.8 s | 0.93 | 0 |
| 1 | **After** | **24 of 30** | 6 | 46.9 s | 69.5 s | 45.4 s | 1.22 | 0 |
| 2 | Before | 3 of 30 | 27 | 124.2 s | 211.3 s | 50.2 s | 0.92 | 0 |
| 2 | **After** | **23 of 30** | 7 | 63.8 s | 124.9 s | 59.5 s | 1.60 | 0 |
| 4 | Before | 0 of 32 | 32 | 185.4 s | 275.4 s | - | 1.29 | 23 |
| 4 | **After** | **25 of 32** | 7 | 115.4 s | 213.2 s | 113.6 s | 1.93 | 1 |

- The fix turns the cited-answer path from mostly withheld into mostly answered: 72 of 92 requests answered, against 7 of 92 before. No answer abstained.
- Shorter answers raise throughput: 1.22 requests per minute with one user and 1.93 with four, because the queue for the single Ollama slot now moves faster and the application work of one user overlaps the generation of another.
- Four users still wait: median 115 s and 95th percentile 213 s. One generation reached the 180 s inference timeout (the Ollama log records one call cancelled at 3 min 0 s, [`ollama-generate-log.txt`](ollama-generate-log.txt)); before the fix, 23 did.
- Of the 20 withheld requests, one is that timeout. The other 19 completed their generation and were withheld by the service; the concurrency script records only the HTTP status, so their reason is inferred from the generation stage, where every withheld answer failed the exact-quote check.
- Warm-up questions, recorded apart: HTTP 200 after 44.2 s (officer-a, including re-indexing after the restored passage) and 34.0 s (officer-b).
- **Service levels on this CPU baseline:** one user receives a cited answer with a median of about 45 s and a 95th percentile of about 70 s; four simultaneous users push the median to about 2 minutes. About one answer in five is still withheld.

## Files

| File | Content |
| --- | --- |
| `deployment.json` | Build, previous and new image digests, running pods, source-hash check, unchanged settings and the corpus restoration |
| `runtime-context.json` | Cluster, images, limits, Ollama settings, model digests and source hashes recorded before the generation stage |
| `stages-raw.json`, `stages-summary.json` | The 30 generation repetitions and their summary |
| `concurrency-raw.json` | Every request of the 1, 2 and 4 user levels |
| `ollama-generate-log.txt` | The Ollama request log lines (UTC time, status, duration) of the 94 generation calls of the concurrency run; the client address is replaced by "app pod" |
| `SHA256SUMS` | Digests of every file in this folder |
