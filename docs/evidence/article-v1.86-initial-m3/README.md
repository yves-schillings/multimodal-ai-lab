# Initial M3 stage-run evidence

This report freezes the initial stage experiment. It does not claim completion of all M1–M5 protocols or of the 1, 2 and 4 simultaneous-user tests. The source application snapshot is commit 153e10bee7ea2794c92ef0946611c06f24ebdbf9. The measurement harness was an additional evaluation script. Its digest and the raw-report digests identify the retained evidence separately from the application commit.

## Tested configuration

The workstation demonstrator ran single-node OKD with 6 virtual CPUs and 24 GiB RAM. No GPU was available to the cluster. The application was limited to 2 CPUs and 3 GiB. Ollama was limited to 3 CPUs and 5 GiB, with keep_alive=0, one loaded model and one parallel request. BGE-M3 and Qwen3:4b were pinned. The question context contained three retrieved passages for each of the 30 fictional questions, ten per language.

## Result and interpretation

Output and citation controls accepted 4 of 30 generations and withheld 26. Seventeen were classified as prompt echoes and nine as invalid JSON. The nine invalid outputs were also truncated prompt copies, so they are part of the same set of 26. These technical controls do not establish semantic correctness for the accepted four. Median generation duration was 58.89 seconds and the nearest-rank 95th percentile was 77.66 seconds over all 30 attempts. Earlier short-source acceptance remains a separate result.

The application image was sha256:8cee7703d342f30dc04e730ff3b39dce13924551ee49dd005899f496d524a7c6. The measured application modules matched the source snapshot. The generator configuration explains the observed conditions, not a proven root cause. Further diagnosis, prompt changes and retesting are separate development work and are not prerequisites for completing this editorial report. Existing source reports are retained unchanged.

This is a frozen initial result. A later run must have a separate record. It must not overwrite or silently aggregate these 30 attempts.

## Source and report identities

- Application source: [153e10bee7ea2794c92ef0946611c06f24ebdbf9](https://github.com/yves-schillings/multimodal-ai-lab/tree/153e10bee7ea2794c92ef0946611c06f24ebdbf9).
- [Runtime context](runtime-context.json), [raw stage records](stages-raw.json), [stage summary](stages-summary.json), and [SHA-256 digests](SHA256SUMS).
- The original measurement runner and helper are retained as [measure_stage_latency.py](measure_stage_latency.py) and [measurement_common.py](measurement_common.py). They are source snapshots from the evaluation, separate from the application commit. Their original location was `scripts/`. Restore that location and the declared corpus fixtures before reproducing the command. These snapshots are not a standalone installer.

## All generation attempts

| Attempt | Question | Language | Outcome | Reason | Wall time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | Q-EN-01 | en | InferenceUnavailable | prompt_echoed | 44.6397 |
| 2 | Q-EN-02 | en | InferenceUnavailable | prompt_echoed | 69.1126 |
| 3 | Q-EN-03 | en | InferenceUnavailable | prompt_echoed | 61.1565 |
| 4 | Q-EN-04 | en | InferenceUnavailable | prompt_echoed | 54.0215 |
| 5 | Q-EN-05 | en | InferenceUnavailable | prompt_echoed | 64.953 |
| 6 | Q-EN-06 | en | InferenceUnavailable | invalid_json | 50.5838 |
| 7 | Q-EN-07 | en | claims |  | 42.884 |
| 8 | Q-EN-08 | en | claims |  | 38.3388 |
| 9 | Q-EN-09 | en | claims |  | 38.1771 |
| 10 | Q-EN-10 | en | InferenceUnavailable | prompt_echoed | 52.5827 |
| 11 | Q-FR-01 | fr | InferenceUnavailable | prompt_echoed | 54.8722 |
| 12 | Q-FR-02 | fr | InferenceUnavailable | invalid_json | 66.8968 |
| 13 | Q-FR-03 | fr | InferenceUnavailable | invalid_json | 58.8872 |
| 14 | Q-FR-04 | fr | InferenceUnavailable | prompt_echoed | 61.3143 |
| 15 | Q-FR-05 | fr | InferenceUnavailable | prompt_echoed | 77.6597 |
| 16 | Q-FR-06 | fr | InferenceUnavailable | prompt_echoed | 69.5131 |
| 17 | Q-FR-07 | fr | InferenceUnavailable | prompt_echoed | 57.2435 |
| 18 | Q-FR-08 | fr | InferenceUnavailable | invalid_json | 61.7627 |
| 19 | Q-FR-09 | fr | InferenceUnavailable | prompt_echoed | 58.4829 |
| 20 | Q-FR-10 | fr | InferenceUnavailable | invalid_json | 66.3128 |
| 21 | Q-NL-01 | nl | InferenceUnavailable | invalid_json | 54.3212 |
| 22 | Q-NL-02 | nl | InferenceUnavailable | prompt_echoed | 77.8697 |
| 23 | Q-NL-03 | nl | InferenceUnavailable | invalid_json | 52.3839 |
| 24 | Q-NL-04 | nl | InferenceUnavailable | prompt_echoed | 65.5287 |
| 25 | Q-NL-05 | nl | InferenceUnavailable | prompt_echoed | 61.7241 |
| 26 | Q-NL-06 | nl | InferenceUnavailable | prompt_echoed | 65.0083 |
| 27 | Q-NL-07 | nl | claims |  | 41.6232 |
| 28 | Q-NL-08 | nl | InferenceUnavailable | prompt_echoed | 60.882 |
| 29 | Q-NL-09 | nl | InferenceUnavailable | invalid_json | 47.7782 |
| 30 | Q-NL-10 | nl | InferenceUnavailable | invalid_json | 58.8917 |

## Editorial scope

OpenShift AI deployment has never been part of this Secloudis laboratory. No model, runtime, prompt or laboratory feature was changed to publish this report. The 26 withheld outputs are availability failures. They are not correct abstentions or accepted answers.
