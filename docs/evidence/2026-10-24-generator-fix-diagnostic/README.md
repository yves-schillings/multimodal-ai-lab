# Generator fix diagnostic - 24 October 2026

Status: **diagnostic, not a laboratory measurement.** It chose the correction for the generator defect that M3 and M1 found. The repeated M3 and M1 runs on the deployed laboratory are the measurements.

## Setup

- Local Ollama 0.32.13 on the workstation, the same version as the laboratory, with `qwen3:4b` at the pinned digest `359d7dd4bcda`, the Q4_K_M quantization of Qwen3-4B-Thinking-2507.
- 16 cases from the [shared fictional corpus](../2026-10-24-measurement-corpus/): 10 answerable questions (4 English, 3 French, 3 Dutch) with their marked passages, and 6 unsupported English questions with their near-miss passages. Each case gives the model three passages, as the service does.
- Temperature 0, `think` false unless stated, 600 output tokens at most. The validation is the service's own: an answer object with `abstained` and one to three claims, each quote an exact substring of its numbered evidence.

## Results

| Variant | Answerable: correct answer | Answerable: other outcomes | Unsupported: correct refusal | Unsupported: other outcomes |
| --- | --- | --- | --- | --- |
| Current service call: JSON prompt, free `json` format | 2 of 10 | 3 prompt echoes, 4 invalid JSON, 1 citation withheld | 0 of 6 | 4 prompt echoes, 2 related facts |
| **Fix: plain-text evidence, output constrained to the answer schema** | **9 of 10** | 1 citation withheld | 0 of 6 | 5 related facts, 1 citation withheld |
| Fix plus a stricter abstention rule | 9 of 10 | 1 citation withheld | 0 of 6 | 6 related facts |
| Fix plus a decision field before the claims | 7 of 10 | 3 citations withheld | 0 of 6 | 4 related facts, 2 citations withheld |
| Fix with `think` true | 0 of 10 | 10 empty responses | 0 of 6 | 6 empty responses |
| Fix plus a yes/no check of each claim by the same model | 9 of 10 | 1 citation withheld | 0 of 6 | 6 related facts kept |

A "related fact" is a true, exactly quoted statement from the evidence that does not answer the question, for example that a grey van was seen when the question asks for its registration number.

## Decision

- **Adopted:** plain-text evidence and an output constrained to the answer schema, in `lab/local_inference.py` and in the gateway (`lab/inference_gateway.py`), which rebuilds every generation request. It removes the prompt echo and makes the cited-answer path usable.
- **Not solved:** with thinking off, this 4B model does not abstain on near-miss questions, whatever the prompt, decision field or second check. With thinking on, Ollama 0.32.13 returns no answer under a schema. Abstention therefore remains a failed M1 threshold. The next options are an instruction-tuned generator or a larger model, each evaluated by M1 before any promotion.
- The fix does not address instructions hidden inside reviewed evidence. More completed answers may make more of them visible in the repeated M1 run.

## Files

| File | Content |
| --- | --- |
| `qwen_fix_experiment.py`, `qwen_fix_results.json` | Current call against the fix |
| `qwen_fix_experiment_2.py`, `qwen_fix_results_2.json` | Stricter abstention rule and decision field |
| `qwen_fix_experiment_think.py`, `qwen_fix_results_thinking.json` | The fix with thinking on |
| `qwen_verifier_experiment.py`, `qwen_verifier_results.json` | The fix with a yes/no check per claim |
| `SHA256SUMS` | Digests of every file in this folder |
