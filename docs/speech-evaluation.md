# French and Dutch synthetic speech measurement

On 7 October 2026, the actual Whisper base adapter was measured against twenty fictional clips generated with eSpeak NG 1.52.0: ten French and ten Dutch. The helper container ran with networking disabled and reused the prepared local model. No speech model was trained or fine-tuned for this run.

| Language | Clips | Reference words | Substitutions | Deletions | Insertions | WER | Audio duration | Processing duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 10 | 149 | 77 | 5 | 18 | 67.11% | 52.354 s | 24.037 s |
| Dutch | 10 | 138 | 74 | 1 | 23 | 71.01% | 61.277 s | 25.612 s |

WER is the sum of substitutions, deletions and insertions divided by the reference-word count. These are micro-averaged per-language results, not averages of clip percentages. The reference normalisation keeps accents and apostrophes, applies NFKC and case folding, and ignores punctuation. No words were removed from the recognised output to improve the score. A clip WER can exceed 100% when insertions are numerous.

The high error rates show that this adapter is unreliable on this particular formant-generated fixture set. They establish a measured failure case, not a general error rate for French or Dutch speakers. A representative benchmark still needs separately labelled human recordings, accents, noise, recording devices and reviewer correction effort. No deployment-quality acceptance is inferred from these results.

The recogniser used faster-whisper 1.2.1, PyAV 16.1.0, CPU INT8, explicit language, voice activity detection and beam size five. The first clip includes initial model-loading overhead. The report includes exact model-file hashes, clip hashes, references, hypotheses and individual error counts. Its generated audio contains only original fictional text.

## Reproduce on Windows PowerShell

From the repository root, with Docker Desktop running and Python dependencies installed:

```powershell
python scripts/prepare_speech.py
docker compose -f deploy/docker/compose.yaml build app
docker build -t multimodal-ai-lab-speech-evaluation tools/speech-evaluation
$speechRepo = (Get-Location).Path
$speechOutput = Join-Path $speechRepo 'data/speech-evaluation'
New-Item -ItemType Directory -Force -Path $speechOutput | Out-Null
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges -e LAB_EVALUATION_NETWORK_DISABLED=1 -v "${speechRepo}:/work:ro" -v "${speechRepo}/models/whisper-base:/weights:ro" -v "${speechOutput}:/output" multimodal-ai-lab-speech-evaluation --corpus /work/tests/fixtures/speech_fr_nl.json --audio-dir /output/audio --model-dir /weights --report /output/speech-fr-nl.json --generate
```

Building downloads the separate synthesiser packages. Generating and recognising the fixtures then runs without network access. The application image does not acquire eSpeak NG. Its third-party licence remains GPL-3.0-or-later, separate from this repository's Apache-2.0 code licence.

The committed fixture specification is [speech_fr_nl.json](../tests/fixtures/speech_fr_nl.json). The dated report and exact tested audio are in [the evidence directory](evidence/2026-10-07-closing/). Timing is host-dependent; corpus and model hashes identify the measured inputs.

Sources: [eSpeak NG project](https://github.com/espeak-ng/espeak-ng), [supported languages](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md). These document the generator, not the accuracy of the recogniser.
