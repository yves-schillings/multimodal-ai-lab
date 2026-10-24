# M4 human French and Dutch recordings - protocol, 24 October 2026

Status: **prepared, not executed.** No human recording exists yet and none is fabricated. M4 needs consenting speakers, so this folder holds only what the recording sessions and the evaluation need: the protocol, the fictional recording script, the consent form, the data templates and the evaluation command.

## Purpose

M4 measures how well speech recognition handles real human voices in French and Dutch, and how long a reviewer needs to correct the drafts. It closes the representative-speech gap: the only speech results so far come from formant-synthesised eSpeak NG clips (word error rate 67.11% in French and 71.01% in Dutch with Whisper base), which do not represent human speakers.

## Inputs

| Item | Location | Content |
| --- | --- | --- |
| Recording script | [`tests/fixtures/measurements/m4_recording_script.json`](../../../tests/fixtures/measurements/m4_recording_script.json) | 24 fictional statements per language: 20 core statements and 4 spares. The French and Dutch sets describe the same fictional situations, so the two languages stay comparable. |
| Consent form | [`consent-form.md`](consent-form.md) | French and Dutch texts, with an English reference. |
| Speaker register | [`speaker-register-template.csv`](speaker-register-template.csv) | Pseudonymous speaker identifiers and recording conditions. The register never holds names. |
| Recording manifest | [`recording-manifest-template.csv`](recording-manifest-template.csv) | One row per audio file, linking a speaker, a statement and a condition. |
| Review times | [`review-times-template.csv`](review-times-template.csv) | Reviewer correction time per recording and model. |
| Evaluation | [`scripts/evaluate_human_speech.py`](../../../scripts/evaluate_human_speech.py) | Word error rate, real-time factor and review minutes per audio minute, per language and model. |

## Speakers and consent

- Recruit at least **4 consenting adult speakers per language**, with different voices: a mix of pitch, age range, speaking rate and regional accent. Speakers may be colleagues or volunteers; nobody records a real statement.
- Each speaker signs the consent form before recording. Consent covers recording, local transcription, measurement and publication of **metrics only**. Audio and transcripts are never published.
- Each speaker receives a pseudonymous identifier such as `S-FR-01`. The link between the identifier and the person stays on the signed form, stored apart from the audio.
- A speaker may withdraw at any time before publication. Their audio, transcripts and rows are then deleted and the metrics are recomputed without them.

## Recording sessions

- **Minimum:** 20 core statements per language, spread over at least 4 speakers with at least 5 statements each.
- **Recommended:** every speaker reads all 20 core statements, which gives at least 80 recordings per language and lets the results separate speaker effects from statement effects.
- Speakers read each statement exactly as written, at a natural pace, without adding names or personal details. A misread statement is recorded again rather than corrected afterwards.
- Record mono WAV at 16 kHz or more, one file per statement, on a common office headset or laptop microphone. Note the device in the speaker register.
- Condition `quiet` is the default. Optionally, one speaker per language reads the spare statements 21 to 24 in condition `noisy` (an open office or a street-side room) to show the effect of background noise separately.
- Store audio, the speaker register and transcripts in an owner-only private folder outside the Git repository.

## Models

- **Whisper base** through the service adapter `lab.speech.SpeechEngine`: CPU, int8, explicit language, voice activity detection, beam size 5. This is the configuration deployed in the laboratory.
- **A larger prepared Whisper model**, proposed: `large-v3` converted for faster-whisper, prepared explicitly before the run (for example `python -c "from faster_whisper.utils import download_model; download_model('large-v3', output_dir='<private-dir>')"`). Record the exact model identifier and file hashes. It runs on the RTX 2080 Ti workstation GPU with `float16`, or on CPU with `int8` if no GPU is available; the report states which.

## Run

```powershell
python scripts/evaluate_human_speech.py --recordings <private>\recording-manifest.csv --audio-dir <private>\audio `
  --model base=models\whisper-base --model large-v3=<private>\whisper-large-v3@cuda/float16 `
  --review-times <private>\review-times.csv --report <private>\m4-report.json --public-summary docs\evidence\<date>-m4-human-speech\summary.json
```

The private report keeps hypotheses for the reviewers. The public summary keeps metrics, hashes and pseudonymous identifiers only.

## Reviewer correction time

- Two reviewers who did not record correct the Whisper base draft and the larger-model draft of each recording, in a balanced order so that half of the recordings are corrected with the base draft first.
- The reviewer listens to the recording and edits the draft until it matches the speech. The timer runs from opening the draft to saving the corrected text.
- The result is **review minutes per audio minute**, per language and model.

## Metrics and threshold

- **Word error rate** per language and model: substitutions, deletions and insertions summed over all recordings, divided by the reference words. Normalisation follows `lab.evaluation.normalize_words` (NFKC, case folding, punctuation ignored, accents and apostrophes kept).
- **Real-time factor** per language and model: processing seconds divided by audio seconds.
- **Review minutes per audio minute** per language and model.
- **Threshold:** the business owner agrees the acceptable word error rate and review time **before the first recording session** and writes them in this folder as `threshold.md`, dated and signed. Without that file, the run does not start.

## Decision

The results decide between Whisper base on CPU and a larger model on GPU for French and Dutch statements. A model that misses the agreed threshold is not promoted, whatever its speed.
