"""M4: word error rate and reviewer correction time on consented human French and Dutch recordings.

Prepared for the M4 protocol; it has not been run on human recordings yet. Whisper base runs through the
service adapter lab.speech.SpeechEngine (CPU, int8). A larger prepared model runs through faster-whisper
with the same decoding settings (explicit language, voice activity detection, beam size 5) on the device
chosen for it. Recordings and transcripts stay in private folders; the published report contains
metrics, hashes and pseudonymous identifiers only. Usage:

    python scripts/evaluate_human_speech.py --recordings <private-manifest.csv> --audio-dir <private-audio>
        --model base=<whisper-base-dir> --model large-v3=<prepared-dir>@cuda/float16
        [--review-times <private-review-times.csv>] --report <private-report.json> --public-summary <summary.json>
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.evaluation import transcript_metrics
from lab.speech import SpeechEngine, validate_audio

SCRIPT = ROOT / "tests" / "fixtures" / "measurements" / "m4_recording_script.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transcriber(name, spec):
    """'<dir>' uses the service adapter on CPU/int8; '<dir>@<device>/<compute_type>' uses faster-whisper directly."""
    if "@" not in spec:
        engine = SpeechEngine(model_dir=Path(spec))
        return lambda audio, language: " ".join(s["text"] for s in engine.transcribe(audio, language)["segments"]), {"adapter": "lab.speech.SpeechEngine", "device": "cpu", "compute_type": "int8", "dir": spec}
    from faster_whisper import WhisperModel
    directory, runtime = spec.rsplit("@", 1)
    device, compute_type = runtime.split("/", 1)
    model = WhisperModel(directory, device=device, compute_type=compute_type, local_files_only=True)
    def run(audio, language):
        segments, _ = model.transcribe(str(audio), language=language, vad_filter=True, beam_size=5, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments)
    return run, {"adapter": "faster_whisper.WhisperModel", "device": device, "compute_type": compute_type, "dir": directory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=SCRIPT, help="Recording script with the reference statements")
    parser.add_argument("--recordings", type=Path, required=True, help="CSV: recording_id,speaker_id,language,statement_id,file,condition")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="name=<dir> or name=<dir>@<device>/<compute_type>")
    parser.add_argument("--review-times", type=Path, help="CSV: reviewer_id,recording_id,model,correction_seconds")
    parser.add_argument("--report", type=Path, required=True, help="Private: contains hypotheses")
    parser.add_argument("--public-summary", type=Path, required=True, help="Publishable: metrics only")
    args = parser.parse_args()
    statements = {s["id"]: s for s in json.loads(args.script.read_text(encoding="utf-8"))["statements"]}
    recordings = list(csv.DictReader(args.recordings.open(encoding="utf-8")))
    for r in recordings:
        if statements[r["statement_id"]]["language"] != r["language"]:
            raise ValueError("Recording language does not match its statement: " + r["recording_id"])
    observations, models = [], {}
    for item in args.model:
        name, spec = item.split("=", 1)
        run, models[name] = transcriber(name, spec)
        for r in recordings:
            audio = args.audio_dir / r["file"]
            duration = validate_audio(audio)
            started = time.perf_counter()
            hypothesis = run(audio, r["language"])
            seconds = time.perf_counter() - started
            metrics = transcript_metrics(statements[r["statement_id"]]["text"], hypothesis)
            observations.append({"model": name, **{k: r[k] for k in ("recording_id", "speaker_id", "language", "statement_id", "condition")},
                                 "audio_sha256": sha(audio), "audio_seconds": duration, "processing_seconds": round(seconds, 3),
                                 "hypothesis": hypothesis, **metrics})
            print(name, r["recording_id"], metrics["wer"], flush=True)
    review = defaultdict(float)
    if args.review_times:
        for row in csv.DictReader(args.review_times.open(encoding="utf-8")):
            review[(row["model"], row["recording_id"])] += float(row["correction_seconds"])
    summary = {}
    for name in models:
        for language in ("fr", "nl"):
            rows = [o for o in observations if o["model"] == name and o["language"] == language]
            if not rows:
                continue
            words = sum(o["reference_words"] for o in rows)
            audio = sum(o["audio_seconds"] for o in rows)
            reviewed = [o for o in rows if (name, o["recording_id"]) in review]
            summary[f"{name}/{language}"] = {
                "recordings": len(rows), "speakers": len({o["speaker_id"] for o in rows}), "reference_words": words,
                "wer": round(sum(o["errors"] for o in rows) / words, 4),
                "real_time_factor": round(sum(o["processing_seconds"] for o in rows) / audio, 4),
                "review_minutes_per_audio_minute": round(sum(review[(name, o["recording_id"])] for o in reviewed) / sum(o["audio_seconds"] for o in reviewed), 4) if reviewed else None,
                "reviewed_recordings": len(reviewed)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"models": models, "summary": summary, "observations": observations}, ensure_ascii=False, indent=2), encoding="utf-8")
    public = {"script_sha256": sha(args.script), "models": {k: {x: v[x] for x in ("adapter", "device", "compute_type")} for k, v in models.items()},
              "aggregation": "Errors summed over recordings divided by reference words, per language and model.",
              "summary": summary, "observations": [{k: o[k] for k in o if k != "hypothesis"} for o in observations]}
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
