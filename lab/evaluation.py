"""Word-error evaluation and optional local MLflow metrics, without transcripts."""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping


def normalize_words(text: str) -> list[str]:
    """Ignore case and punctuation; retain accents, numbers, and apostrophes.

    This deliberately does not rewrite names, dates, or numbers to conceal
    transcription errors. WER alone does not establish a statement's accuracy.
    """
    if not isinstance(text, str):
        raise TypeError("Reference and hypothesis must be strings.")
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized, flags=re.UNICODE)


def transcript_metrics(reference: str, hypothesis: str) -> dict[str, int | float | None]:
    """Return Levenshtein edit counts and WER; never include input text."""
    expected, actual = normalize_words(reference), normalize_words(hypothesis)
    if len(expected) > 5000 or len(actual) > 5000:
        raise ValueError("Evaluation is limited to 5,000 words per transcript.")
    # Entries are (distance, substitutions, deletions, insertions). Keeping one
    # row bounds memory while preserving the counts of a minimal edit path.
    previous = [(j, 0, 0, j) for j in range(len(actual) + 1)]
    for i, reference_word in enumerate(expected, 1):
        current = [(i, 0, i, 0)]
        for j, hypothesis_word in enumerate(actual, 1):
            if reference_word == hypothesis_word:
                current.append(previous[j - 1])
                continue
            diagonal, above, left = previous[j - 1], previous[j], current[j - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2] + 1, above[3]),
                (left[0] + 1, left[1], left[2], left[3] + 1),
            )
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    wer = errors / len(expected) if expected else (0.0 if not actual else None)
    return {
        "reference_words": len(expected),
        "hypothesis_words": len(actual),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "wer": wer,
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return WER as a ratio, which can exceed 1.0 when insertions are numerous."""
    result = transcript_metrics(reference, hypothesis)
    if result["wer"] is None:
        raise ValueError("WER is undefined for an empty reference with a nonempty hypothesis.")
    return float(result["wer"])


def evaluate_transcripts(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate error counts by reference-word weight, including per-language WER.

    Each case requires reference/hypothesis and may provide language (en/fr/nl).
    Neither original text nor case identifiers are returned or sent to MLflow.
    """
    totals = {key: 0 for key in ("sample_count", "reference_words", "hypothesis_words", "substitutions", "deletions", "insertions", "errors")}
    languages: dict[str, dict[str, int | float | None]] = {}
    for case in cases:
        language = str(case.get("language", "unspecified")).lower()
        if language not in {"en", "fr", "nl", "unspecified"}:
            raise ValueError("Evaluation language must be en, fr, nl, or unspecified.")
        metrics = transcript_metrics(case["reference"], case["hypothesis"])
        language_totals = languages.setdefault(language, dict.fromkeys(totals, 0))
        totals["sample_count"] += 1
        language_totals["sample_count"] += 1
        for key in totals:
            if key == "sample_count":
                continue
            totals[key] += int(metrics[key])
            language_totals[key] += int(metrics[key])
    if not totals["sample_count"]:
        raise ValueError("Provide at least one evaluation sample.")
    for counts in [totals, *languages.values()]:
        counts["wer"] = counts["errors"] / counts["reference_words"] if counts["reference_words"] else (0.0 if not counts["hypothesis_words"] else None)
    return {**totals, "by_language": languages}


def record_local_mlflow_evaluation(
    metrics: Mapping[str, Any],
    tracking_directory: Path,
    *,
    experiment_name: str = "synthetic-statement-speech-evaluation",
    run_name: str = "local-speech-evaluation",
    model_name: str = "base",
) -> str:
    """Write numeric metrics to a local SQLite MLflow store, never remote tracking.

    Only known numeric metric names are accepted. The caller controls the local
    directory; no transcript, audio, local filename, or arbitrary metadata is
    logged. MLflow is imported only when explicitly invoking this function.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise RuntimeError("Install MLflow to record local evaluation metrics.") from None
    directory = Path(tracking_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifact_directory = directory / "artifacts"
    artifact_directory.mkdir(exist_ok=True)
    uri = "sqlite:///" + (directory / "mlflow.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name, artifact_location=artifact_directory.as_uri())
    run = client.create_run(experiment_id, tags={"mlflow.runName": run_name, "data_scope": "synthetic_only", "task": "speech_recognition"})
    run_id = run.info.run_id
    allowed = {"sample_count", "reference_words", "hypothesis_words", "substitutions", "deletions", "insertions", "errors", "wer"}
    try:
        # A model identifier is configuration, not a transcript or audio filename.
        client.log_param(run_id, "model", model_name)
        for key in sorted(allowed):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                client.log_metric(run_id, key, float(value))
        for language, values in metrics.get("by_language", {}).items():
            if language not in {"en", "fr", "nl", "unspecified"}:
                continue
            for key in sorted(allowed):
                value = values.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    client.log_metric(run_id, f"{language}_{key}", float(value))
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    return run_id
