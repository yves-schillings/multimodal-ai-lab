"""Local speech recognition for synthetic statement-recording demonstrations.

Importing this module neither downloads a model nor opens an audio device. Model
preparation is a separate, explicit operation. A transcript is always a draft
requiring human validation; probabilities describe recognition uncertainty only.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = frozenset({"en", "fr", "nl"})
DEFAULT_MAX_AUDIO_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_DURATION_SECONDS = 300.0


class AudioValidationError(ValueError):
    """The submitted file is not an acceptable, bounded audio recording."""


class UnsupportedLanguageError(ValueError):
    """The requested or detected language is outside the demonstration scope."""


class SpeechDependencyError(RuntimeError):
    """A local runtime dependency is missing."""


class ModelUnavailableError(RuntimeError):
    """The selected model has not been prepared locally."""


def _language_code(language: str | None) -> str | None:
    if language is None:
        return None
    if not isinstance(language, str):
        raise UnsupportedLanguageError("Choose English (en), French (fr), Dutch (nl), or auto.")
    value = language.strip().lower()
    if value == "auto":
        return None
    if value not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError("Choose English (en), French (fr), Dutch (nl), or auto.")
    return value


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return default
    return result if math.isfinite(result) else default


def validate_audio(
    path: Path,
    *,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> float:
    """Decode audio to measure its actual duration, without trusting file suffixes.

    Decoding is bounded by both file size and elapsed audio duration. Counting
    decoded samples also handles formats whose container duration is absent or
    inaccurate. No audio, filename, or recognised text is written to logs.
    """
    path = Path(path)
    try:
        if not path.is_file():
            raise AudioValidationError("Select an existing audio file.")
        size = path.stat().st_size
    except OSError:
        raise AudioValidationError("The audio file cannot be read.") from None
    if size == 0:
        raise AudioValidationError("The audio file is empty.")
    if size > max_audio_bytes:
        raise AudioValidationError(f"Audio must not exceed {max_audio_bytes // (1024 * 1024)} MiB.")
    try:
        import av
    except ImportError:
        raise SpeechDependencyError("Install the local speech dependencies before transcribing audio.") from None

    duration = 0.0
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.audio:
                raise AudioValidationError("The file contains no audio stream.")
            for frame in container.decode(audio=0):
                rate = _finite_number(getattr(frame, "sample_rate", 0))
                samples = _finite_number(getattr(frame, "samples", 0))
                if rate <= 0 or samples < 0:
                    raise AudioValidationError("The audio stream has invalid timing metadata.")
                duration += samples / rate
                if duration > max_duration_seconds + 0.001:
                    raise AudioValidationError(f"Audio must not exceed {max_duration_seconds:g} seconds.")
    except AudioValidationError:
        raise
    except Exception:
        raise AudioValidationError("The file could not be decoded as audio.") from None
    if duration <= 0:
        raise AudioValidationError("The audio file contains no decodable samples.")
    return round(duration, 3)


class SpeechEngine:
    """CPU/int8 faster-whisper adapter with explicit local model availability."""

    def __init__(
        self,
        model_name: str = "base",
        model_dir: Path | None = None,
        *,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
        max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    ) -> None:
        self.model_name = os.environ.get("LAB_WHISPER_MODEL", model_name) if model_name == "base" else model_name
        configured_dir = model_dir or os.environ.get("LAB_WHISPER_MODEL_DIR")
        self.model_dir = Path(configured_dir) if configured_dir else None
        if not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0 or max_audio_bytes <= 0:
            raise ValueError("Audio duration and file-size limits must be positive.")
        self.max_duration_seconds = float(max_duration_seconds)
        self.max_audio_bytes = int(max_audio_bytes)
        self._model: Any = None
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.model_dir is not None and not self.model_dir.is_dir():
            raise ModelUnavailableError("Prepare the speech model locally before starting a transcription.")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SpeechDependencyError("Install the local speech dependencies before transcribing audio.") from None
        target = str(self.model_dir) if self.model_dir is not None else self.model_name
        try:
            self._model = WhisperModel(
                target,
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
        except Exception:
            raise ModelUnavailableError("The speech model is unavailable locally. Run explicit model preparation first.") from None
        return self._model

    def transcribe(self, path: Path, language: str | None = None) -> dict[str, Any]:
        selected_language = _language_code(language)
        duration = validate_audio(
            path,
            max_duration_seconds=self.max_duration_seconds,
            max_audio_bytes=self.max_audio_bytes,
        )
        started = time.perf_counter()
        # A single engine serialises local CPU inference and initial model loading.
        with self._lock:
            model = self._load_model()
            segments, info = model.transcribe(
                str(path),
                language=selected_language,
                vad_filter=True,
                word_timestamps=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
            detected_language = str(getattr(info, "language", selected_language or "")).lower()
            if detected_language not in SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError("The detected language is outside English, French, and Dutch. Select the recording language explicitly.")
            result_segments = []
            for index, segment in enumerate(segments):
                words = []
                for word in getattr(segment, "words", None) or []:
                    probability = min(1.0, max(0.0, _finite_number(getattr(word, "probability", 0))))
                    words.append({
                        "start": round(max(0.0, _finite_number(getattr(word, "start", 0))), 3),
                        "end": round(max(0.0, _finite_number(getattr(word, "end", 0))), 3),
                        "text": str(getattr(word, "word", "")).strip(),
                        "probability": round(probability, 4),
                    })
                review_reasons = []
                if any(word["probability"] < 0.60 for word in words):
                    review_reasons.append("low_word_probability")
                if _finite_number(getattr(segment, "avg_logprob", 0)) < -1.0:
                    review_reasons.append("low_segment_probability")
                if _finite_number(getattr(segment, "no_speech_prob", 0)) > 0.60:
                    review_reasons.append("possible_silence")
                if not words:
                    review_reasons.append("word_probabilities_unavailable")
                result_segments.append({
                    "id": index,
                    "start": round(max(0.0, _finite_number(getattr(segment, "start", 0))), 3),
                    "end": round(max(0.0, _finite_number(getattr(segment, "end", 0))), 3),
                    "text": str(getattr(segment, "text", "")).strip(),
                    "review_required": bool(review_reasons),
                    "review_reasons": review_reasons,
                    "words": words,
                })
        return {
            "provider": "faster-whisper",
            "model": self.model_name,
            "language": detected_language,
            "duration_seconds": duration,
            "processing_seconds": round(time.perf_counter() - started, 3),
            "requires_human_validation": True,
            "segments": result_segments,
        }
