"""Offline tests: no microphone, model download, network, or paid service."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lab.evaluation import evaluate_transcripts, normalize_words, record_local_mlflow_evaluation, transcript_metrics, word_error_rate
from lab.speech import AudioValidationError, ModelUnavailableError, SpeechEngine, UnsupportedLanguageError, validate_audio


def audio_module(*, duration: float = 2.0, audio_present: bool = True):
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.audio = [object()] if audio_present else []
    container.decode.return_value = [SimpleNamespace(samples=int(duration * 16000), sample_rate=16000)]
    return SimpleNamespace(open=MagicMock(return_value=container)), container


def backend_module(*, probability: float = 0.9, language: str = "fr"):
    segment = SimpleNamespace(
        id=99,
        start=0.1,
        end=1.5,
        text=" Bonjour monsieur. ",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        words=[SimpleNamespace(start=0.1, end=0.7, word=" Bonjour", probability=probability)],
    )
    model = MagicMock()
    model.transcribe.return_value = (iter([segment]), SimpleNamespace(language=language))
    module = SimpleNamespace(WhisperModel=MagicMock(return_value=model))
    return module, model, segment


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.wav"
        self.path.write_bytes(b"test fixture; decoded by a fake PyAV backend")
        self.env = patch.dict(os.environ, {"LAB_WHISPER_MODEL": "base", "LAB_WHISPER_MODEL_DIR": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_initialization_does_not_load_a_model(self):
        backend, _, _ = backend_module()
        with patch.dict("sys.modules", {"faster_whisper": backend}):
            engine = SpeechEngine()
        backend.WhisperModel.assert_not_called()
        self.assertIsNone(engine._model)

    def test_offline_model_settings_and_timestamped_output(self):
        av, _ = audio_module()
        backend, model, _ = backend_module()
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            result = SpeechEngine().transcribe(self.path, "fr")
        backend.WhisperModel.assert_called_once_with("base", device="cpu", compute_type="int8", local_files_only=True)
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertTrue(model.transcribe.call_args.kwargs["word_timestamps"])
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["duration_seconds"], 2.0)
        self.assertTrue(result["requires_human_validation"])
        self.assertEqual(result["segments"][0]["text"], "Bonjour monsieur.")
        self.assertEqual(result["segments"][0]["id"], 0)
        self.assertFalse(result["segments"][0]["review_required"])

    def test_uncertain_words_are_flagged_for_review(self):
        av, _ = audio_module()
        backend, _, _ = backend_module(probability=0.42)
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            result = SpeechEngine().transcribe(self.path, "fr")
        self.assertTrue(result["segments"][0]["review_required"])
        self.assertEqual(result["segments"][0]["review_reasons"], ["low_word_probability"])
        self.assertNotIn("speaker", result["segments"][0])
        self.assertNotIn("credibility", result["segments"][0])

    def test_missing_word_probabilities_require_review(self):
        av, _ = audio_module()
        backend, _, segment = backend_module()
        segment.words = None
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            result = SpeechEngine().transcribe(self.path)
        self.assertTrue(result["segments"][0]["review_required"])

    def test_auto_language_is_sent_as_none(self):
        av, _ = audio_module()
        backend, model, _ = backend_module(language="nl")
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            result = SpeechEngine().transcribe(self.path, "auto")
        self.assertIsNone(model.transcribe.call_args.kwargs["language"])
        self.assertEqual(result["language"], "nl")

    def test_unsupported_requested_language_fails_before_model_loading(self):
        with patch("lab.speech.validate_audio") as validator:
            with self.assertRaises(UnsupportedLanguageError):
                SpeechEngine().transcribe(self.path, "de")
        validator.assert_not_called()

    def test_unsupported_detected_language_is_not_silently_accepted(self):
        av, _ = audio_module()
        backend, _, _ = backend_module(language="de")
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            with self.assertRaises(UnsupportedLanguageError):
                SpeechEngine().transcribe(self.path, "auto")

    def test_duration_is_decoded_not_trusted_from_container_metadata(self):
        av, container = audio_module(duration=5)
        container.duration = 1
        with patch.dict("sys.modules", {"av": av}):
            self.assertEqual(validate_audio(self.path), 5.0)

    def test_excess_duration_stops_before_model_is_loaded(self):
        av, _ = audio_module(duration=301)
        backend, _, _ = backend_module()
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            with self.assertRaisesRegex(AudioValidationError, "300 seconds"):
                SpeechEngine().transcribe(self.path, "en")
        backend.WhisperModel.assert_not_called()

    def test_excess_size_is_rejected_without_decoder(self):
        av, _ = audio_module()
        with patch.dict("sys.modules", {"av": av}):
            with self.assertRaises(AudioValidationError):
                validate_audio(self.path, max_audio_bytes=1)
        av.open.assert_not_called()

    def test_empty_missing_and_non_audio_files_are_rejected(self):
        self.path.write_bytes(b"")
        with self.assertRaises(AudioValidationError):
            validate_audio(self.path)
        with self.assertRaises(AudioValidationError):
            validate_audio(self.path.parent / "missing.wav")
        self.path.write_bytes(b"fake image")
        av, _ = audio_module(audio_present=False)
        with patch.dict("sys.modules", {"av": av}):
            with self.assertRaisesRegex(AudioValidationError, "no audio"):
                validate_audio(self.path)

    def test_decode_errors_are_sanitized(self):
        av, _ = audio_module()
        av.open.side_effect = RuntimeError("private recording filename or decoder details")
        with patch.dict("sys.modules", {"av": av}):
            with self.assertRaises(AudioValidationError) as error:
                validate_audio(self.path)
        self.assertNotIn("private", str(error.exception))

    def test_missing_model_gives_explicit_preparation_instruction(self):
        av, _ = audio_module()
        backend, _, _ = backend_module()
        backend.WhisperModel.side_effect = RuntimeError("cache miss")
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            with self.assertRaisesRegex(ModelUnavailableError, "preparation"):
                SpeechEngine().transcribe(self.path, "en")
        self.assertTrue(backend.WhisperModel.call_args.kwargs["local_files_only"])

    def test_explicit_model_directory_is_used(self):
        av, _ = audio_module()
        backend, _, _ = backend_module()
        with patch.dict("sys.modules", {"av": av, "faster_whisper": backend}):
            SpeechEngine(model_dir=self.path.parent).transcribe(self.path, "fr")
        self.assertEqual(backend.WhisperModel.call_args.args[0], str(self.path.parent))

    def test_invalid_limits_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SpeechEngine(max_duration_seconds=value)


class EvaluationTests(unittest.TestCase):
    def test_normalization_handles_unicode_without_removing_accents(self):
        self.assertEqual(normalize_words("L’AUDITION, à 18:40 !"), ["l'audition", "à", "18", "40"])
        self.assertEqual(word_error_rate("Hello, WORLD!", "hello world"), 0.0)
        self.assertEqual(word_error_rate("cote", "côte"), 1.0)

    def test_counts_substitution_deletion_and_insertion(self):
        self.assertEqual(transcript_metrics("the blue car", "the red car")["substitutions"], 1)
        self.assertEqual(transcript_metrics("the blue car", "the car")["deletions"], 1)
        self.assertEqual(transcript_metrics("the car", "the blue car")["insertions"], 1)
        self.assertAlmostEqual(word_error_rate("the blue car", "the red car"), 1 / 3)

    def test_wer_can_exceed_one(self):
        self.assertEqual(word_error_rate("car", "the big red car"), 3.0)

    def test_empty_reference_is_handled_explicitly(self):
        self.assertEqual(word_error_rate("", ""), 0)
        self.assertIsNone(transcript_metrics("", "hallucination")["wer"])
        with self.assertRaises(ValueError):
            word_error_rate("", "hallucination")

    def test_aggregate_is_word_weighted_not_average_of_ratios(self):
        result = evaluate_transcripts([
            {"reference": "one two three four", "hypothesis": "one two three four", "language": "en"},
            {"reference": "bonjour", "hypothesis": "bonsoir", "language": "fr"},
        ])
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["reference_words"], 5)
        self.assertAlmostEqual(result["wer"], 0.2)
        self.assertEqual(result["by_language"]["fr"]["wer"], 1)
        self.assertNotIn("bonjour", str(result))

    def test_empty_evaluation_and_unsupported_language_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_transcripts([])
        with self.assertRaises(ValueError):
            evaluate_transcripts([{"reference": "a", "hypothesis": "a", "language": "de"}])

    def test_mlflow_logs_only_allowlisted_metrics_to_local_store(self):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client.create_experiment.return_value = "1"
        client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="local-run"))
        factory = MagicMock(return_value=client)
        tracking_module = SimpleNamespace(MlflowClient=factory)
        metrics = {
            "wer": 0.125,
            "sample_count": 2,
            "raw_transcript": "PRIVATE CONTENT MUST NOT BE LOGGED",
            "by_language": {"fr": {"wer": 0.25, "text": "PRIVATE CONTENT MUST NOT BE LOGGED"}},
        }
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict("sys.modules", {"mlflow": SimpleNamespace(), "mlflow.tracking": tracking_module}):
                with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://must-not-be-used.example"}):
                    result = record_local_mlflow_evaluation(metrics, Path(temp))
        self.assertEqual(result, "local-run")
        self.assertTrue(factory.call_args.kwargs["tracking_uri"].startswith("sqlite:///"))
        self.assertNotIn("https://", factory.call_args.kwargs["tracking_uri"])
        self.assertNotIn("PRIVATE CONTENT", str(client.mock_calls))
        self.assertEqual({call.args[1] for call in client.log_metric.call_args_list}, {"wer", "sample_count", "fr_wer"})
        client.set_terminated.assert_called_once_with("local-run", status="FINISHED")


if __name__ == "__main__":
    unittest.main()
