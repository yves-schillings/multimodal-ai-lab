"""Offline workflow tests using synthetic data and isolated temporary databases."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from lab.retrieval import answer_question
from lab.store import ConflictError, StatementStore


class StatementWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = StatementStore(Path(self.tmp.name))

    def review_all(self, actor="officer-a", case_id="demo-case-a"):
        case = self.store.get_case(actor, case_id)
        for segment in list(case["segments"]):
            case = self.store.update_segment(actor, case_id, segment["id"], segment["text"], case["revision"])
        return case

    def test_actor_access_isolation_on_every_case_operation(self):
        self.assertEqual([case["id"] for case in self.store.list_cases("officer-a")], ["demo-case-a"])
        self.assertEqual([case["id"] for case in self.store.list_cases("officer-b")], ["demo-case-b"])
        self.assertEqual(len(self.store.list_cases("reviewer")), 2)
        case_b = self.store.get_case("officer-b", "demo-case-b")
        operations = [
            lambda: self.store.get_case("officer-a", "demo-case-b"),
            lambda: self.store.save_transcript("officer-a", "demo-case-b", {"segments": []}),
            lambda: self.store.update_segment("officer-a", "demo-case-b", case_b["segments"][0]["id"], "Unauthorized edit", case_b["revision"]),
            lambda: self.store.make_draft("officer-a", "demo-case-b"),
            lambda: self.store.approve("officer-a", "demo-case-b", case_b["revision"]),
            lambda: answer_question(self.store, "officer-a", "demo-case-b", "What happened?"),
        ]
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(PermissionError):
                operation()
        with self.assertRaises(PermissionError):
            self.store.list_cases("unrecognized-user")

    def test_retrieval_excludes_other_case_and_unreviewed_transcript(self):
        self.review_all("officer-b", "demo-case-b")
        result = answer_question(self.store, "reviewer", "demo-case-a", "What happened at Cedar Lane workshop window?")
        self.assertTrue(result["abstained"])
        self.assertNotIn("Cedar Lane", result["answer"])
        self.assertTrue(answer_question(self.store, "officer-a", "demo-case-a", "Where was the bicycle?")["abstained"])
        self.review_all()
        result = answer_question(self.store, "officer-a", "demo-case-a", "Where was the bicycle left?")
        self.assertFalse(result["abstained"])
        self.assertIn("library", result["answer"])
        self.assertEqual(result["mode"], "extractive")
        self.assertTrue(all(source["case_id"] in (None, "demo-case-a") for source in result["sources"]))
        for source in result["sources"]:
            self.assertIn(source["text"], result["answer"])

    def test_draft_is_exact_reviewed_text_with_timestamp_sources(self):
        case = self.review_all()
        draft_case = self.store.make_draft("officer-a", case["id"])
        draft = draft_case["draft"]
        self.assertEqual(draft["mode"], "extractive")
        self.assertEqual(draft["revision"], draft_case["revision"])
        self.assertEqual(len(draft["sources"]), len(case["segments"]))
        for segment, source in zip(case["segments"], draft["sources"]):
            self.assertEqual(segment["text"], source["text"])
            self.assertEqual(segment["id"], source["id"])
            self.assertIn(segment["text"], draft["text"])
        self.assertIn("[00:00.000–00:05.400]", draft["text"])

    def test_revision_conflict_does_not_overwrite_or_append_audit(self):
        before = self.store.get_case("officer-a", "demo-case-a")
        segment = before["segments"][0]
        saved = self.store.update_segment("officer-a", before["id"], segment["id"], "First reviewed correction.", before["revision"])
        with self.assertRaises(ConflictError):
            self.store.update_segment("officer-a", before["id"], segment["id"], "Stale overwrite.", before["revision"])
        after = self.store.get_case("officer-a", before["id"])
        self.assertEqual(after["segments"][0]["text"], "First reviewed correction.")
        self.assertEqual(after["revision"], saved["revision"])
        self.assertEqual(len(after["audit"]), len(saved["audit"]))

    def test_concurrent_writers_only_one_can_commit_the_same_revision(self):
        before = self.store.get_case("officer-a", "demo-case-a")
        def write(text):
            try:
                return self.store.update_segment("officer-a", before["id"], before["segments"][0]["id"], text, before["revision"])
            except ConflictError:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, ["Concurrent correction one.", "Concurrent correction two."]))
        self.assertEqual(sum(result == "conflict" for result in results), 1)
        after = self.store.get_case("officer-a", before["id"])
        self.assertEqual(after["revision"], before["revision"] + 1)
        self.assertEqual(len(after["audit"]), len(before["audit"]) + 1)

    def test_initial_transcript_immutable_and_audit_has_no_text(self):
        before = self.store.get_case("officer-a", "demo-case-a")
        segment = before["segments"][0]
        changed = self.store.update_segment("officer-a", before["id"], segment["id"], "A unique correction never recorded in audit.", before["revision"])
        self.assertEqual(changed["segments"][0]["original_text"], segment["text"])
        replacement = self.store.save_transcript("officer-a", before["id"], {"engine": "test", "segments": [{"start": 0, "end": 2, "text": "Replacement transcript."}]})
        self.assertEqual(len(replacement["transcript_history"]), 2)
        with closing(sqlite3.connect(self.store.path)) as db:
            original = db.execute("SELECT text FROM initial_segments WHERE id=?", (segment["id"],)).fetchone()[0]
        self.assertEqual(original, segment["text"])
        audit = json.dumps(replacement["audit"])
        self.assertNotIn(segment["text"], audit)
        self.assertNotIn("unique correction", audit)
        self.assertNotIn("Replacement transcript", audit)

    def test_approval_requires_review_draft_and_separate_reviewer(self):
        case = self.store.get_case("officer-a", "demo-case-a")
        with self.assertRaises(ValueError):
            self.store.make_draft("officer-a", case["id"])
        with self.assertRaises(ValueError):
            self.store.approve("reviewer", case["id"], case["revision"])
        case = self.review_all()
        with self.assertRaises(ValueError):
            self.store.approve("reviewer", case["id"], case["revision"])
        case = self.store.make_draft("officer-a", case["id"])
        with self.assertRaises(PermissionError):
            self.store.approve("officer-a", case["id"], case["revision"])
        approved = self.store.approve("reviewer", case["id"], case["revision"])
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["approved_revision"], case["draft"]["revision"])
        self.assertEqual(approved["approved_by"], "reviewer")

    def test_reviewer_cannot_approve_their_own_latest_edit(self):
        case = self.review_all("reviewer")
        case = self.store.make_draft("reviewer", case["id"])
        with self.assertRaises(PermissionError):
            self.store.approve("reviewer", case["id"], case["revision"])

    def test_a_different_last_editor_does_not_hide_reviewers_own_edits(self):
        case = self.review_all("reviewer")
        first = case["segments"][0]
        case = self.store.update_segment("officer-a", case["id"], first["id"], first["text"], case["revision"])
        case = self.store.make_draft("officer-a", case["id"])
        with self.assertRaises(PermissionError):
            self.store.approve("reviewer", case["id"], case["revision"])

    def test_edit_after_approval_invalidates_draft_and_approval(self):
        case = self.review_all()
        case = self.store.make_draft("officer-a", case["id"])
        case = self.store.approve("reviewer", case["id"], case["revision"])
        updated = self.store.update_segment("officer-a", case["id"], case["segments"][0]["id"], "A subsequent correction.", case["revision"])
        self.assertEqual(updated["status"], "in_review")
        self.assertIsNone(updated["approved_revision"])
        self.assertIsNone(updated["approved_by"])
        self.assertIsNone(updated["draft"])
        self.assertTrue(updated["audit"][-1]["metadata"]["invalidated_approval"])

    def test_replacement_transcript_invalidates_approval_and_requires_new_review(self):
        case = self.review_all()
        case = self.store.make_draft("officer-a", case["id"])
        case = self.store.approve("reviewer", case["id"], case["revision"])
        updated = self.store.save_transcript("officer-a", case["id"], {"segments": [{"start": 0, "end": 1, "text": "A replacement recording transcript."}]})
        self.assertIsNone(updated["draft"])
        self.assertIsNone(updated["approved_revision"])
        self.assertFalse(updated["segments"][0]["reviewed"])
        with self.assertRaises(ValueError):
            self.store.make_draft("officer-a", case["id"])

    def test_lexical_fallback_returns_only_grounded_source_quotes(self):
        self.review_all()
        with patch("lab.retrieval.TfidfVectorizer", None):
            result = answer_question(self.store, "officer-a", "demo-case-a", "Where was the bicycle left?")
        self.assertFalse(result["abstained"])
        self.assertEqual(result["retrieval_backend"], "lexical_count_cosine")
        self.assertIn("library", result["answer"])
        self.assertTrue(all(source["text"] in result["answer"] for source in result["sources"]))

    def test_unanswerable_question_abstains_with_both_retrieval_backends(self):
        self.review_all()
        for use_fallback in (False, True):
            with self.subTest(fallback=use_fallback):
                if use_fallback:
                    with patch("lab.retrieval.TfidfVectorizer", None):
                        result = answer_question(self.store, "officer-a", "demo-case-a", "What is the capital of Mongolia?")
                    self.assertEqual(result["retrieval_backend"], "lexical_count_cosine")
                else:
                    result = answer_question(self.store, "officer-a", "demo-case-a", "What is the capital of Mongolia?")
                self.assertTrue(result["abstained"])
                self.assertEqual(result["sources"], [])

    def test_create_case_and_reopen_preserve_only_the_expected_cases(self):
        case = self.store.create_case("officer-a", "Synthetic additional statement")
        reopened = StatementStore(Path(self.tmp.name))
        self.assertEqual(len(reopened.list_cases("officer-a")), 2)
        self.assertEqual(len(reopened.list_cases("reviewer")), 3)
        self.assertEqual(reopened.get_case("officer-a", case["id"])["segments"], [])
        with self.assertRaises(ValueError):
            reopened.make_draft("officer-a", case["id"])

    def test_invalid_transcript_does_not_destroy_previous_version(self):
        case = self.store.get_case("officer-a", "demo-case-a")
        for timestamps in ((4, 2), (-1, 2), (0, float("nan"))):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                self.store.save_transcript("officer-a", case["id"], {"segments": [{"start": timestamps[0], "end": timestamps[1], "text": "Invalid segment"}]})
        after = self.store.get_case("officer-a", case["id"])
        self.assertEqual(after["revision"], case["revision"])
        self.assertEqual(after["segments"], case["segments"])


if __name__ == "__main__":
    unittest.main()
