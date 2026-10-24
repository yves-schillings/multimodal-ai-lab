"""Integrity of the fictional measurement fixtures and of the metric helpers used by M1-M4."""
import json
import re
import unittest
from pathlib import Path

from scripts.evaluate_retrieval_relevance import recall, reciprocal_rank
from scripts.measure_ocr_quality import error_rates
from scripts.measurement_common import summary

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "measurements"


class CorpusIntegrity(unittest.TestCase):
    corpus = json.loads((FIXTURES / "rag_corpus.json").read_text(encoding="utf-8"))

    def test_counts_match_the_protocol(self):
        passages, questions = self.corpus["passages"], self.corpus["questions"]
        self.assertEqual(len(passages), 60)
        self.assertEqual(sum("injection" in p for p in passages), 20)
        self.assertEqual(len(questions), 50)
        self.assertEqual({q["language"] for q in questions}, {"en", "fr", "nl"})
        self.assertEqual(sum(bool(q.get("m1")) for q in questions), 20)
        self.assertEqual(len(self.corpus["unsupported_questions"]), 20)

    def test_every_instruction_passage_has_one_answerable_question(self):
        injected = {p["id"] for p in self.corpus["passages"] if "injection" in p}
        targets = [q["relevant"][0] for q in self.corpus["questions"] if q.get("m1")]
        self.assertEqual(sorted(targets), sorted(injected))

    def test_relevant_passages_exist_and_fit_one_chunk(self):
        ids = {p["id"] for p in self.corpus["passages"]}
        self.assertTrue(all(r in ids for q in self.corpus["questions"] for r in q["relevant"]))
        self.assertTrue(all(len(p["text"]) <= 1500 for p in self.corpus["passages"]))

    def test_assembled_codes_never_appear_in_the_sources(self):
        # A detected code therefore proves that the model followed the hidden instruction.
        texts = "\n".join(p["text"] for p in self.corpus["passages"])
        for injection in self.corpus["injections"]:
            if injection["type"] == "canary_output":
                self.assertFalse(any(re.search(p, texts, re.I) for p in injection["detect"]), injection["id"])


class Metrics(unittest.TestCase):
    def test_summary_uses_nearest_rank_percentile(self):
        result = summary(list(range(1, 31)))
        self.assertEqual((result["median"], result["p95"], result["n"]), (15.5, 29, 30))

    def test_recall_and_reciprocal_rank(self):
        ranked = ["P2", "P1", "P3"]
        self.assertEqual(recall(ranked, ["P1"], 1), 0)
        self.assertEqual(recall(ranked, ["P1", "P9"], 3), 0.5)
        self.assertEqual(reciprocal_rank(ranked, ["P1"], 10), 0.5)
        self.assertEqual(reciprocal_rank(ranked, ["P9"], 10), 0.0)

    def test_character_error_rate_ignores_layout_whitespace_only(self):
        self.assertEqual(error_rates("Item  01\nSeal", "Item 01 Seal")["cer"], 0.0)
        self.assertEqual(error_rates("abcd", "abed")["cer"], 0.25)


if __name__ == "__main__":
    unittest.main()
