"""Tests of the paired, case-clustered analysis of RAG strategies."""
import pytest

from lab.rag_paired_analysis import (by_case, case_bootstrap, compare, holm, leave_one_case_out,
                                     paired_differences, question_scores, sign_flip_check, strict_success)


def trial(question_id, expect="answer", correct=True, support=True, link=None, abstained=False, stop="answered",
          repeat=1, variant="fixed"):
    """Build a minimal run-record trial with the verdict fields the analysis reads.

    Args:
        question_id (str): Question identifier, ``<case>-<question>``.
        expect (str): Expected outcome of the evaluation set.
        correct (bool): The frozen judge's ``correct``.
        support (bool): The judge's ``support`` for answered trials.
        link (bool | None): The judge's ``entity_link_ok``; None leaves the field out.
        abstained (bool): Whether the strategy abstained.
        stop (str): Stop reason of the strategy.
        repeat (int): Repeat number.
        variant (str): Strategy name.
    Returns:
        dict: The trial.
    """
    verdict = {"kind": "x", "correct": correct, "abstained": abstained, "answered": not abstained, "review": []}
    if not abstained:
        verdict["support"] = support
        if link is not None:
            verdict["entity_link_ok"] = link
    return {"question_id": question_id, "expect": expect, "verdict": verdict, "stop_reason": stop,
            "repeat": repeat, "variant": variant}


def test_strict_success_needs_support_and_the_right_object():
    """A correct answer counts only when its quote supports it and it cites no other item's identifier."""
    assert strict_success(trial("C06-Q1"))
    assert not strict_success(trial("C06-Q1", support=False))
    assert not strict_success(trial("C06-Q1", link=False))
    assert strict_success(trial("C06-Q1", link=True))


def test_withheld_or_late_abstention_is_never_a_success():
    """A withheld answer or a deadline is a failure even where abstention was expected."""
    assert strict_success(trial("C06-Q2", expect="abstain", abstained=True, stop="model_abstained"))
    assert not strict_success(trial("C06-Q2", expect="abstain", abstained=True, stop="model_unavailable"))
    assert not strict_success(trial("C06-Q2", expect="abstain", abstained=True, stop="deadline"))
    assert not strict_success(trial("C06-Q3", abstained=True, correct=False, stop="model_abstained"))


def test_access_probes_are_never_scored():
    """Access probes stay out of every quality score."""
    with pytest.raises(ValueError):
        strict_success(trial("C06-Q9", expect="refused"))
    assert question_scores([trial("C06-Q9", expect="refused")]) == {}


def test_question_score_is_the_mean_of_repeats():
    """Three repeats with two successes give a question score of two thirds."""
    trials = [trial("C06-Q1", repeat=1), trial("C06-Q1", repeat=2), trial("C06-Q1", repeat=3, correct=False)]
    assert question_scores(trials) == {"C06-Q1": pytest.approx(2 / 3)}


def test_paired_differences_need_the_same_questions():
    """A paired comparison refuses strategies that answered different questions."""
    with pytest.raises(ValueError):
        paired_differences({"C06-Q1": 1.0}, {"C06-Q2": 1.0})


def test_question_weighting_and_leave_one_case_out():
    """Each question weighs the same, and removing a case recomputes the mean without it."""
    groups = by_case({"C06-Q1": 1.0, "C06-Q2": 1.0, "C07-Q1": 0.0, "C07-Q2": 0.0, "C07-Q3": -1.0})
    assert leave_one_case_out(groups) == {"C06": pytest.approx(-0.333, abs=1e-3), "C07": 1.0}


def test_sign_flip_check_with_no_difference_and_with_a_regular_gain():
    """No difference gives 1; the same gain in all ten cases gives 2 extreme assignments of 1,024."""
    assert sign_flip_check({f"C{i:02d}": [0.0] for i in range(10)}) == 1.0
    assert sign_flip_check({f"C{i:02d}": [1.0] for i in range(10)}) == round(2 / 1024, 4)


def test_case_bootstrap_is_reproducible_and_brackets_a_constant_gain():
    """The fixed seed gives the same interval twice, and a constant gain gives a zero-width interval."""
    groups = {f"C{i:02d}": [0.5, 0.5] for i in range(10)}
    assert case_bootstrap(groups, resamples=200) == (0.5, 0.5)
    mixed = {f"C{i:02d}": [float(i % 3 - 1)] for i in range(10)}
    assert case_bootstrap(mixed, resamples=200) == case_bootstrap(mixed, resamples=200)


def test_holm_is_monotone_and_capped():
    """Holm multiplies the smallest value by the family size and never decreases or exceeds 1."""
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.08, "c": 0.5}
    assert holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_compare_reports_the_difference_and_its_cases():
    """The comparison gives both success rates, the difference and the questions where each strategy won."""
    first = [trial("C06-Q1"), trial("C07-Q1", correct=False)]
    second = [trial("C06-Q1", correct=False), trial("C07-Q1", correct=False)]
    result = compare(first, second, resamples=100)
    assert result["difference"] == 0.5
    assert result["questions_better_first"] == ["C06-Q1"]
    assert result["per_case"]["C07"] == {"questions": 1, "difference": 0.0}


def test_review_sheet_covers_every_queued_repeat_once_per_distinct_output():
    """Every queued trial of every repeat is covered, and identical repeats share one sheet item."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_rag_strategies.py"
    spec = importlib.util.spec_from_file_location("compare_rag_strategies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trials = []
    for repeat in (1, 2, 3):
        flagged = trial("C06-Q1", correct=False, repeat=repeat)
        flagged["verdict"]["review"] = ["answer given where abstention was expected"]
        flagged.update(question="q", answer="same" if repeat < 3 else "different")
        plain = trial("C06-Q2", repeat=repeat)
        plain.update(question="q", answer="ok")
        trials += [flagged, plain]
    sheet, key = module.review_sample({"fixed": trials}, per_side=0)
    assert len(sheet) == 2
    assert sorted(r for item in key.values() for r in item["queued_repeats_covered"]) == [1, 2, 3]
