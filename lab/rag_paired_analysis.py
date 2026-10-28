"""Paired, case-clustered comparison of RAG strategies, as fixed by test-analysis-plan.json.

The run record stores one verdict per trial from the frozen judge. This module adds the analysis written
before the held-out results were read:

- strict success: a correct answer that is also supported by a cited quote and never cites another
  item's identifier, or an abstention where abstention is expected;
- one score per question and strategy: the mean of its repeats;
- each question weighs the same;
- a paired difference per question, with a case bootstrap and a case-level sign-flip check, because
  the questions of one case are not independent;
- a per-case table and a leave-one-case-out check;
- Holm adjustment over all pairwise contrasts, which stay exploratory except the primary one.

It reads the run record only. It never changes a verdict or a trial.
"""
from itertools import product
import random

# Stop reasons that mean the output failed a check or the service failed: never a success, even as an abstention.
FAILURE_STOPS = ("model_unavailable", "deadline", "invalid_model_plan")


def case_of(question_id):
    """Return the case identifier of a question identifier such as ``C06-Q3``.

    Args:
        question_id (str): Question identifier written as ``<case>-<question>``.
    Returns:
        str: The case identifier, for example ``C06``.
    """
    return question_id.split("-", 1)[0]


def strict_success(trial):
    """Return True when a trial meets the strict success rule of the analysis plan.

    Args:
        trial (dict): A run-record trial with ``expect`` and the judge's ``verdict``.
    Returns:
        bool: For an expected answer, a correct, supported answer that cites no other item's identifier.
        For an expected abstention, a deliberate abstention. For ``abstain_or_all``, either of the two.
        An error, a deadline, an invalid plan or a withheld answer (``FAILURE_STOPS``) is never a success.
    Raises:
        ValueError: The trial is an access probe, which never enters a quality score.
    """
    expect, verdict = trial["expect"], trial["verdict"]
    if expect == "refused":
        raise ValueError("Access probes are reported apart and never scored for quality.")
    if verdict.get("kind") == "error" or trial.get("stop_reason") in FAILURE_STOPS:
        return False
    if verdict.get("abstained"):
        return expect != "answer" and bool(verdict["correct"])
    if expect == "abstain":
        return False
    return bool(verdict["correct"]) and verdict.get("support") is True and verdict.get("entity_link_ok") is not False


def question_scores(trials, outcome=strict_success):
    """Return, per question, the mean outcome over its repeats.

    Args:
        trials (list[dict]): Trials of one strategy.
        outcome (callable): Function from a trial to a boolean; strict success by default.
    Returns:
        dict[str, float]: Question identifier mapped to the share of its repeats with the outcome.
    """
    votes = {}
    for trial in trials:
        if trial["expect"] != "refused":
            votes.setdefault(trial["question_id"], []).append(1.0 if outcome(trial) else 0.0)
    return {question: sum(values) / len(values) for question, values in votes.items()}


def paired_differences(scores_a, scores_b):
    """Return the per-question differences ``a - b`` on the questions both strategies answered.

    Args:
        scores_a (dict[str, float]): Question scores of the first strategy.
        scores_b (dict[str, float]): Question scores of the second strategy.
    Returns:
        dict[str, float]: Question identifier mapped to the difference.
    Raises:
        ValueError: The two strategies do not cover the same questions.
    """
    if set(scores_a) != set(scores_b):
        raise ValueError("Both strategies must cover the same questions for a paired comparison.")
    return {question: scores_a[question] - scores_b[question] for question in sorted(scores_a)}


def by_case(differences):
    """Group per-question differences by case.

    Args:
        differences (dict[str, float]): Question identifier mapped to a difference.
    Returns:
        dict[str, list[float]]: Case identifier mapped to the differences of its questions.
    """
    groups = {}
    for question, value in differences.items():
        groups.setdefault(case_of(question), []).append(value)
    return dict(sorted(groups.items()))


def question_weighted_mean(groups, cases):
    """Return the mean difference over all questions of the given cases, each question weighing the same.

    Args:
        groups (dict[str, list[float]]): Case identifier mapped to its question differences.
        cases (list[str]): Cases to include; a case listed twice counts twice (bootstrap resample).
    Returns:
        float: The question-weighted mean difference.
    """
    total = sum(sum(groups[case]) for case in cases)
    count = sum(len(groups[case]) for case in cases)
    return total / count


def case_bootstrap(groups, resamples=10_000, seed=20261028):
    """Return the percentile 95 % interval of the question-weighted mean, resampling whole cases.

    Args:
        groups (dict[str, list[float]]): Case identifier mapped to its question differences.
        resamples (int): Number of bootstrap resamples.
        seed (int): Seed of the pseudo-random generator, fixed in the analysis plan.
    Returns:
        tuple[float, float]: Lower and upper bounds (2.5th and 97.5th percentiles), rounded to three decimals.
    """
    generator, cases = random.Random(seed), list(groups)
    estimates = sorted(question_weighted_mean(groups, [generator.choice(cases) for _ in cases])
                       for _ in range(resamples))
    low = estimates[int(0.025 * (resamples - 1))]
    high = estimates[int(round(0.975 * (resamples - 1)))]
    return round(low, 3), round(high, 3)


def sign_flip_check(groups):
    """Return the two-sided share of case-level sign flips at least as extreme as the observed total.

    Every assignment of signs to the per-case sums is enumerated (2 to the number of cases). This is a
    sensitivity check under the assumption that, without a real difference, each case's total is as
    likely to be positive as negative; it is not an exact test of the strategies.

    Args:
        groups (dict[str, list[float]]): Case identifier mapped to its question differences.
    Returns:
        float: Share of assignments whose absolute total reaches the observed absolute total, rounded to four decimals.
    """
    totals = [sum(values) for values in groups.values()]
    observed = abs(sum(totals))
    extreme = sum(1 for signs in product((1, -1), repeat=len(totals))
                  if abs(sum(sign * total for sign, total in zip(signs, totals))) >= observed - 1e-12)
    return round(extreme / 2 ** len(totals), 4)


def leave_one_case_out(groups):
    """Return the question-weighted mean difference with each case removed in turn.

    Args:
        groups (dict[str, list[float]]): Case identifier mapped to its question differences.
    Returns:
        dict[str, float]: Removed case mapped to the estimate without it, rounded to three decimals.
    """
    return {case: round(question_weighted_mean(groups, [c for c in groups if c != case]), 3) for case in groups}


def holm(p_values):
    """Return Holm-adjusted values for a family of contrasts.

    Args:
        p_values (dict[str, float]): Contrast name mapped to its unadjusted value.
    Returns:
        dict[str, float]: Contrast name mapped to its adjusted value, monotone and capped at 1, rounded to four decimals.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = round(running, 4)
    return adjusted


def compare(trials_a, trials_b, outcome=strict_success, resamples=10_000, seed=20261028):
    """Compare two strategies on the same questions with the case-clustered analysis of the plan.

    Args:
        trials_a (list[dict]): Trials of the first strategy.
        trials_b (list[dict]): Trials of the second strategy.
        outcome (callable): Trial outcome; strict success by default.
        resamples (int): Bootstrap resamples.
        seed (int): Bootstrap seed.
    Returns:
        dict: Success rates of both strategies, the difference ``a - b`` with its case-bootstrap interval,
        the sign-flip check, the per-case table, the leave-one-case-out estimates and the number of
        questions and cases.
    Raises:
        ValueError: The strategies do not cover the same questions.
    """
    scores_a, scores_b = question_scores(trials_a, outcome), question_scores(trials_b, outcome)
    differences = paired_differences(scores_a, scores_b)
    groups = by_case(differences)
    every_case = list(groups)
    return {
        "questions": len(differences), "cases": len(groups),
        "success_first": round(sum(scores_a.values()) / len(scores_a), 3),
        "success_second": round(sum(scores_b.values()) / len(scores_b), 3),
        "difference": round(question_weighted_mean(groups, every_case), 3),
        "case_bootstrap_95": case_bootstrap(groups, resamples, seed),
        "sign_flip_check": sign_flip_check(groups),
        "per_case": {case: {"questions": len(values), "difference": round(sum(values) / len(values), 3)}
                     for case, values in groups.items()},
        "leave_one_case_out": leave_one_case_out(groups),
        "questions_better_first": sorted(q for q, d in differences.items() if d > 0),
        "questions_better_second": sorted(q for q, d in differences.items() if d < 0),
    }
