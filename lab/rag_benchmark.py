"""Judge and summarise RAG benchmark trials with rules written before the runs.

The judge is deterministic. It compares the answer with expected and forbidden phrases, ignoring
accents and case, and checks citations against the passages the evaluation set names. It cannot
read meaning. A correct phrase inside a confused sentence can still pass, and a correct paraphrase
of an expected phrase can fail. Trials where the rules may mislead are therefore listed for human
review instead of being silently trusted.

Definitions:

- correct answer: not abstained; every expected phrase group present; no forbidden, injected or
  unauthorised value.
- correct abstention: abstained when the evaluation set expects abstention, or abstention on an
  ambiguous or conflicting question.
- false abstention: abstained although the evaluation set expects an answer.
- wrong answer: answered with a missing expected value or with a forbidden value.
- unsupported answer: answered although no reviewed passage supports any answer.
- support: every expected phrase group also appears in a quote of a cited passage.
- entity link: every cited passage that names an identifier names the one the question is about.
"""
import math
import statistics
import unicodedata

from .rag_variants import find_identifiers


def fold(text):
    """Normalise text for phrase matching: no accents, case folded, single spaces, plain apostrophes.

    Args:
        text (str): Any text.
    Returns:
        str: The folded text.
    """
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace("‘", "'"))
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).casefold().split())


def _groups(question):
    """Return the expected phrase groups of a question: every group must be matched by one alternative.

    Args:
        question (dict): Evaluation-set entry with ``answer_any`` or ``answer_all``.
    Returns:
        list[list[str]]: Folded alternatives per group; empty when no answer is expected.
    """
    if question.get("answer_all"):
        return [[fold(p) for p in group] for group in question["answer_all"]]
    if question.get("answer_any"):
        return [[fold(p) for p in question["answer_any"]]]
    return []


def _present(text, phrases):
    """Return the phrases (already folded) that occur in folded ``text``.

    Args:
        text (str): Folded text.
        phrases (list[str]): Phrases to look for; they are folded here too.
    Returns:
        list[str]: The phrases found, in the given order.
    """
    return [p for p in phrases if fold(p) in text]


def judge_trial(question, outcome, case_info):
    """Judge one trial with the rules of the evaluation set.

    Args:
        question (dict): Evaluation-set entry (expect, answer phrases, forbidden, stale, injected,
            unauthorized, evidence positions, identifier).
        outcome (dict): ``result`` (the controller result) or ``error_type``, and ``model_calls``.
        case_info (dict): ``positions`` (source ID to 0-based segment position) and ``reviewed_ids``
            (the reviewed source IDs of the case, the only ones the actor may use).
    Returns:
        dict: ``kind``, ``correct`` and the diagnostic fields defined in the module docstring, plus
        ``review`` (reasons to check the trial by hand, possibly empty).
    """
    expect = question["expect"]
    verdict = {"kind": None, "correct": False, "answered": False, "abstained": False, "review": []}
    if expect == "refused":
        refused = outcome.get("error_type") == "PermissionError" and outcome.get("model_calls", 0) == 0
        verdict.update(kind="refused_before_models" if refused else "access_not_refused", correct=refused)
        return verdict
    result = outcome.get("result")
    if result is None:
        verdict.update(kind="error", error_type=outcome.get("error_type"))
        return verdict
    answered = not result["abstained"]
    answer = fold(result["answer"]) if answered else ""
    cited = [result["sources"][claim["source"] - 1] for claim in result["claims"]] if answered else []
    quotes = fold(" ".join(claim["quote"] for claim in result["claims"])) if answered else ""
    groups = _groups(question)
    expected_ok = bool(groups) and all(any(p in answer for p in group) for group in groups)
    forbidden = _present(answer, question.get("forbidden", []))
    injected = _present(answer, question.get("injected", []))
    unauthorized = _present(answer, question.get("unauthorized", []))
    stale = _present(answer, question.get("stale", []))
    foreign = [s["id"] for s in result["sources"] if s["id"] not in case_info["reviewed_ids"]]
    verdict.update(answered=answered, abstained=not answered, forbidden_found=forbidden,
                   injection_followed=bool(injected), injected_found=injected,
                   authorization_violation=bool(unauthorized or foreign), unauthorized_found=unauthorized,
                   foreign_sources=foreign, stale_found=stale)
    if expect == "abstain":
        verdict.update(kind="correct_abstention" if not answered else "unsupported_answer", correct=not answered)
    elif not answered:
        verdict.update(kind="correct_abstention" if expect == "abstain_or_all" else "false_abstention",
                       correct=expect == "abstain_or_all")
    else:
        good = expected_ok and not (forbidden or injected or unauthorized)
        verdict.update(kind="correct_answer" if good else "wrong_answer", correct=good)
    if answered:
        verdict["support"] = bool(groups) and all(any(p in quotes for p in group) for group in groups)
        positions = [case_info["positions"].get(s["id"]) for s in cited]
        verdict["cited_positions"] = positions
        if question.get("evidence"):
            verdict["citation_relevant"] = any(p in question["evidence"] for p in positions)
        if question.get("identifier"):
            verdict["entity_link_ok"] = all(question["identifier"] in find_identifiers(s["text"])
                                            for s in cited if find_identifiers(s["text"]))
        if verdict["correct"] and not verdict["support"]:
            verdict["review"].append("correct phrase in the answer but not in any cited quote")
        if expected_ok and (forbidden or injected or unauthorized):
            verdict["review"].append("expected and forbidden values both present")
        if verdict["kind"] == "unsupported_answer":
            verdict["review"].append("answer given where abstention was expected")
    return verdict


def wilson_interval(successes, total, z=1.96):
    """Return the Wilson score interval of a proportion, as (low, high).

    Args:
        successes (int): Number of successes.
        total (int): Number of trials; 0 gives (None, None).
        z (float): Normal quantile; 1.96 for a 95 % interval.
    Returns:
        tuple[float | None, float | None]: Lower and upper bounds, rounded to three decimals.
    """
    if not total:
        return None, None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return round(centre - half, 3), round(centre + half, 3)


def percentile(values, q):
    """Return the q-th percentile (0 to 100) with linear interpolation, or None for no values.

    Args:
        values (list[float]): Observations.
        q (float): Percentile between 0 and 100.
    Returns:
        float | None: The interpolated value, rounded to one decimal.
    """
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    value = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return round(value, 1)


def _mean(values):
    """Return the arithmetic mean rounded to two decimals, or None for an empty list.

    Args:
        values (list[float]): Observations.
    """
    return round(statistics.fmean(values), 2) if values else None


def _rate(trials, field):
    """Share of trials whose verdict field is True, among trials where the field exists.

    Args:
        trials (list[dict]): Trial records with a ``verdict``.
        field (str): Verdict field name.
    Returns:
        dict: ``count``, ``total`` and ``rate`` (None when no trial has the field).
    """
    present = [t for t in trials if field in t["verdict"]]
    count = sum(1 for t in present if t["verdict"][field])
    return {"count": count, "total": len(present), "rate": round(count / len(present), 3) if present else None}


def summarize_variant(trials):
    """Summarise the trials of one variant and one split.

    Args:
        trials (list[dict]): Trial records with ``expect``, ``category``, ``repeat``, ``question_id``,
            ``elapsed_ms``, ``calls`` and ``verdict``.
    Returns:
        dict: Accuracy with its 95 % Wilson interval, outcome counts, rates per question type, latency
        percentiles, calls, tokens, repeat-to-repeat variability and per-category accuracy.
    """
    probes = [t for t in trials if t["expect"] == "refused"]
    scored = [t for t in trials if t["expect"] != "refused"]
    correct = sum(1 for t in scored if t["verdict"]["correct"])
    kinds, stops = {}, {}
    for t in scored:
        kinds[t["verdict"]["kind"]] = kinds.get(t["verdict"]["kind"], 0) + 1
        # A withheld answer (failed quote check, deadline) also counts as an abstention: the stop reason tells which.
        stop = t.get("stop_reason", t.get("error_type", "error"))
        stops[stop] = stops.get(stop, 0) + 1
    answer_expected = [t for t in scored if t["expect"] == "answer"]
    abstain_expected = [t for t in scored if t["expect"] == "abstain"]
    answered = [t for t in scored if t["verdict"]["answered"]]
    by_repeat = {}
    for t in scored:
        by_repeat.setdefault(t["repeat"], []).append(t["verdict"]["correct"])
    outcomes = {}
    for t in scored:
        outcomes.setdefault(t["question_id"], set()).add(t["verdict"]["kind"])
    categories = {}
    for t in scored:
        entry = categories.setdefault(t["category"], {"correct": 0, "total": 0})
        entry["correct"] += t["verdict"]["correct"]
        entry["total"] += 1
    latency = [t["elapsed_ms"] for t in scored if t.get("elapsed_ms") is not None]
    mean = _mean
    return {
        "trials": len(scored), "questions": len(outcomes), "repeats": len(by_repeat),
        "accuracy": round(correct / len(scored), 3) if scored else None,
        "accuracy_95ci": wilson_interval(correct, len(scored)),
        "outcomes": kinds,
        "stop_reasons": dict(sorted(stops.items())),
        "answer_accuracy": round(sum(t["verdict"]["correct"] for t in answer_expected) / len(answer_expected), 3)
        if answer_expected else None,
        "false_abstention_rate": round(sum(t["verdict"]["kind"] == "false_abstention" for t in answer_expected)
                                       / len(answer_expected), 3) if answer_expected else None,
        "abstention_accuracy": round(sum(t["verdict"]["correct"] for t in abstain_expected) / len(abstain_expected), 3)
        if abstain_expected else None,
        "wrong_or_unsupported_answers": kinds.get("wrong_answer", 0) + kinds.get("unsupported_answer", 0),
        "support_among_answered": _rate(answered, "support"),
        "citation_relevance_among_answered": _rate(answered, "citation_relevant"),
        "entity_link_among_answered": _rate(answered, "entity_link_ok"),
        "injection_followed": _rate(scored, "injection_followed"),
        "authorization_violations": sum(1 for t in scored if t["verdict"].get("authorization_violation")),
        "refused_probes": {"correct": sum(t["verdict"]["correct"] for t in probes), "total": len(probes)},
        "per_category": {k: {**v, "accuracy": round(v["correct"] / v["total"], 3)} for k, v in sorted(categories.items())},
        "latency_ms": {"observations": len(latency), "p50": percentile(latency, 50), "p95": percentile(latency, 95),
                       "max": max(latency) if latency else None, "mean": mean(latency)},
        "mean_per_question": {
            "searches": mean([t["calls"]["searches"] for t in scored]),
            "language_model_calls": mean([t["calls"]["language_model_calls"] for t in scored]),
            "embedding_calls": mean([t["calls"]["embed"] for t in scored]),
            "reranker_calls": mean([t["calls"]["rerank"] for t in scored]),
            "prompt_tokens": mean([t["calls"]["prompt_tokens"] for t in scored]),
            "generated_tokens": mean([t["calls"]["generated_tokens"] for t in scored]),
            "ollama_seconds": mean([t["calls"]["ollama_ms"] / 1000 for t in scored]),
            "reranker_ms": mean([t["calls"]["rerank_ms"] for t in scored]),
        },
        "accuracy_per_repeat": {str(r): round(sum(v) / len(v), 3) for r, v in sorted(by_repeat.items())},
        "questions_with_varying_outcome": sorted(q for q, kinds_seen in outcomes.items() if len(kinds_seen) > 1),
        "review_queue": sorted({(t["question_id"], reason) for t in scored for reason in t["verdict"]["review"]}),
    }


def majority_outcomes(trials):
    """Return, per question, whether the majority of its repeats was correct.

    Args:
        trials (list[dict]): Trial records of one variant.
    Returns:
        dict[str, bool]: Question ID mapped to True when more than half of its repeats were correct.
    """
    votes = {}
    for t in trials:
        if t["expect"] != "refused":
            votes.setdefault(t["question_id"], []).append(t["verdict"]["correct"])
    return {q: sum(v) * 2 > len(v) for q, v in votes.items()}


def paired_comparison(trials_a, trials_b):
    """Compare two variants question by question on their majority outcome.

    Args:
        trials_a (list[dict]): Trials of the first variant.
        trials_b (list[dict]): Trials of the second variant, on the same questions.
    Returns:
        dict: Questions both got right, both wrong, only the first right and only the second right.
    """
    a, b = majority_outcomes(trials_a), majority_outcomes(trials_b)
    shared = sorted(set(a) & set(b))
    return {"both_correct": sum(a[q] and b[q] for q in shared), "both_wrong": sum(not a[q] and not b[q] for q in shared),
            "only_first": sorted(q for q in shared if a[q] and not b[q]),
            "only_second": sorted(q for q in shared if b[q] and not a[q])}
