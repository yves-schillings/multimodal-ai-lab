"""Review SHAP and LIME explanations of two synthetic invoice classifiers question by question.

The 27 October experiment in lab.engineering_explainability explains one case with
Tree SHAP on boosted trees and LIME on a logistic regression. This module leaves
that code and its results unchanged. It reuses the same synthetic data, models and
600/200 split, and answers separate questions on the same fixed stratified test
cases and the same output: predictive performance and calibration, SHAP additivity,
LIME local fidelity, the separate [0, 1] range diagnostic, stability across LIME
seeds and SHAP backgrounds, sensitivity to the reference background, and SHAP-LIME
agreement. Every number names its scale: class-1 log-odds (decision_function) or
class-1 probability (predict_proba[:, 1]).

It also reports the dependence between the synthetic features, the logistic
model's coefficients with bootstrap intervals and an interaction check, and a
minimal bounded counterfactual validated on each model. THRESHOLDS mirrors the
illustrative values fixed in the review protocol before the run.

Importing this module uses only the standard library and the lazily importing base
module. Scientific packages load when a function runs. Requested runs use the CPU
with one numerical thread, reuse the base module's in-process model cache, and do
not write files, call a network service or use a GPU.

A passed check answers only its own question. No result here establishes reviewer
usefulness, causality, fairness, real-invoice performance or production readiness."""
from __future__ import annotations

import hashlib
from importlib import metadata
from itertools import combinations
from math import factorial, isfinite, sqrt
import os
import platform
import time

from lab.engineering_explainability import (
    CLASS_NAMES, FEATURES, _bounded_int, _fit_synthetic_model, _fit_synthetic_tree_model,
    _linear_coefficients, validate_invoice,
)


SCALES = ("log_odds", "probability")
SCALE_LABELS = {
    "log_odds": "class-1 log-odds: the raw margin returned by decision_function",
    "probability": "class-1 probability: predict_proba[:, 1]",
}
MODEL_LABELS = {
    "logistic_regression": "StandardScaler followed by sklearn LogisticRegression",
    "hist_gradient_boosting": "sklearn HistGradientBoostingClassifier",
}
THRESHOLDS = {
    "additivity_abs_tolerance": 1e-8,
    "implementation_crosscheck_abs_tolerance": 1e-6,
    "lime_case_abs_error_probability": 0.05,
    "lime_case_abs_error_log_odds": 0.2,
    "lime_heldout_weighted_r2_min": 0.8,
    "stability_top3_jaccard_min": 0.8,
    "stability_spearman_min": 0.8,
    "background_top3_jaccard_min": 0.8,
    "background_spearman_min": 0.8,
    "agreement_spearman_min": 0.8,
    "agreement_top3_sign_fraction_min": 1.0,
    "no_interaction_abs_tolerance": 1e-9,
}
CASE_SELECTION_SEED_OFFSET = 3000
BACKGROUND_SEED_OFFSET = 101
PLAUSIBLE_NEIGHBOURS = 256
PLAUSIBLE_SEED_OFFSET = 9301
BOOTSTRAP_SEED_OFFSET = 2000
COEFFICIENT_SEED_OFFSET = 4000
RELIABILITY_BINS = 10
EQUAL_MASS_BINS = 5
COUNTERFACTUAL_AMOUNT_STEP = 0.01
TIE_SHIFT = 1e-6
KERNEL_WIDTH = 0.75 * sqrt(len(FEATURES))
CATEGORICAL = tuple(index for index, feature in enumerate(FEATURES) if feature["binary"])
FEATURE_KEYS = tuple(feature["key"] for feature in FEATURES)
STRATA = (
    ("A", 1, True, "true label 1 and both models correct"),
    ("B", 0, True, "true label 0 and both models correct"),
    ("C", 1, False, "true label 1 and at least one model wrong"),
    ("D", 0, False, "true label 0 and at least one model wrong"),
)
MUTABLE_BINARY_FIELDS = ("purchase_order_match", "delivery_confirmed")
IMMUTABLE_FIELDS = ("supplier_history_days", "duplicate_reference", "recent_bank_change")
REFERENCE_INVOICE = {
    "amount_ratio": 2.4, "purchase_order_match": 0, "supplier_history_days": 90,
    "duplicate_reference": 1, "delivery_confirmed": 0, "recent_bank_change": 1,
}
# Copy of the label-generating logit in _fit_synthetic_model, per original unit.
GENERATING_LOGIT = {
    "intercept": 1.0, "amount_ratio": 0.85, "purchase_order_match": -2.3,
    "supplier_history_days": -0.001, "duplicate_reference": 2.5,
    "delivery_confirmed": -1.35, "recent_bank_change": 1.1,
}
REVIEWER_STUDY = {
    "status": "Proposed protocol only. Not run.",
    "task": "Reviewers see synthetic invoice cases with the model score and decide the route (routine or review) and whether the score looks wrong. Conditions: score only, score with SHAP bars, score with LIME weights, score with a counterfactual. Some cases come from a deliberately corrupted model with one feature sign inverted.",
    "participants": "At least 12 accounts-payable or audit reviewers who did not build the model; within-subject design, conditions counterbalanced with a Latin square, cases randomised.",
    "measures": "Primary: detection rate of corrupted-model cases. Secondary: decision accuracy against the synthetic ground truth, forward-simulation accuracy, over-reliance rate on wrong scores, time per case, and a short trust and workload questionnaire.",
    "analysis": "Mixed-effects logistic regression with reviewer and case as random effects; effect sizes with 95 percent intervals; pre-registered before recruitment.",
}


def review_parameters(
    *, seed: int = 1729, n_samples: int = 800, lime_samples: int = 4000,
    cases_per_stratum: int = 5, lime_seeds: int = 10, shap_backgrounds: int = 5,
    background_rows: int = 100, bootstrap_resamples: int = 1000,
    coefficient_bootstrap_resamples: int = 200,
) -> dict:
    """Validate the review budgets and return the complete parameter record.

    Args:
        seed (int): Generation, split and fitting seed, 0 through 2**31 - 1.
        n_samples (int): Total synthetic rows, 200 through 5000 (25 percent held out).
        lime_samples (int): LIME perturbations per explanation, 500 through 20000.
        cases_per_stratum (int): Test cases drawn per stratum A-D, 1 through 25.
        lime_seeds (int): LIME random states per case for stability, 2 through 20.
        shap_backgrounds (int): Training background samples for SHAP stability, 2 through 10.
        background_rows (int): Rows per SHAP background, 10 through 150 (the smallest
            permitted training split has 150 rows).
        bootstrap_resamples (int): Test-set bootstrap resamples, 10 through 5000.
        coefficient_bootstrap_resamples (int): Logistic refits on resampled training
            rows, 10 through 1000.

    Returns:
        dict: Every tunable value plus the fixed offsets, bin counts, neighbourhood size,
        counterfactual grid step and thread limit, in the same shape as the protocol's
        parameters object, so a caller can compare the two before running.

    Raises:
        ValueError: A value is boolean, non-integer or outside its inclusive bounds.

    This pure helper performs no computation beyond validation."""
    _bounded_int("seed", seed, 0, 2**31 - 1)
    _bounded_int("n_samples", n_samples, 200, 5000)
    _bounded_int("lime_samples", lime_samples, 500, 20000)
    _bounded_int("cases_per_stratum", cases_per_stratum, 1, 25)
    _bounded_int("lime_seeds", lime_seeds, 2, 20)
    _bounded_int("shap_backgrounds", shap_backgrounds, 2, 10)
    _bounded_int("background_rows", background_rows, 10, 150)
    _bounded_int("bootstrap_resamples", bootstrap_resamples, 10, 5000)
    _bounded_int("coefficient_bootstrap_resamples", coefficient_bootstrap_resamples, 10, 1000)
    return {
        "seed": seed, "n_samples": n_samples, "test_fraction": 0.25, "decision_threshold": 0.5,
        "cases_per_stratum": cases_per_stratum, "case_selection_seed_offset": CASE_SELECTION_SEED_OFFSET,
        "background_rows": background_rows, "shap_backgrounds": shap_backgrounds,
        "background_seed_offsets": [BACKGROUND_SEED_OFFSET + index for index in range(shap_backgrounds)],
        "lime_samples": lime_samples, "lime_seeds": lime_seeds, "lime_seed_offsets": list(range(lime_seeds)),
        "plausible_neighbours": PLAUSIBLE_NEIGHBOURS, "plausible_seed_offset": PLAUSIBLE_SEED_OFFSET,
        "reliability_bins": RELIABILITY_BINS, "equal_mass_bins": EQUAL_MASS_BINS,
        "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed_offset": BOOTSTRAP_SEED_OFFSET,
        "coefficient_bootstrap_resamples": coefficient_bootstrap_resamples,
        "coefficient_bootstrap_seed_offset": COEFFICIENT_SEED_OFFSET,
        "counterfactual_amount_step": COUNTERFACTUAL_AMOUNT_STEP, "threads": 1,
    }


def calibration_table(y_true, probabilities, *, bins: int = RELIABILITY_BINS, equal_mass_groups: int = EQUAL_MASS_BINS) -> dict:
    """Build a reliability table and two expected calibration errors.

    Args:
        y_true: Sequence of observed labels, each 0 or 1.
        probabilities: Sequence of predicted class-1 probabilities in [0, 1], same length.
        bins (int): Equal-width bins on [0, 1], 1 through 100. A probability of exactly
            1.0 falls into the last bin.
        equal_mass_groups (int): Groups of nearly equal size over the sorted
            probabilities (stable sort), 1 through 100, for the secondary ECE.

    Returns:
        dict: scale ('probability'), one row per equal-width bin with count, mean
        predicted probability, observed positive rate and gap (null for empty bins),
        ece_equal_width, one row per equal-mass group, and ece_equal_mass. ECE is the
        count-weighted mean absolute gap over non-empty bins.

    Raises:
        ValueError: Empty or mismatched inputs, labels outside {0, 1}, non-finite or
            out-of-range probabilities, or invalid bin counts.

    NumPy is imported lazily. The table describes this sample only; with few rows per
    bin the gaps are noisy and ECE depends on the binning."""
    import numpy as np
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("y_true and probabilities must be non-empty and of equal length")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    _bounded_int("bins", bins, 1, 100)
    _bounded_int("equal_mass_groups", equal_mass_groups, 1, 100)
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for number in range(bins):
        members = index == number
        row = {"bin": number, "lower": number / bins, "upper": (number + 1) / bins, "count": int(members.sum()),
               "mean_predicted": None, "observed_rate": None, "gap": None}
        if row["count"]:
            mean_predicted, observed = float(p[members].mean()), float(y[members].mean())
            row.update(mean_predicted=mean_predicted, observed_rate=observed, gap=mean_predicted - observed)
            ece += row["count"] / y.size * abs(mean_predicted - observed)
        rows.append(row)
    groups, mass_ece = [], 0.0
    for number, members in enumerate(np.array_split(np.argsort(p, kind="stable"), equal_mass_groups)):
        if not members.size:
            continue
        mean_predicted, observed = float(p[members].mean()), float(y[members].mean())
        groups.append({"group": number, "count": int(members.size), "mean_predicted": mean_predicted,
                       "observed_rate": observed, "gap": mean_predicted - observed})
        mass_ece += members.size / y.size * abs(mean_predicted - observed)
    return {"scale": "probability", "bins": rows, "ece_equal_width": float(ece),
            "equal_mass_groups": groups, "ece_equal_mass": float(mass_ece)}


def _performance(deps, y_true, probabilities, predictions, *, seed: int, resamples: int) -> dict:
    """Measure held-out performance and calibration with bootstrap intervals.

    Args:
        deps (dict): Scientific namespace from the base bundle (np and sklearn metrics).
        y_true (numpy.ndarray): Observed test labels, 0 or 1.
        probabilities (numpy.ndarray): Predicted class-1 probabilities for the same rows.
        predictions (numpy.ndarray): Estimator class predictions at its 0.5 rule.
        seed (int): Review seed; the bootstrap uses default_rng(seed + 2000).
        resamples (int): Number of percentile-bootstrap resamples of the rows.

    Returns:
        dict: Point accuracy, ROC AUC, Brier score, log loss and equal-width ECE, the
        reliability table, 95 percent percentile intervals for each metric, and the
        number of resamples skipped for AUC because they contained one class.

    Metric or calibration errors propagate. The intervals describe resampling of
    these 200 synthetic rows, not uncertainty about real invoices."""
    np = deps["np"]
    from sklearn.metrics import log_loss

    def measure(y, p, labels):
        """Compute the five point metrics for one resample.

        Args:
            y (numpy.ndarray): Labels of the resample.
            p (numpy.ndarray): Class-1 probabilities of the resample.
            labels (numpy.ndarray): Class predictions of the resample.

        Returns:
            dict: accuracy, roc_auc (None when the resample has one class), brier_score,
            log_loss and ece_equal_width, all on the probability or 0-1 scale."""
        return {
            "accuracy": float(deps["accuracy_score"](y, labels)),
            "roc_auc": float(deps["roc_auc_score"](y, p)) if np.unique(y).size == 2 else None,
            "brier_score": float(deps["brier_score_loss"](y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece_equal_width": calibration_table(y, p)["ece_equal_width"],
        }

    point = measure(y_true, probabilities, predictions)
    rng = np.random.default_rng(seed + BOOTSTRAP_SEED_OFFSET)
    draws = {name: [] for name in point}
    skipped_auc = 0
    for _ in range(resamples):
        rows = rng.integers(0, y_true.size, y_true.size)
        sample = measure(y_true[rows], probabilities[rows], predictions[rows])
        for name, value in sample.items():
            if value is None:
                skipped_auc += 1
            else:
                draws[name].append(value)
    return {
        "rows": int(y_true.size), "positive_rate": float(y_true.mean()),
        "point": point,
        "bootstrap_95_percent": {
            name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] for name, values in draws.items()
        },
        "bootstrap_resamples": resamples, "bootstrap_seed": seed + BOOTSTRAP_SEED_OFFSET,
        "auc_resamples_skipped_single_class": skipped_auc,
        "reliability": calibration_table(y_true, probabilities),
        "scales": "accuracy is a 0-1 proportion; ROC AUC is a ranking probability; Brier, log loss and ECE use predicted class-1 probabilities.",
    }


def select_review_cases(y_true, linear_predictions, tree_predictions, *, seed: int, per_stratum: int) -> dict:
    """Choose the fixed stratified test cases explained for both models.

    Args:
        y_true: Observed test labels, 0 or 1.
        linear_predictions: Logistic-regression class predictions for the same rows.
        tree_predictions: Boosted-tree class predictions for the same rows.
        seed (int): Review seed; one default_rng(seed + 3000) draws every stratum in order.
        per_stratum (int): Cases requested per stratum, 1 through 25.

    Returns:
        dict: The rule, sorted selected test positions, one entry per stratum (code,
        description, available, requested, selected, shortfall and positions), and a
        per-position list giving each case's stratum.

    Raises:
        ValueError: Inputs differ in length, contain values other than 0 and 1, or
            per_stratum is out of bounds.

    Strata: A label 1 both correct, B label 0 both correct, C label 1 at least one model
    wrong, D label 0 at least one model wrong. A short stratum is taken whole and its
    shortfall is reported; nothing is substituted. The choice depends only on labels and
    predictions, never on explanation results."""
    import numpy as np
    y = np.asarray(y_true).reshape(-1)
    linear = np.asarray(linear_predictions).reshape(-1)
    tree = np.asarray(tree_predictions).reshape(-1)
    if not y.size or y.shape != linear.shape or y.shape != tree.shape:
        raise ValueError("labels and both prediction arrays must be non-empty and of equal length")
    if not all(np.all(np.isin(values, (0, 1))) for values in (y, linear, tree)):
        raise ValueError("labels and predictions must contain only 0 and 1")
    _bounded_int("per_stratum", per_stratum, 1, 25)
    both_correct = (linear == y) & (tree == y)
    rng = np.random.default_rng(seed + CASE_SELECTION_SEED_OFFSET)
    strata, cases = [], []
    for code, label, correct, description in STRATA:
        members = np.nonzero((y == label) & (both_correct == correct))[0]
        take = min(per_stratum, int(members.size))
        chosen = np.sort(rng.choice(members, take, replace=False)) if take else np.array([], dtype=int)
        strata.append({"code": code, "description": description, "available": int(members.size),
                       "requested": per_stratum, "selected": take, "shortfall": per_stratum - take,
                       "positions": [int(position) for position in chosen]})
        cases.extend({"position": int(position), "stratum": code} for position in chosen)
    cases.sort(key=_case_position)
    return {
        "rule": "Four strata from labels and both models' predictions at threshold 0.5; default_rng(seed + 3000) draws min(per_stratum, size) positions per stratum without replacement, in the order A, B, C, D; no substitution.",
        "positions": [case["position"] for case in cases], "strata": strata, "cases": cases,
    }


def _case_position(case: dict) -> int:
    """Return the test position used to sort selected cases.

    Args:
        case (dict): One selected case with an integer position.

    Returns:
        int: case['position']. A missing key raises KeyError."""
    return case["position"]


def _top_k(values, k: int = 3) -> frozenset:
    """Return the indices of the k largest absolute attributions.

    Args:
        values: Sequence of finite attributions in schema order.
        k (int): Number of leading features to keep.

    Returns:
        frozenset: Feature indices. A stable sort keeps schema order for ties."""
    import numpy as np
    order = np.argsort(-np.abs(np.asarray(values, dtype=float)), kind="stable")
    return frozenset(int(index) for index in order[:k])


def _jaccard(first: frozenset, second: frozenset) -> float:
    """Return the Jaccard similarity of two feature sets.

    Args:
        first (frozenset): Feature indices from one explanation.
        second (frozenset): Feature indices from another explanation.

    Returns:
        float: Intersection size divided by union size, or 1.0 for two empty sets."""
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


def _spearman(first, second):
    """Return the Spearman correlation of two absolute attribution vectors.

    Args:
        first: Attributions of one explanation in schema order.
        second: Attributions of another explanation in the same order.

    Returns:
        float | None: Rank correlation of the absolute values with average ranks for
        ties, or None when either vector is constant (correlation undefined)."""
    import numpy as np
    from scipy.stats import spearmanr
    a, b = np.abs(np.asarray(first, dtype=float)), np.abs(np.asarray(second, dtype=float))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if isfinite(value) else None


def _median(values):
    """Return the median of the defined values, or None when there are none.

    Args:
        values: Iterable of numbers or None.

    Returns:
        float | None: Median of the non-None values."""
    import numpy as np
    defined = [value for value in values if value is not None]
    return float(np.median(defined)) if defined else None


def _minimum(values):
    """Return the minimum of the defined values, or None when there are none.

    Args:
        values: Iterable of numbers or None.

    Returns:
        float | None: Smallest non-None value."""
    defined = [value for value in values if value is not None]
    return float(min(defined)) if defined else None


def _pairwise_stability(runs, jaccard_min: float, spearman_min: float) -> dict:
    """Summarise how much one case's attributions vary across repeated runs.

    Args:
        runs: Two or more attribution vectors for the same case, model and scale,
            one per LIME seed or SHAP background.
        jaccard_min (float): Minimum mean pairwise top-3 Jaccard for a pass.
        spearman_min (float): Minimum mean pairwise Spearman for a pass.

    Returns:
        dict: Number of runs and pairs, mean and minimum pairwise top-3 Jaccard, mean and
        minimum Spearman of absolute attributions, undefined Spearman pairs, features
        whose sign differs between runs, how often each feature is in the top 3, and
        the pass flag. An undefined correlation prevents a pass.

    The pairs compare repeated explanations, not explanations with the model."""
    import numpy as np
    matrix = np.asarray(runs, dtype=float)
    jaccards, spearmans = [], []
    for first, second in combinations(range(len(matrix)), 2):
        jaccards.append(_jaccard(_top_k(matrix[first]), _top_k(matrix[second])))
        spearmans.append(_spearman(matrix[first], matrix[second]))
    defined = [value for value in spearmans if value is not None]
    undefined = len(spearmans) - len(defined)
    signs = np.sign(matrix)
    mean_jaccard = float(np.mean(jaccards))
    mean_spearman = float(np.mean(defined)) if defined else None
    return {
        "runs": int(len(matrix)), "pairs": len(jaccards),
        "mean_top3_jaccard": mean_jaccard, "min_top3_jaccard": float(np.min(jaccards)),
        "mean_spearman_abs": mean_spearman, "min_spearman_abs": float(np.min(defined)) if defined else None,
        "undefined_spearman_pairs": undefined,
        "features_with_sign_change": [FEATURE_KEYS[index] for index in range(len(FEATURES)) if np.unique(signs[:, index]).size > 1],
        "top3_frequency": {FEATURE_KEYS[index]: int(sum(index in _top_k(row) for row in matrix)) for index in range(len(FEATURES))},
        "passed": bool(undefined == 0 and mean_jaccard >= jaccard_min and mean_spearman is not None and mean_spearman >= spearman_min),
    }


def _output_functions(model) -> dict:
    """Return the two explained outputs of one fitted binary classifier.

    Args:
        model: Fitted estimator or pipeline with decision_function and predict_proba.

    Returns:
        dict: 'log_odds' and 'probability' callables taking an (n, 6) array in original
        units and returning a length-n float array."""
    def log_odds(rows):
        """Return the class-1 raw margin, which is the class-1 log-odds here.

        Args:
            rows (numpy.ndarray): Inputs in original units and schema order.

        Returns:
            numpy.ndarray: decision_function values. Model errors propagate."""
        return model.decision_function(rows)

    def probability(rows):
        """Return the class-1 probability of the classifier.

        Args:
            rows (numpy.ndarray): Inputs in original units and schema order.

        Returns:
            numpy.ndarray: predict_proba(rows)[:, 1]. Model errors propagate."""
        return model.predict_proba(rows)[:, 1]

    return {"log_odds": log_odds, "probability": probability}


def _exact_shap(shap, np, function, background, rows):
    """Compute interventional SHAP values with shap's exact enumeration.

    Args:
        shap: Imported shap module.
        np: Imported NumPy module.
        function: Callable returning one output per row on the explained scale.
        background (numpy.ndarray): Reference rows in original units.
        rows (numpy.ndarray): Rows to explain in original units.

    Returns:
        tuple: (values of shape (n, 6), base values of shape (n,), model outputs of
        shape (n,)). Masked features take background values independently of the kept
        features, which is the interventional assumption.

    shap.explainers.Exact enumerates all 64 coalitions; explainer errors propagate."""
    explainer = shap.explainers.Exact(function, shap.maskers.Independent(background, max_samples=len(background)))
    explanation = explainer(rows, silent=True)
    values = np.asarray(explanation.values, dtype=float).reshape(len(rows), len(FEATURES))
    base = np.asarray(explanation.base_values, dtype=float).reshape(len(rows))
    return values, base, np.asarray(function(rows), dtype=float).reshape(len(rows))


def _enumerated_shap(np, function, background, row):
    """Compute interventional Shapley values by direct enumeration, without shap.

    Args:
        np: Imported NumPy module.
        function: Callable returning one output per row on the explained scale.
        background (numpy.ndarray): Reference rows in original units.
        row (numpy.ndarray): One case in original units.

    Returns:
        tuple: (values of shape (6,), base value v(empty set), full value v(all
        features)). v(S) is the mean output over background rows whose features in S
        are replaced by the case's values; values use the Shapley weights
        |S|! (M - |S| - 1)! / M!.

    This independent reference checks shap's implementations. It evaluates 2**6 times
    the background size rows in one call."""
    count = len(FEATURES)
    masks = ((np.arange(2**count)[:, None] >> np.arange(count)) & 1).astype(bool)
    hybrids = np.where(masks[:, None, :], row[None, None, :], background[None, :, :])
    coalition = np.asarray(function(hybrids.reshape(-1, count)), dtype=float).reshape(2**count, len(background)).mean(axis=1)
    sizes = masks.sum(axis=1)
    weights = np.array([factorial(size) * factorial(count - size - 1) / factorial(count) if size < count else 0.0 for size in sizes])
    values = np.zeros(count)
    for index in range(count):
        without = np.nonzero(~masks[:, index])[0]
        values[index] = float(np.sum(weights[without] * (coalition[without | (1 << index)] - coalition[without])))
    return values, float(coalition[0]), float(coalition[-1])


def _model_specific_shap(shap, np, model_id: str, model, background, rows, scale: str):
    """Run the model-specific interventional SHAP explainer when one exists.

    Args:
        shap: Imported shap module.
        np: Imported NumPy module.
        model_id (str): 'logistic_regression' or 'hist_gradient_boosting'.
        model: The fitted pipeline or boosted-tree classifier.
        background (numpy.ndarray): Primary background rows in original units.
        rows (numpy.ndarray): Rows to explain in original units.
        scale (str): 'log_odds' or 'probability'.

    Returns:
        tuple: (values of shape (n, 6) or None, the explainer's own base values of shape
        (n,) or None, method description). TreeExplainer uses
        feature_perturbation='interventional' with model_output 'raw' or 'probability';
        LinearExplainer explains the logistic margin on scaled inputs. The logistic
        probability has no model-specific explainer, so None is returned with the reason.

    shap's own additivity assertion is disabled here because the review measures
    additivity and implementation agreement itself; other explainer errors propagate."""
    if model_id == "hist_gradient_boosting":
        output = "raw" if scale == "log_odds" else "probability"
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output=output)
        explanation = explainer(rows, check_additivity=False)
        values = np.asarray(explanation.values, dtype=float).reshape(len(rows), len(FEATURES))
        base = np.broadcast_to(np.asarray(explanation.base_values, dtype=float).reshape(-1), (len(rows),)).copy()
        return values, base, f"TreeExplainer, interventional, model_output='{output}'"
    if scale == "log_odds":
        scaler, estimator = model.steps[0][1], model.steps[1][1]
        masker = shap.maskers.Independent(scaler.transform(background), max_samples=len(background))
        explanation = shap.LinearExplainer(estimator, masker)(scaler.transform(rows))
        values = np.asarray(explanation.values, dtype=float).reshape(len(rows), len(FEATURES))
        base = np.broadcast_to(np.asarray(explanation.base_values, dtype=float).reshape(-1), (len(rows),)).copy()
        return values, base, "LinearExplainer with an Independent masker on scaled inputs"
    return None, None, "Not available: LinearExplainer explains the linear margin only, not the probability."


def _tree_thresholds(np, model) -> dict:
    """Collect the split thresholds of a fitted HistGradientBoostingClassifier.

    Args:
        np: Imported NumPy module.
        model: Fitted binary HistGradientBoostingClassifier.

    Returns:
        dict: Feature index mapped to a sorted array of distinct numerical thresholds.

    Reads the private _predictors node arrays of scikit-learn 1.9; an AttributeError
    or KeyError propagates if that internal layout changes."""
    collected = {index: [] for index in range(len(FEATURES))}
    for iteration in model._predictors:
        for predictor in iteration:
            nodes = predictor.nodes[predictor.nodes["is_leaf"] == 0]
            for feature, threshold in zip(nodes["feature_idx"], nodes["num_threshold"]):
                collected[int(feature)].append(float(threshold))
    return {index: np.unique(np.asarray(values, dtype=float)) for index, values in collected.items()}


def _rescaled_margin_shap(np, function, background, row):
    """Rebuild probability attributions by rescaling margin attributions per reference.

    Args:
        np: Imported NumPy module.
        function: Log-odds callable of the boosted trees.
        background (numpy.ndarray): Reference rows in original units.
        row (numpy.ndarray): One case in original units.

    Returns:
        tuple: (values of shape (6,), number of references whose margin equals the
        case's margin). For each reference r the single-reference margin Shapley values
        are multiplied by (sigmoid(f(x)) - sigmoid(f(r))) / (f(x) - f(r)), using the
        derivative sigmoid'(f(x)) when the margins are equal, and then averaged.

    This reproduces how interventional Tree SHAP is observed to produce probability
    output; it is additive per reference but is not the exact Shapley value of the
    probability function."""
    margin = float(function(row.reshape(1, -1))[0])
    case_probability = 1.0 / (1.0 + np.exp(-margin))
    total, equal = np.zeros(len(FEATURES)), 0
    for reference in background:
        values, reference_margin, _ = _enumerated_shap(np, function, reference.reshape(1, -1), row)
        if reference_margin == margin:
            equal += 1
            factor = case_probability * (1.0 - case_probability)
        else:
            factor = (case_probability - 1.0 / (1.0 + np.exp(-reference_margin))) / (margin - reference_margin)
        total += values * factor
    return total / len(background), equal


def _tree_shap_diagnostic(shap, np, model, functions, background, rows) -> dict:
    """Explain where interventional Tree SHAP departs from exact enumeration.

    Args:
        shap: Imported shap module.
        np: Imported NumPy module.
        model: Fitted binary HistGradientBoostingClassifier.
        functions (dict): Output callables from _output_functions for this model.
        background (numpy.ndarray): Primary background rows in original units.
        rows (numpy.ndarray): Explained rows in original units.

    Returns:
        dict: Per-feature counts of background and case values exactly equal to a split
        threshold; after adding 1e-6 to every tied continuous value, the maximum absolute
        Tree SHAP minus enumeration difference per scale, Tree SHAP's additivity error with
        its own base value per scale, and its difference from the rescaled-margin reconstruction,
        with the number of equal-margin references. Diagnostic only: verdicts use the
        unmodified data, and the shifted inputs are different points whose attributions
        are not results."""
    thresholds = _tree_thresholds(np, model)
    shifted_background, shifted_rows = background.copy(), rows.copy()
    ties = {}
    for index, feature in enumerate(FEATURES):
        tied_background = np.isin(background[:, index], thresholds[index])
        tied_rows = np.isin(rows[:, index], thresholds[index])
        ties[feature["key"]] = {"background_values_on_threshold": int(tied_background.sum()),
                                "case_values_on_threshold": int(tied_rows.sum())}
        if not feature["binary"]:
            shifted_background[tied_background, index] += TIE_SHIFT
            shifted_rows[tied_rows, index] += TIE_SHIFT
    tree = {scale: _model_specific_shap(shap, np, "hist_gradient_boosting", model, shifted_background, shifted_rows, scale) for scale in SCALES}
    after = {scale: float(np.abs(tree[scale][0] - np.array([_enumerated_shap(np, functions[scale], shifted_background, row)[0] for row in shifted_rows])).max())
             for scale in SCALES}
    rebuilt = [_rescaled_margin_shap(np, functions["log_odds"], shifted_background, row) for row in shifted_rows]
    return {
        "ties": ties, "shift": TIE_SHIFT,
        "after_shift_tree_shap_vs_enumeration_max_abs_difference": after,
        "after_shift_tree_probability_vs_rescaled_margin_max_abs_difference": float(np.abs(tree["probability"][0] - np.array([item[0] for item in rebuilt])).max()),
        "after_shift_tree_additivity_max_abs_error": {scale: float(np.abs(tree[scale][1] + tree[scale][0].sum(axis=1) - functions[scale](shifted_rows)).max()) for scale in SCALES},
        "equal_margin_references": int(sum(item[1] for item in rebuilt)),
        "scales": "log_odds differences in class-1 log-odds; probability differences in class-1 probability",
        "note": "Diagnostic only. Verdicts use the unmodified data.",
    }


def _lime_explain(deps, bundle, function, row, scale: str, seed: int, lime_samples: int) -> dict:
    """Fit one LIME surrogate on the stated scale and keep its exact perturbations.

    Args:
        deps (dict): Scientific namespace with np and LimeTabularExplainer.
        bundle (dict): Base bundle supplying the 600 training rows.
        function: Callable returning the explained output for original-unit rows.
        row (numpy.ndarray): The case in original units.
        scale (str): 'log_odds' uses LIME regression mode on decision_function;
            'probability' uses classification mode on [1 - p, p], label 1.
        seed (int): LIME random_state.
        lime_samples (int): Number of LIME perturbations.

    Returns:
        dict: The explainer, intercept, coefficients in LIME's design space (continuous:
        per training standard deviation; binary: 1 when equal to the case), LIME's own
        prediction at the case and fitting weighted R2, and the captured perturbations.

    Raises:
        RuntimeError: LIME did not request exactly one batch of the configured size or
            did not place the case first.

    Configuration matches the 27 October helper. No file or network I/O."""
    np = deps["np"]
    captured = []

    def predict(samples):
        """Return the explained output for LIME's perturbations and keep a copy.

        Args:
            samples (numpy.ndarray): LIME perturbations in original units.

        Returns:
            numpy.ndarray: Log-odds vector in regression mode, or the two-column
            [1 - p, p] array in classification mode. Model errors propagate.

        The copy lets the review measure plausibility and fitting metrics on the exact
        rows LIME used."""
        captured.append(np.array(samples, dtype=float, copy=True))
        values = np.asarray(function(samples), dtype=float)
        return values if scale == "log_odds" else np.column_stack((1.0 - values, values))

    options = {
        "mode": "regression" if scale == "log_odds" else "classification",
        "feature_names": [feature["label"] for feature in FEATURES],
        "categorical_features": list(CATEGORICAL),
        "categorical_names": {1: ["not matched", "matched"], 3: ["not repeated", "repeated"],
                              4: ["not confirmed", "confirmed"], 5: ["not changed", "changed"]},
        "discretize_continuous": False, "sample_around_instance": True,
        "random_state": seed, "kernel_width": KERNEL_WIDTH,
    }
    if scale == "probability":
        options["class_names"] = list(CLASS_NAMES)
    explainer = deps["LimeTabularExplainer"](bundle["x_train"], **options)
    call = {"num_features": len(FEATURES), "num_samples": lime_samples}
    if scale == "probability":
        call["labels"] = (1,)
    explanation = explainer.explain_instance(row, predict, **call)
    if len(captured) != 1 or captured[0].shape != (lime_samples, len(FEATURES)) or not np.array_equal(captured[0][0], row):
        raise RuntimeError("LIME did not request one batch of the configured perturbations with the case first")
    coefficients = np.zeros(len(FEATURES))
    # In regression mode LIME stores the fitted signs under label 1 and their negation under 0.
    for index, weight in explanation.local_exp[1]:
        coefficients[int(index)] = float(weight)
    return {
        "explainer": explainer, "intercept": float(explanation.intercept[1]), "coefficients": coefficients,
        "local_prediction": float(np.asarray(explanation.local_pred).reshape(-1)[0]),
        "fit_weighted_r2": float(explanation.score), "samples": captured[0],
    }


def _lime_design(np, explainer, row, samples):
    """Map original-unit rows into the LIME design space of one case.

    Args:
        np: Imported NumPy module.
        explainer: The fitted LimeTabularExplainer (training mean and scale).
        row (numpy.ndarray): The explained case in original units.
        samples (numpy.ndarray): Rows to map, shape (n, 6).

    Returns:
        numpy.ndarray: Continuous columns standardised with LIME's training scaler;
        binary columns 1.0 where the value equals the case's value, else 0.0."""
    design = (samples - explainer.scaler.mean_) / explainer.scaler.scale_
    for index in CATEGORICAL:
        design[:, index] = (samples[:, index] == row[index]).astype(float)
    return design


def _surrogate_metrics(np, r2_score, lime: dict, row, samples, function, scale: str) -> dict:
    """Score a LIME surrogate against the real model on given neighbours.

    Args:
        np: Imported NumPy module.
        r2_score: scikit-learn r2_score callable.
        lime (dict): Result of _lime_explain.
        row (numpy.ndarray): The explained case in original units.
        samples (numpy.ndarray): Neighbours in original units.
        function: Callable returning the explained model output.
        scale (str): Scale of the surrogate and model output.

    Returns:
        dict: rows, scale, kernel-weighted R2, MAE and RMSE of surrogate versus model, and
        on the probability scale the unweighted fraction of surrogate values outside
        [0, 1]. Weights use LIME's own kernel on the design-space distance to the case."""
    explainer = lime["explainer"]
    design = _lime_design(np, explainer, row, samples)
    case_design = _lime_design(np, explainer, row, row.reshape(1, -1))[0]
    weights = explainer.base.kernel_fn(np.linalg.norm(design - case_design, axis=1))
    surrogate = lime["intercept"] + design @ lime["coefficients"]
    actual = np.asarray(function(samples), dtype=float)
    result = {
        "rows": int(len(samples)), "scale": scale,
        "weighted_r2": float(r2_score(actual, surrogate, sample_weight=weights)),
        "weighted_mae": float(np.average(np.abs(actual - surrogate), weights=weights)),
        "weighted_rmse": float(np.sqrt(np.average((actual - surrogate) ** 2, weights=weights))),
    }
    if scale == "probability":
        result["surrogate_outside_unit_interval_fraction"] = float(((surrogate < 0.0) | (surrogate > 1.0)).mean())
    return result


def _lime_attribution(np, lime: dict, row, background):
    """Convert LIME coefficients into contributions relative to a SHAP background.

    Args:
        np: Imported NumPy module.
        lime (dict): Result of _lime_explain.
        row (numpy.ndarray): The explained case in original units.
        background (numpy.ndarray): The primary SHAP background rows.

    Returns:
        numpy.ndarray: coefficient times (design value at the case minus its mean over
        the background), on the surrogate's scale. This is the interventional SHAP value
        of the linear surrogate with that background, so it can be ranked and signed
        against SHAP on the same scale."""
    case_design = _lime_design(np, lime["explainer"], row, row.reshape(1, -1))[0]
    reference = _lime_design(np, lime["explainer"], row, background).mean(axis=0)
    return lime["coefficients"] * (case_design - reference)


def _plausible_neighbours(np, x_train, row, seed: int):
    """Generate the schema-clipped neighbourhood of the 27 October helper.

    Args:
        np: Imported NumPy module.
        x_train (numpy.ndarray): Training rows giving each feature's standard deviation.
        row (numpy.ndarray): The case in original units.
        seed (int): Review seed; the generator is default_rng(seed + 9301).

    Returns:
        numpy.ndarray: 256 rows. Continuous features move by Gaussian noise of 0.25
        training standard deviations and are clipped to the schema bounds; each binary
        flag flips with probability 0.10. Every row is inside the schema."""
    rng = np.random.default_rng(seed + PLAUSIBLE_SEED_OFFSET)
    neighbourhood = np.tile(row, (PLAUSIBLE_NEIGHBOURS, 1)).astype(float)
    for index, feature in enumerate(FEATURES):
        if feature["binary"]:
            flips = rng.random(PLAUSIBLE_NEIGHBOURS) < 0.10
            neighbourhood[flips, index] = 1.0 - row[index]
        else:
            spread = float(x_train[:, index].std())
            neighbourhood[:, index] += rng.normal(0.0, 0.25 * spread, PLAUSIBLE_NEIGHBOURS)
            neighbourhood[:, index] = np.clip(neighbourhood[:, index], feature["minimum"], feature["maximum"])
    return neighbourhood


def _plausibility(np, samples) -> dict:
    """Measure how many perturbations leave the schema bounds.

    Args:
        np: Imported NumPy module.
        samples (numpy.ndarray): Perturbations in original units.

    Returns:
        dict: rows, the fraction outside the bounds per feature (binary: not 0 or 1),
        and the fraction of rows with at least one value outside."""
    outside_any = np.zeros(len(samples), dtype=bool)
    by_feature = {}
    for index, feature in enumerate(FEATURES):
        values = samples[:, index]
        if feature["binary"]:
            outside = ~np.isin(values, (0.0, 1.0))
        else:
            outside = (values < feature["minimum"]) | (values > feature["maximum"])
        outside_any |= outside
        by_feature[feature["key"]] = float(outside.mean())
    return {"rows": int(len(samples)), "outside_bounds_fraction_by_feature": by_feature,
            "outside_bounds_fraction_any": float(outside_any.mean())}


def feature_dependence(x) -> dict:
    """Describe the linear and rank dependence between the six features.

    Args:
        x: Training rows, shape (n, 6), in original units.

    Returns:
        dict: Pearson and Spearman matrices keyed by feature, the largest absolute
        off-diagonal value of each, and the variance inflation factor per feature
        (1 / (1 - R2) of an ordinary least-squares fit on the other five).

    Raises:
        ValueError: x is not a two-dimensional array with six columns and more rows
            than columns.

    Dependence on this synthetic sample says nothing about real invoices."""
    import numpy as np
    from scipy.stats import spearmanr
    data = np.asarray(x, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(FEATURES) or data.shape[0] <= len(FEATURES):
        raise ValueError("x must have six columns and more rows than columns")
    pearson = np.corrcoef(data, rowvar=False)
    spearman = np.asarray(spearmanr(data).statistic, dtype=float)
    off_diagonal = ~np.eye(len(FEATURES), dtype=bool)
    vif = {}
    for index, key in enumerate(FEATURE_KEYS):
        design = np.column_stack((np.ones(len(data)), np.delete(data, index, axis=1)))
        target = data[:, index]
        fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        r2 = 1.0 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
        vif[key] = float(1.0 / (1.0 - r2))
    return {
        "rows": int(len(data)),
        "pearson": {key: {other: float(pearson[i, j]) for j, other in enumerate(FEATURE_KEYS)} for i, key in enumerate(FEATURE_KEYS)},
        "spearman": {key: {other: float(spearman[i, j]) for j, other in enumerate(FEATURE_KEYS)} for i, key in enumerate(FEATURE_KEYS)},
        "max_abs_off_diagonal_pearson": float(np.abs(pearson[off_diagonal]).max()),
        "max_abs_off_diagonal_spearman": float(np.abs(spearman[off_diagonal]).max()),
        "variance_inflation_factor": vif,
    }


def _alternative_values(np, rows, x_train):
    """Choose one changed value per feature for the interaction check.

    Args:
        np: Imported NumPy module.
        rows (numpy.ndarray): Cases in original units.
        x_train (numpy.ndarray): Training rows giving the standard deviations.

    Returns:
        numpy.ndarray: Same shape as rows. Binary features are flipped; continuous
        features move up by one training standard deviation, or down when up would
        leave the schema bounds."""
    changed = rows.astype(float).copy()
    for index, feature in enumerate(FEATURES):
        if feature["binary"]:
            changed[:, index] = 1.0 - rows[:, index]
        else:
            step = float(x_train[:, index].std())
            up = rows[:, index] + step
            changed[:, index] = np.where(up <= feature["maximum"], up, rows[:, index] - step)
    return changed


def interaction_check(functions: dict, rows, x_train, tolerance: float) -> dict:
    """Measure pairwise interactions with mixed second differences.

    Args:
        functions (dict): 'log_odds' and 'probability' callables of one model.
        rows: Cases in original units, shape (n, 6).
        x_train: Training rows for the continuous step sizes.
        tolerance (float): Largest absolute second difference treated as no interaction.

    Returns:
        dict: Per scale, the maximum absolute f(x_ij) - f(x_i) - f(x_j) + f(x) over cases
        and the 15 feature pairs, the pair where it occurs, the number of pairs above the
        tolerance and the no-interaction verdict.

    An additive model in log-odds gives zero on that scale but not on probability,
    because the sigmoid couples the features."""
    import numpy as np
    base = np.asarray(rows, dtype=float)
    changed = _alternative_values(np, base, np.asarray(x_train, dtype=float))
    result = {}
    for scale in SCALES:
        function, largest, pairs_above, worst = functions[scale], 0.0, 0, None
        reference = function(base)
        for first, second in combinations(range(len(FEATURES)), 2):
            with_first, with_second = base.copy(), base.copy()
            with_first[:, first] = changed[:, first]
            with_second[:, second] = changed[:, second]
            with_both = with_first.copy()
            with_both[:, second] = changed[:, second]
            value = float(np.abs(function(with_both) - function(with_first) - function(with_second) + reference).max())
            pairs_above += value > tolerance
            if worst is None or value > largest:
                largest, worst = value, [FEATURE_KEYS[first], FEATURE_KEYS[second]]
        result[scale] = {"scale": scale, "max_abs_second_difference": largest, "largest_pair": worst,
                         "pairs_above_tolerance": int(pairs_above), "pairs": 15, "tolerance": tolerance,
                         "no_interaction": bool(largest <= tolerance)}
    return result


def _linear_model_report(deps, bundle, seed: int, resamples: int, row) -> dict:
    """Describe the interpretable model: scaling, coefficients, odds ratios and intervals.

    Args:
        deps (dict): Scientific namespace from the base bundle.
        bundle (dict): Base bundle with the fitted pipeline and training rows.
        seed (int): Review seed; resampling uses default_rng(seed + 4000).
        resamples (int): Number of refits on training rows drawn with replacement.
        row (numpy.ndarray): Reference input passed to _linear_coefficients.

    Returns:
        dict: Scaler means and scales, the original-unit intercept, and per feature the
        coefficient per training standard deviation and per original unit, odds ratios
        per unit and per stated increment, 95 percent bootstrap intervals, the known
        generating coefficient and whether it lies inside the interval.

    Refit errors propagate. The intervals describe resampling of 600 synthetic rows
    with L2 regularisation (C = 1), which shrinks coefficients towards zero."""
    np = deps["np"]
    from sklearn.base import clone
    fitted = _linear_coefficients(bundle, row)
    scaler = bundle["model"].steps[0][1]
    x, y = bundle["x_train"], bundle["y_train"]
    rng = np.random.default_rng(seed + COEFFICIENT_SEED_OFFSET)
    standardized, original, intercepts, skipped = [], [], [], 0
    for _ in range(resamples):
        rows = rng.integers(0, len(x), len(x))
        if np.unique(y[rows]).size < 2:
            skipped += 1
            continue
        refit = clone(bundle["model"]).fit(x[rows], y[rows])
        refit_scaler, refit_estimator = refit.steps[0][1], refit.steps[1][1]
        raw = refit_estimator.coef_[0] / refit_scaler.scale_
        standardized.append(refit_estimator.coef_[0])
        original.append(raw)
        intercepts.append(float(refit_estimator.intercept_[0] - np.dot(raw, refit_scaler.mean_)))
    standardized, original = np.asarray(standardized), np.asarray(original)

    def interval(values):
        """Return the 2.5 and 97.5 percentiles of bootstrap values.

        Args:
            values (numpy.ndarray): One coefficient's bootstrap draws.

        Returns:
            list: Two floats, lower then upper."""
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    features = []
    for index, item in enumerate(fitted["features"]):
        unit_interval = interval(original[:, index])
        increment = item["interpretation_increment"]
        generating = GENERATING_LOGIT[item["feature_key"]]
        features.append({
            "feature_key": item["feature_key"], "unit": item["unit"],
            "coefficient_per_training_standard_deviation": item["coefficient_per_training_standard_deviation"],
            "coefficient_per_training_standard_deviation_95_percent": interval(standardized[:, index]),
            "coefficient_per_original_unit": item["coefficient_per_original_unit"],
            "coefficient_per_original_unit_95_percent": unit_interval,
            "odds_ratio_per_original_unit": float(np.exp(item["coefficient_per_original_unit"])),
            "interpretation_increment": increment,
            "odds_ratio_per_increment": item["odds_multiplier_for_increment"],
            "odds_ratio_per_increment_95_percent": [float(np.exp(bound * increment)) for bound in unit_interval],
            "generating_coefficient_per_original_unit": generating,
            "generating_inside_95_percent_interval": bool(unit_interval[0] <= generating <= unit_interval[1]),
        })
    return {
        "preprocessing": {"method": "StandardScaler fitted on the training rows only",
                          "training_mean": {key: float(scaler.mean_[i]) for i, key in enumerate(FEATURE_KEYS)},
                          "training_scale": {key: float(scaler.scale_[i]) for i, key in enumerate(FEATURE_KEYS)}},
        "scale": "class-1 log-odds per unit; odds ratios are exp(coefficient x increment)",
        "intercept_original_units": fitted["intercept_original_units"],
        "intercept_original_units_95_percent": interval(np.asarray(intercepts)),
        "generating_intercept": GENERATING_LOGIT["intercept"],
        "features": features,
        "bootstrap_refits": int(len(original)), "bootstrap_skipped_single_class": skipped,
        "bootstrap_seed": seed + COEFFICIENT_SEED_OFFSET,
        "reading": "Each coefficient is the change in class-1 log-odds for one unit of that input with the other inputs held fixed. The model has no interaction terms, so the change is the same at every value of the other inputs on the log-odds scale. It is an association learned from synthetic data, not a causal effect.",
    }


def bounded_counterfactual(model, row, *, amount_scale: float, step: float = COUNTERFACTUAL_AMOUNT_STEP) -> dict:
    """Find and validate the cheapest bounded change that flips one model decision.

    Args:
        model: Fitted classifier with predict and predict_proba.
        row: One valid invoice in original units and schema order.
        amount_scale (float): Positive training standard deviation of amount_ratio, used
            to express amount changes in comparable units.
        step (float): Amount grid step, greater than 0 and at most 1.

    Returns:
        dict: found; when found, the original and new class and probability, the
        counterfactual input, the changed fields, the cost, and three validation flags:
        decision_flipped (re-predicted with the model), schema_valid (validate_invoice)
        and immutable_fields_unchanged. validated is true only when all three hold.

    Raises:
        ValueError: row is not a valid invoice, or amount_scale or step is invalid.

    Only purchase_order_match, delivery_confirmed and amount_ratio may change; supplier
    history, duplicate reference and bank change are immutable facts. The search is
    exhaustive over the four flag combinations and the amount grid 0.05 to 12.00 plus
    the original amount. Cost = flipped flags + |amount change| / amount_scale; ties go
    to fewer changed fields, then to the smaller amount change. A counterfactual
    describes the model's boundary; it is not advice on how to avoid a review."""
    import numpy as np
    original_row = np.asarray(validate_invoice(dict(zip(FEATURE_KEYS, (float(value) for value in row)))), dtype=float)
    if not (isinstance(amount_scale, float) and isfinite(amount_scale) and amount_scale > 0):
        raise ValueError("amount_scale must be a positive finite float")
    if not (isinstance(step, float) and 0 < step <= 1):
        raise ValueError("step must be a float in (0, 1]")
    amount, order_match, delivery = (FEATURE_KEYS.index(key) for key in ("amount_ratio", *MUTABLE_BINARY_FIELDS))
    original_class = int(model.predict(original_row.reshape(1, -1))[0])
    original_probability = float(model.predict_proba(original_row.reshape(1, -1))[0, 1])
    amount_feature = FEATURES[amount]
    grid = np.unique(np.concatenate((
        np.round(np.arange(amount_feature["minimum"], amount_feature["maximum"] + step / 2, step), 10),
        [original_row[amount]],
    )))
    grid = grid[(grid >= amount_feature["minimum"]) & (grid <= amount_feature["maximum"])]
    blocks = []
    for order_value in (0.0, 1.0):
        for delivery_value in (0.0, 1.0):
            block = np.tile(original_row, (grid.size, 1))
            block[:, order_match], block[:, delivery], block[:, amount] = order_value, delivery_value, grid
            blocks.append(block)
    candidates = np.vstack(blocks)
    flipped = candidates[model.predict(candidates) != original_class]
    base = {"found": False, "original_class": original_class, "original_probability": original_probability,
            "searched_candidates": int(len(candidates)),
            "mutable_fields": ["amount_ratio", *MUTABLE_BINARY_FIELDS], "immutable_fields": list(IMMUTABLE_FIELDS)}
    if not len(flipped):
        base["reason"] = "No bounded change of the mutable fields flips the decision."
        return base
    flag_changes = (flipped[:, order_match] != original_row[order_match]).astype(int) + (flipped[:, delivery] != original_row[delivery]).astype(int)
    amount_change = np.abs(flipped[:, amount] - original_row[amount])
    cost = flag_changes + amount_change / amount_scale
    changed_fields = flag_changes + (amount_change > 0).astype(int)
    best = flipped[np.lexsort((amount_change, changed_fields, cost))[0]]
    # Validate on the model and the schema, independently of the search's batch prediction.
    new_class = int(model.predict(best.reshape(1, -1))[0])
    counterfactual = {key: float(best[index]) for index, key in enumerate(FEATURE_KEYS)}
    try:
        validate_invoice(counterfactual)
        schema_valid, schema_error = True, None
    except ValueError as error:
        schema_valid, schema_error = False, str(error)
    immutable = all(best[FEATURE_KEYS.index(key)] == original_row[FEATURE_KEYS.index(key)] for key in IMMUTABLE_FIELDS)
    flipped_decision = new_class != original_class
    base.update({
        "found": True, "counterfactual": counterfactual,
        "changes": [{"feature_key": key, "from": float(original_row[index]), "to": float(best[index])}
                    for index, key in enumerate(FEATURE_KEYS) if best[index] != original_row[index]],
        "cost": float(np.min(cost)), "changed_fields": int(np.sum(best != original_row)),
        "counterfactual_class": new_class,
        "counterfactual_probability": float(model.predict_proba(best.reshape(1, -1))[0, 1]),
        "decision_flipped": flipped_decision, "schema_valid": schema_valid, "schema_error": schema_error,
        "immutable_fields_unchanged": bool(immutable),
        "validated": bool(flipped_decision and schema_valid and immutable),
    })
    return base


def _explain_case(deps, bundle, function, scale: str, row, shap_runs, median_run, index: int, background, *, seed: int, lime_seeds: int, lime_samples: int) -> dict:
    """Explain one case on one scale and run every per-case quality check.

    Args:
        deps (dict): Scientific namespace from the base bundle.
        bundle (dict): Base bundle with the training rows.
        function: Callable returning the explained model output.
        scale (str): 'log_odds' or 'probability'.
        row (numpy.ndarray): The case in original units.
        shap_runs (list): Per background, the (values, base, outputs) of _exact_shap
            over all explained rows; the first is the primary background.
        median_run (tuple): _exact_shap result with the single median baseline.
        index (int): Position of this case within the explained rows.
        background (numpy.ndarray): Primary background rows.
        seed (int): Review seed; LIME seeds are seed + 0 to seed + lime_seeds - 1.
        lime_seeds (int): Number of LIME runs.
        lime_samples (int): Perturbations per LIME run.

    Returns:
        dict: Model output, SHAP base/contributions and additivity, LIME coefficients,
        attributions, case error, fitting and held-out fidelity, plausible-neighbourhood
        fidelity, range diagnostic, plausibility and self-checks, stability over LIME
        seeds and SHAP backgrounds, background sensitivity and SHAP-LIME agreement,
        each with its scale and pass flag.

    Held-out fidelity for seed j uses the perturbations of seed j + 1 (cyclic), which
    were not used to fit seed j's surrogate. LIME errors propagate."""
    np = deps["np"]
    r2 = deps["r2_score"]
    limits = THRESHOLDS
    shap_values, shap_base, outputs = (item[index] for item in shap_runs[0])
    output = float(outputs)
    additivity_error = abs(float(shap_base) + float(shap_values.sum()) - output)
    runs = [_lime_explain(deps, bundle, function, row, scale, seed + offset, lime_samples) for offset in range(lime_seeds)]
    primary = runs[0]
    heldout_all = [_surrogate_metrics(np, r2, run, row, runs[(number + 1) % lime_seeds]["samples"][1:], function, scale)
                   for number, run in enumerate(runs)]
    fitting = _surrogate_metrics(np, r2, primary, row, primary["samples"], function, scale)
    plausible = _surrogate_metrics(np, r2, primary, row, _plausible_neighbours(np, bundle["x_train"], row, seed), function, scale)
    case_design = _lime_design(np, primary["explainer"], row, row.reshape(1, -1))[0]
    recomputed_local = primary["intercept"] + float(case_design @ primary["coefficients"])
    case_error = abs(primary["local_prediction"] - output)
    error_limit = limits["lime_case_abs_error_probability"] if scale == "probability" else limits["lime_case_abs_error_log_odds"]
    heldout = heldout_all[0]
    attributions = [_lime_attribution(np, run, row, background) for run in runs]
    lime_attr = attributions[0]
    shap_top = _top_k(shap_values)
    same_sign = [bool(np.sign(lime_attr[feature]) == np.sign(shap_values[feature])) for feature in sorted(shap_top)]
    agreement_spearman = _spearman(shap_values, lime_attr)
    median_values, median_base, _ = (item[index] for item in median_run)
    sensitivity_spearman = _spearman(shap_values, median_values)
    sensitivity_jaccard = _jaccard(shap_top, _top_k(median_values))
    lime_record = {
        "seed": seed, "scale": scale,
        "intercept": primary["intercept"],
        "coefficients_design_space": dict(zip(FEATURE_KEYS, (float(value) for value in primary["coefficients"]))),
        "attribution_vs_primary_background": dict(zip(FEATURE_KEYS, (float(value) for value in lime_attr))),
        "surrogate_at_case": primary["local_prediction"],
        "case_abs_error": case_error, "case_abs_error_threshold": error_limit,
        "case_error_passed": bool(case_error <= error_limit),
        "fit_weighted_r2": primary["fit_weighted_r2"],
        "heldout_neighbours": {**heldout, "drawn_by_lime_seed": seed + 1},
        "heldout_weighted_r2_threshold": limits["lime_heldout_weighted_r2_min"],
        "heldout_passed": bool(heldout["weighted_r2"] >= limits["lime_heldout_weighted_r2_min"]),
        "heldout_weighted_r2_all_seeds": {"min": float(min(item["weighted_r2"] for item in heldout_all)),
                                          "median": _median(item["weighted_r2"] for item in heldout_all),
                                          "max": float(max(item["weighted_r2"] for item in heldout_all))},
        "plausible_neighbourhood": plausible,
        "plausibility_of_fitting_perturbations": _plausibility(np, primary["samples"]),
        "self_check": {
            "local_prediction_abs_difference": abs(recomputed_local - primary["local_prediction"]),
            "fit_r2_abs_difference": abs(fitting["weighted_r2"] - primary["fit_weighted_r2"]),
        },
    }
    lime_record["fidelity_passed"] = lime_record["case_error_passed"] and lime_record["heldout_passed"]
    if scale == "probability":
        lime_record["range"] = {
            "surrogate_at_case_inside_unit_interval": bool(0.0 <= primary["local_prediction"] <= 1.0),
            "fitting_neighbours_outside_fraction": fitting["surrogate_outside_unit_interval_fraction"],
            "heldout_neighbours_outside_fraction": heldout["surrogate_outside_unit_interval_fraction"],
        }
    return {
        "scale": scale, "model_output": output,
        "shap": {
            "scale": scale, "base_value": float(shap_base),
            "contributions": dict(zip(FEATURE_KEYS, (float(value) for value in shap_values))),
            "additivity_abs_error": additivity_error,
            "additivity_passed": bool(additivity_error <= limits["additivity_abs_tolerance"]),
        },
        "lime": lime_record,
        "stability": {
            "lime_over_seeds": _pairwise_stability(attributions, limits["stability_top3_jaccard_min"], limits["stability_spearman_min"]),
            "shap_over_backgrounds": _pairwise_stability([run[0][index] for run in shap_runs], limits["stability_top3_jaccard_min"], limits["stability_spearman_min"]),
        },
        "background_sensitivity": {
            "scale": scale,
            "median_baseline_base_value": float(median_base),
            "base_value_difference": float(median_base - shap_base),
            "median_baseline_contributions": dict(zip(FEATURE_KEYS, (float(value) for value in median_values))),
            "max_abs_contribution_difference": float(np.abs(median_values - shap_values).max()),
            "top3_jaccard": sensitivity_jaccard, "spearman_abs": sensitivity_spearman,
            "insensitive": bool(sensitivity_jaccard >= limits["background_top3_jaccard_min"]
                                and sensitivity_spearman is not None
                                and sensitivity_spearman >= limits["background_spearman_min"]),
        },
        "agreement": {
            "scale": scale,
            "spearman_abs_shap_vs_lime_attribution": agreement_spearman,
            "spearman_abs_shap_vs_raw_lime_coefficients": _spearman(shap_values, primary["coefficients"]),
            "top3_jaccard": _jaccard(shap_top, _top_k(lime_attr)),
            "shap_top3": [FEATURE_KEYS[feature] for feature in sorted(shap_top)],
            "lime_top3": [FEATURE_KEYS[feature] for feature in sorted(_top_k(lime_attr))],
            "top3_sign_agreement_fraction": float(np.mean(same_sign)),
            "agrees": bool(agreement_spearman is not None and agreement_spearman >= limits["agreement_spearman_min"]
                           and np.mean(same_sign) >= limits["agreement_top3_sign_fraction_min"]),
        },
        "_lime_samples_sha256": hashlib.sha256(primary["samples"].astype("<f8").tobytes()).hexdigest(),
    }


def _verdict(passed: int, evaluated: int) -> str:
    """Return 'pass' only when every evaluated case passed.

    Args:
        passed (int): Number of passing cases.
        evaluated (int): Number of evaluated cases.

    Returns:
        str: 'pass' or 'fail'. Zero evaluated cases is a failure, not a vacuous pass."""
    return "pass" if evaluated and passed == evaluated else "fail"


def _summarize(np, records: list, scale: str, crosscheck: dict) -> dict:
    """Aggregate the per-case checks of one model and scale.

    Args:
        np: Imported NumPy module.
        records (list): _explain_case results for the selected cases (not the reference).
        scale (str): Scale shared by the records.
        crosscheck (dict): Implementation cross-check results for this model and scale.

    Returns:
        dict: One entry per protocol check (C1, C1b, C2/C3, C4, C5, C6, C7) with values,
        thresholds, passed and evaluated case counts and verdicts, each with its scale."""
    count = len(records)
    lime = [record["lime"] for record in records]
    summary = {
        "scale": scale, "evaluated_cases": count,
        "C1_additivity_shap": {
            "max_abs_error": float(max(record["shap"]["additivity_abs_error"] for record in records)),
            "tolerance": THRESHOLDS["additivity_abs_tolerance"],
            "passed_cases": sum(record["shap"]["additivity_passed"] for record in records),
            "lime": "Not a design property of LIME; the surrogate's distance to the model output is C2.",
        },
        "C1b_implementation_crosscheck_shap": crosscheck,
        "C2_C3_local_fidelity_lime": {
            "case_abs_error_max": float(max(item["case_abs_error"] for item in lime)),
            "case_abs_error_median": _median(item["case_abs_error"] for item in lime),
            "case_abs_error_threshold": lime[0]["case_abs_error_threshold"],
            "case_error_passed_cases": sum(item["case_error_passed"] for item in lime),
            "fit_weighted_r2_min": float(min(item["fit_weighted_r2"] for item in lime)),
            "fit_weighted_r2_median": _median(item["fit_weighted_r2"] for item in lime),
            "heldout_weighted_r2_min": float(min(item["heldout_neighbours"]["weighted_r2"] for item in lime)),
            "heldout_weighted_r2_median": _median(item["heldout_neighbours"]["weighted_r2"] for item in lime),
            "heldout_weighted_r2_threshold": THRESHOLDS["lime_heldout_weighted_r2_min"],
            "heldout_passed_cases": sum(item["heldout_passed"] for item in lime),
            "plausible_weighted_r2_min": float(min(item["plausible_neighbourhood"]["weighted_r2"] for item in lime)),
            "plausible_weighted_r2_median": _median(item["plausible_neighbourhood"]["weighted_r2"] for item in lime),
            "passed_cases": sum(item["fidelity_passed"] for item in lime),
            "self_check_max_local_prediction_difference": float(max(item["self_check"]["local_prediction_abs_difference"] for item in lime)),
            "self_check_max_fit_r2_difference": float(max(item["self_check"]["fit_r2_abs_difference"] for item in lime)),
        },
    }
    summary["C1_additivity_shap"]["verdict"] = _verdict(summary["C1_additivity_shap"]["passed_cases"], count)
    fidelity = summary["C2_C3_local_fidelity_lime"]
    fidelity["verdict"] = _verdict(fidelity["passed_cases"], count)
    if scale == "probability":
        inside = sum(item["range"]["surrogate_at_case_inside_unit_interval"] for item in lime)
        summary["C4_probability_range_lime"] = {
            "cases_inside_unit_interval": inside,
            "mean_fitting_neighbours_outside_fraction": float(np.mean([item["range"]["fitting_neighbours_outside_fraction"] for item in lime])),
            "max_fitting_neighbours_outside_fraction": float(max(item["range"]["fitting_neighbours_outside_fraction"] for item in lime)),
            "mean_heldout_neighbours_outside_fraction": float(np.mean([item["range"]["heldout_neighbours_outside_fraction"] for item in lime])),
            "verdict": _verdict(inside, count),
        }
    else:
        summary["C4_probability_range_lime"] = {"verdict": "not applicable", "reason": "Log-odds are unbounded."}
    for name, key in (("lime", "lime_over_seeds"), ("shap", "shap_over_backgrounds")):
        stability = [record["stability"][key] for record in records]
        passed = sum(item["passed"] for item in stability)
        summary.setdefault("C5_stability", {})[name] = {
            "runs_per_case": stability[0]["runs"],
            "median_mean_top3_jaccard": _median(item["mean_top3_jaccard"] for item in stability),
            "min_mean_top3_jaccard": float(min(item["mean_top3_jaccard"] for item in stability)),
            "median_mean_spearman_abs": _median(item["mean_spearman_abs"] for item in stability),
            "min_mean_spearman_abs": _minimum(item["mean_spearman_abs"] for item in stability),
            "cases_with_sign_change": sum(bool(item["features_with_sign_change"]) for item in stability),
            "passed_cases": passed, "verdict": _verdict(passed, count),
        }
    sensitivity = [record["background_sensitivity"] for record in records]
    insensitive = sum(item["insensitive"] for item in sensitivity)
    summary["C6_background_sensitivity_shap"] = {
        "median_top3_jaccard": _median(item["top3_jaccard"] for item in sensitivity),
        "median_spearman_abs": _median(item["spearman_abs"] for item in sensitivity),
        "max_abs_contribution_difference": float(max(item["max_abs_contribution_difference"] for item in sensitivity)),
        "base_value_difference": float(sensitivity[0]["base_value_difference"]),
        "insensitive_cases": insensitive, "verdict": _verdict(insensitive, count),
    }
    agreement = [record["agreement"] for record in records]
    agrees = sum(item["agrees"] for item in agreement)
    summary["C7_shap_lime_agreement"] = {
        "median_spearman_abs": _median(item["spearman_abs_shap_vs_lime_attribution"] for item in agreement),
        "min_spearman_abs": _minimum(item["spearman_abs_shap_vs_lime_attribution"] for item in agreement),
        "median_spearman_abs_raw_coefficients": _median(item["spearman_abs_shap_vs_raw_lime_coefficients"] for item in agreement),
        "median_top3_jaccard": _median(item["top3_jaccard"] for item in agreement),
        "mean_top3_sign_agreement_fraction": float(np.mean([item["top3_sign_agreement_fraction"] for item in agreement])),
        "agreeing_cases": agrees, "verdict": _verdict(agrees, count),
    }
    return summary


def _check_table(models: dict) -> list:
    """Flatten every model, scale and check verdict into one readable table.

    Args:
        models (dict): Per model id, the 'scales' mapping of _summarize results.

    Returns:
        list: One dict per check with check id, quality, method, model id, scale, the
        headline value and its name, the threshold, passed and evaluated counts, and
        the verdict ('pass', 'fail' or 'not applicable')."""
    table = []
    for model_id, model in models.items():
        for scale, summary in model["scales"].items():
            count = summary["evaluated_cases"]

            def add(check, quality, method, value_name, value, threshold, passed, verdict):
                """Append one row to the check table.

                Args:
                    check (str): Protocol check id.
                    quality (str): The quality the check measures.
                    method (str): 'SHAP' or 'LIME'.
                    value_name (str): Name of the headline value.
                    value: Headline value on the row's scale, or None.
                    threshold: Pass threshold, or None.
                    passed: Number of passing cases, or None.
                    verdict (str): 'pass', 'fail' or 'not applicable'."""
                table.append({"check": check, "quality": quality, "method": method, "model_id": model_id,
                              "scale": scale, "value_name": value_name, "value": value, "threshold": threshold,
                              "passed_cases": passed, "evaluated_cases": count, "verdict": verdict})

            item = summary["C1_additivity_shap"]
            add("C1", "additivity", "SHAP", "max |base + sum - output|", item["max_abs_error"], item["tolerance"], item["passed_cases"], item["verdict"])
            add("C1", "additivity", "LIME", None, None, None, None, "not applicable")
            item = summary["C1b_implementation_crosscheck_shap"]
            add("C1b", "implementation agreement", "SHAP", "max |Exact - enumeration|", item["exact_vs_enumeration_max_abs_difference"], item["tolerance"], None, item["exact_verdict"])
            if item["model_specific_vs_enumeration_max_abs_difference"] is not None:
                add("C1b", "implementation agreement", "SHAP", "max |" + item["model_specific_method"] + " - enumeration|", item["model_specific_vs_enumeration_max_abs_difference"], item["tolerance"], None, item["model_specific_verdict"])
            item = summary["C2_C3_local_fidelity_lime"]
            add("C2", "local fidelity at the case", "LIME", "max |surrogate - output| at the case", item["case_abs_error_max"], item["case_abs_error_threshold"], item["case_error_passed_cases"], _verdict(item["case_error_passed_cases"], count))
            add("C3", "local fidelity on fresh neighbours", "LIME", "min held-out weighted R2", item["heldout_weighted_r2_min"], item["heldout_weighted_r2_threshold"], item["heldout_passed_cases"], _verdict(item["heldout_passed_cases"], count))
            add("C2+C3", "local fidelity", "LIME", "cases passing both", item["passed_cases"], None, item["passed_cases"], item["verdict"])
            item = summary["C4_probability_range_lime"]
            add("C4", "probability range diagnostic", "LIME", "cases with surrogate inside [0, 1]", item.get("cases_inside_unit_interval"), None, item.get("cases_inside_unit_interval"), item["verdict"])
            for method in ("lime", "shap"):
                item = summary["C5_stability"][method]
                add("C5", "stability", method.upper(), "median mean top-3 Jaccard", item["median_mean_top3_jaccard"], THRESHOLDS["stability_top3_jaccard_min"], item["passed_cases"], item["verdict"])
            item = summary["C6_background_sensitivity_shap"]
            add("C6", "reference sensitivity (training sample vs median baseline)", "SHAP", "median top-3 Jaccard", item["median_top3_jaccard"], THRESHOLDS["background_top3_jaccard_min"], item["insensitive_cases"], item["verdict"])
            item = summary["C7_shap_lime_agreement"]
            add("C7", "SHAP-LIME agreement", "SHAP and LIME", "median Spearman of |attributions|", item["median_spearman_abs"], THRESHOLDS["agreement_spearman_min"], item["agreeing_cases"], item["verdict"])
    return table


def _package_versions() -> dict:
    """Return installed versions of the scientific distributions used here.

    Returns:
        dict: Distribution name mapped to its version, or None when not installed.

    Reads installed metadata only; nothing is imported or installed."""
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "shap", "lime", "threadpoolctl", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_explainability_review(
    *, seed: int = 1729, n_samples: int = 800, lime_samples: int = 4000,
    cases_per_stratum: int = 5, lime_seeds: int = 10, shap_backgrounds: int = 5,
    background_rows: int = 100, bootstrap_resamples: int = 1000,
    coefficient_bootstrap_resamples: int = 200,
) -> dict:
    """Run the complete pre-registered explainability review on both models.

    Args:
        seed (int): Shared generation, split, fitting and review seed.
        n_samples (int): Total synthetic rows; 25 percent are held out.
        lime_samples (int): LIME perturbations per explanation.
        cases_per_stratum (int): Selected test cases per stratum A-D.
        lime_seeds (int): LIME random states per case (stability and held-out fidelity).
        shap_backgrounds (int): Training background samples per case (stability).
        background_rows (int): Rows per SHAP background.
        bootstrap_resamples (int): Test-set bootstrap resamples for metric intervals.
        coefficient_bootstrap_resamples (int): Logistic refits for coefficient intervals.

    Returns:
        dict: Finite-JSON-compatible report with parameters, thresholds, environment,
        data and split, performance and calibration per model, the selected cases, per
        model and scale the per-case records and check summaries, the reference case,
        the flattened check table, assumptions and data dependence, the interpretable
        model, interaction checks, counterfactuals, what is not measured, limitations,
        durations, all_checks_passed and production_approval=False.

    Raises:
        ValueError: A budget is outside its bounds.
        ExplainabilityUnavailable: Optional scientific distributions are missing.

    Other scientific errors propagate. CPU only with one numerical thread; no file,
    network or GPU use. Failed checks are recorded, never hidden or retried."""
    started = time.perf_counter()
    parameters = review_parameters(
        seed=seed, n_samples=n_samples, lime_samples=lime_samples, cases_per_stratum=cases_per_stratum,
        lime_seeds=lime_seeds, shap_backgrounds=shap_backgrounds, background_rows=background_rows,
        bootstrap_resamples=bootstrap_resamples, coefficient_bootstrap_resamples=coefficient_bootstrap_resamples,
    )
    bundle = _fit_synthetic_model(seed, n_samples)
    deps = bundle["deps"]
    np, shap = deps["np"], deps["shap"]
    tree, tree_parameters, tree_metrics = _fit_synthetic_tree_model(seed, n_samples)
    models = {"logistic_regression": bundle["model"], "hist_gradient_boosting": tree}
    x_train, x_test, y_test = bundle["x_train"], bundle["x_test"], bundle["y_test"]
    timings = {}
    with deps["threadpool_limits"](limits=1):
        section = time.perf_counter()
        predictions = {model_id: model.predict(x_test).astype(int) for model_id, model in models.items()}
        probabilities = {model_id: model.predict_proba(x_test)[:, 1] for model_id, model in models.items()}
        logit_check = {model_id: float(np.abs(1.0 / (1.0 + np.exp(-model.decision_function(x_test))) - probabilities[model_id]).max())
                       for model_id, model in models.items()}
        performance = {}
        for model_id in models:
            performance[model_id] = _performance(deps, y_test, probabilities[model_id], predictions[model_id], seed=seed, resamples=bootstrap_resamples)
            performance[model_id]["model"] = MODEL_LABELS[model_id]
            performance[model_id]["sigmoid_of_log_odds_vs_probability_max_abs_difference"] = logit_check[model_id]
        base_metrics = {"logistic_regression": bundle["test_metrics"], "hist_gradient_boosting": tree_metrics}
        for model_id, metrics in base_metrics.items():
            performance[model_id]["matches_base_helper_metrics"] = all(
                abs(performance[model_id]["point"][name] - metrics[name]) <= 1e-12 for name in ("accuracy", "roc_auc", "brier_score"))
        timings["performance_and_calibration"] = time.perf_counter() - section

        selection = select_review_cases(y_test, predictions["logistic_regression"], predictions["hist_gradient_boosting"], seed=seed, per_stratum=cases_per_stratum)
        positions = np.asarray(selection["positions"], dtype=int)
        reference_row = np.asarray(validate_invoice(REFERENCE_INVOICE), dtype=float)
        rows = np.vstack((x_test[positions], reference_row))
        backgrounds = [x_train[np.random.default_rng(seed + offset).choice(len(x_train), background_rows, replace=False)]
                       for offset in parameters["background_seed_offsets"]]
        median_row = np.median(x_train, axis=0).reshape(1, -1)
        cases = []
        for number, case in enumerate(selection["cases"]):
            position = case["position"]
            cases.append({
                "case_number": number, "position": position, "stratum": case["stratum"],
                "synthetic_record_index": int(bundle["test_ids"][position]), "true_label": int(y_test[position]),
                "features": dict(zip(FEATURE_KEYS, (float(value) for value in x_test[position]))),
                "models": {model_id: {"probability": float(probabilities[model_id][position]),
                                      "predicted_class": int(predictions[model_id][position]),
                                      "correct": bool(predictions[model_id][position] == y_test[position])}
                           for model_id in models},
            })

        results, sample_hashes = {}, {}
        for model_id, model in models.items():
            section = time.perf_counter()
            functions = _output_functions(model)
            scales = {}
            for scale in SCALES:
                function = functions[scale]
                shap_runs = [_exact_shap(shap, np, function, background, rows) for background in backgrounds]
                median_run = _exact_shap(shap, np, function, median_row, rows)
                enumerated = np.array([_enumerated_shap(np, function, backgrounds[0], row)[0] for row in rows])
                specific, specific_base, specific_method = _model_specific_shap(shap, np, model_id, model, backgrounds[0], rows, scale)
                tolerance = THRESHOLDS["implementation_crosscheck_abs_tolerance"]
                exact_gap = float(np.abs(shap_runs[0][0] - enumerated).max())
                specific_gap = None if specific is None else float(np.abs(specific - enumerated).max())
                crosscheck = {
                    "scale": scale, "rows": int(len(rows)), "tolerance": tolerance,
                    "exact_vs_enumeration_max_abs_difference": exact_gap,
                    "exact_verdict": "pass" if exact_gap <= tolerance else "fail",
                    "model_specific_method": specific_method,
                    "model_specific_vs_enumeration_max_abs_difference": specific_gap,
                    "model_specific_verdict": "not applicable" if specific_gap is None else ("pass" if specific_gap <= tolerance else "fail"),
                    "model_specific_additivity_max_abs_error": None if specific is None else float(np.abs(specific_base + specific.sum(axis=1) - shap_runs[0][2]).max()),
                    "model_specific_base_vs_exact_base_max_abs_difference": None if specific is None else float(np.abs(specific_base - shap_runs[0][1]).max()),
                }
                records = [_explain_case(deps, bundle, function, scale, row, shap_runs, median_run, index, backgrounds[0],
                                         seed=seed, lime_seeds=lime_seeds, lime_samples=lime_samples)
                           for index, row in enumerate(rows)]
                for index, record in enumerate(records):
                    sample_hashes.setdefault(index, set()).add(record.pop("_lime_samples_sha256"))
                    label = cases[index] if index < len(cases) else {"position": None, "stratum": "reference"}
                    record.update(position=label["position"], stratum=label["stratum"])
                scales[scale] = {"label": SCALE_LABELS[scale], "summary": _summarize(np, records[:-1], scale, crosscheck),
                                 "cases": records[:-1], "reference_case": records[-1]}
            results[model_id] = {"model": MODEL_LABELS[model_id], "scales": scales}
            if model_id == "hist_gradient_boosting":
                results[model_id]["tree_shap_diagnostic"] = _tree_shap_diagnostic(shap, np, model, functions, backgrounds[0], rows)
            timings["explanations_" + model_id] = time.perf_counter() - section

        section = time.perf_counter()
        check_models = {model_id: {"scales": {scale: results[model_id]["scales"][scale]["summary"] for scale in SCALES}} for model_id in models}
        table = _check_table(check_models)
        dependence = feature_dependence(x_train)
        interpretable = _linear_model_report(deps, bundle, seed, coefficient_bootstrap_resamples, reference_row)
        interactions = {model_id: interaction_check(_output_functions(model), x_test[positions], x_train, THRESHOLDS["no_interaction_abs_tolerance"])
                        for model_id, model in models.items()}
        amount_scale = float(x_train[:, FEATURE_KEYS.index("amount_ratio")].std())
        counterfactuals = {}
        for model_id, model in models.items():
            selected = [{"position": case["position"], "stratum": case["stratum"], **bounded_counterfactual(model, x_test[case["position"]], amount_scale=amount_scale)}
                        for case in cases]
            reference = bounded_counterfactual(model, reference_row, amount_scale=amount_scale)
            found = [item for item in selected if item["found"]]
            counterfactuals[model_id] = {
                "summary": {
                    "cases": len(selected), "found": len(found),
                    "validated": sum(item["validated"] for item in found),
                    "decision_flipped": sum(item["decision_flipped"] for item in found),
                    "schema_valid": sum(item["schema_valid"] for item in found),
                    "immutable_fields_unchanged": sum(item["immutable_fields_unchanged"] for item in found),
                    "median_cost": _median(item["cost"] for item in found),
                    "changed_field_counts": {key: sum(any(change["feature_key"] == key for change in item["changes"]) for item in found)
                                             for key in ("amount_ratio", *MUTABLE_BINARY_FIELDS)},
                    "verdict": "not applicable" if not found else ("pass" if all(item["validated"] for item in found) else "fail"),
                },
                "cases": selected, "reference_case": reference,
            }
        timings["model_and_counterfactual_reports"] = time.perf_counter() - section

    plausibility = [results["logistic_regression"]["scales"]["probability"]["cases"][index]["lime"]["plausibility_of_fitting_perturbations"]
                    for index in range(len(cases))]
    verdicts = [row["verdict"] for row in table if row["verdict"] != "not applicable"]
    verdicts += [counterfactuals[model_id]["summary"]["verdict"] for model_id in models
                 if counterfactuals[model_id]["summary"]["verdict"] != "not applicable"]
    timings["total"] = time.perf_counter() - started
    return {
        "ok": True,
        "scenario": "Pre-registered review of SHAP and LIME on two synthetic invoice classifiers, same split, same cases, same outputs",
        "intended_use": "Engineering study only. Scores support review and are not proof of wrongdoing.",
        "production_approval": False,
        "parameters": parameters, "thresholds": dict(THRESHOLDS),
        "environment": {
            "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "packages": _package_versions(), "device": "CPU", "gpu_used": False, "numerical_threads": 1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "data": {
            "origin": "Entirely synthetic fictional invoices with a known label-generating logit",
            "dataset_sha256": bundle["fingerprint"], "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
            "train_test_index_overlap": int(np.intersect1d(bundle["train_ids"], bundle["test_ids"]).size),
            "train_positive_rate": float(bundle["y_train"].mean()), "test_positive_rate": float(y_test.mean()),
            "generating_logit_per_original_unit": dict(GENERATING_LOGIT),
        },
        "models": {
            "logistic_regression": {"model": MODEL_LABELS["logistic_regression"], "parameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}},
            "hist_gradient_boosting": {"model": MODEL_LABELS["hist_gradient_boosting"], "parameters": tree_parameters},
        },
        "scales": dict(SCALE_LABELS),
        "performance_and_calibration": performance,
        "case_selection": {**selection, "selected_cases": cases, "reference_case": {
            "features": dict(REFERENCE_INVOICE), "ground_truth_label": None,
            "role": "Fixed fictional high-score input of 27 October; explained with the same methods, excluded from every aggregate."}},
        "explanations": results,
        "check_table": table,
        "assumptions_and_data": {
            "feature_dependence_training_rows": dependence,
            "interventional_shap": "Masked features take background values independently of the kept features. With dependent features the hybrid rows can be impossible combinations. The generator draws the six features independently, so the assumption matches this synthetic data; real invoices would need their dependence measured first.",
            "hybrid_rows_inside_schema": bool(all(
                feature["minimum"] <= values.min() and values.max() <= feature["maximum"]
                for index, feature in enumerate(FEATURES) for values in (rows[:, index], *(background[:, index] for background in backgrounds), median_row[:, index]))),
            "lime_neighbourhood": {
                "definition": "Continuous features: Gaussian noise with the training standard deviation around the case, not clipped. Binary features: resampled from the training frequencies, independently of the case. Weights: sqrt(exp(-d^2 / w^2)), d the Euclidean distance in LIME's design space.",
                "kernel_width": KERNEL_WIDTH,
                "same_perturbations_for_both_models_and_scales": bool(all(len(hashes) == 1 for hashes in sample_hashes.values())),
                "fitting_perturbations_outside_bounds_primary_seed": {
                    "cases": len(plausibility),
                    "mean_fraction_any": float(np.mean([item["outside_bounds_fraction_any"] for item in plausibility])),
                    "max_fraction_any": float(max(item["outside_bounds_fraction_any"] for item in plausibility)),
                    "mean_fraction_by_feature": {key: float(np.mean([item["outside_bounds_fraction_by_feature"][key] for item in plausibility])) for key in FEATURE_KEYS},
                },
            },
        },
        "interpretable_model": {**interpretable, "interactions": interactions["logistic_regression"],
                                "correlation_limit": "Coefficients hold the other inputs fixed. With correlated inputs that combination can be unrealistic and the coefficients become unstable; the correlation matrix and variance inflation factors measure how far that applies to these synthetic rows."},
        "interactions": interactions,
        "counterfactuals": counterfactuals,
        "not_measured": {
            "reviewer_usefulness": {"measured": False, "reason": "Requires people performing a task; no participant data exist.", "proposed_study": dict(REVIEWER_STUDY)},
            "causality": {"measured": False, "reason": "SHAP, LIME, coefficients and counterfactuals describe the fitted model, not the world. No causal claim is made."},
        },
        "all_checks_passed": all(verdict == "pass" for verdict in verdicts),
        "timing_seconds": timings,
        "limitations": [
            "Synthetic data, six numeric features and one 600/200 split do not establish usefulness on real invoices.",
            "The label rule is linear, so the logistic model is correctly specified here; this is not a neutral comparison of model families.",
            "Twenty cases selected by a stratified rule describe these cases, not every possible input.",
            "Thresholds are illustrative values fixed before the run, not business, legal or production criteria.",
            "Interventional SHAP assumes independent features; it matches this generator but must be checked on real data.",
            "LIME's continuous perturbations are not clipped and can leave the schema bounds; its binary features are resampled from training frequencies.",
            "SHAP-LIME agreement compares two explanations with each other, not with a known true attribution.",
            "Counterfactuals describe the model's decision boundary, not advice or causal effects.",
            "Reviewer usefulness and causality are not measured.",
            "Durations describe this local CPU run, not production performance.",
        ],
    }
