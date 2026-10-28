"""Run bounded, fictional invoice experiments with real SHAP and LIME libraries.

The six FEATURES define numeric ratios, days and binary flags, not an invoice
upload or a fraud-detection service. Public helpers validate caller inputs,
fit reproducible synthetic classifiers on a stratified 75/25 train/test split,
and return in-memory dictionaries suitable for finite JSON serialization.
Scientific packages are imported lazily; importing this module uses only the
standard library and does not install packages, read models, call a network
service or write files. Requested runs perform CPU fitting/explanation and keep
up to four fitted bundles per cached helper in process memory.

SHAP contributions reconstruct class-1 log-odds within a numerical tolerance.
LIME approximates class-1 probability and can fail or leave the [0, 1] range.
Neither successful execution nor numerical reconstruction proves causality,
real-invoice usefulness, fairness, legal compliance or production readiness."""
from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from math import isfinite
from numbers import Real
import hashlib
import platform
import time
from typing import Mapping


FEATURES = (
    {"key": "amount_ratio", "label": "Amount versus supplier median", "unit": "ratio", "minimum": 0.05, "maximum": 12.0, "binary": False},
    {"key": "purchase_order_match", "label": "Purchase order matched", "unit": "0 or 1", "minimum": 0, "maximum": 1, "binary": True},
    {"key": "supplier_history_days", "label": "Supplier history", "unit": "days", "minimum": 0, "maximum": 3650, "binary": False},
    {"key": "duplicate_reference", "label": "Repeated invoice reference", "unit": "0 or 1", "minimum": 0, "maximum": 1, "binary": True},
    {"key": "delivery_confirmed", "label": "Delivery confirmation recorded", "unit": "0 or 1", "minimum": 0, "maximum": 1, "binary": True},
    {"key": "recent_bank_change", "label": "Recent bank-detail change", "unit": "0 or 1", "minimum": 0, "maximum": 1, "binary": True},
)
CLASS_NAMES = ("routine review", "review suggested")
REQUIRED_DISTRIBUTIONS = ("numpy", "scipy", "scikit-learn", "shap", "lime")


class ExplainabilityUnavailable(RuntimeError):
    """Signal that optional explainability distributions are unavailable.

    Raised by _dependencies when installed-package metadata does not include every
    required scientific distribution. This RuntimeError subclass carries the human
    readable installation guidance in its normal exception message; it has no extra
    attributes and does not install or repair packages. Import or ABI failures after
    the metadata check can still propagate separately."""


def invoice_schema() -> list[dict]:
    """Return independent copies of the six demonstration field definitions.

    Returns:
        list[dict]: Ordered field metadata with key, label, unit, inclusive minimum
        and maximum, and a binary flag. Ratios/days are numeric; binary fields use
        0 or 1. Changing a returned dictionary does not change the module's schema.

    This helper has no arguments or I/O and does not import scientific packages."""
    return [dict(feature) for feature in FEATURES]


def explainability_status() -> dict:
    """Inspect distribution metadata for the optional local experiment.

    Returns:
        dict: available, missing_dependencies, package_versions, the CPU exercise
        description, fields from invoice_schema, and an intended-use warning.

    The check reads installed-distribution metadata only; it does not import SHAP,
    LIME or NumPy, train a model, or modify the environment. available confirms that
    metadata exists, not that every package can import or is ABI-compatible. Other
    metadata-access errors are not swallowed."""
    versions, missing = {}, []
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return {
        "available": not missing,
        "missing_dependencies": missing,
        "package_versions": versions,
        "device": "CPU for this synthetic logistic-regression exercise",
        "fields": invoice_schema(),
        "intended_use": "Synthetic engineering study only. The score is a review aid, not proof of wrongdoing.",
    }


def validate_invoice(invoice: Mapping[str, Real]) -> list[float]:
    """Validate one synthetic input and order its values for the classifiers.

    Args:
        invoice (Mapping[str, Real]): Exactly the six FEATURES keys. Values must be
            finite and within each inclusive schema bound. Binary fields accept
            numeric 0/1, including booleans; booleans are rejected for other fields.
            Supplier history is expressed in days; amount_ratio is unitless.

    Returns:
        list[float]: Values in FEATURES order, independent of mapping key order.

    Raises:
        ValueError: The mapping, keys, numeric types, finite values, ranges or binary
            domains violate the demonstration schema.

    No model is loaded and the input mapping is not mutated. The field bounds are
    teaching constraints, not validated business rules or evidence of wrongdoing."""
    if not isinstance(invoice, Mapping):
        raise ValueError("invoice must be a mapping of the six numeric demonstration fields")
    expected = {feature["key"] for feature in FEATURES}
    if set(invoice) != expected:
        missing, extra = sorted(expected - set(invoice)), sorted(set(invoice) - expected)
        raise ValueError(f"invoice fields differ from the schema: missing={missing}, extra={extra}")
    values = []
    for feature in FEATURES:
        value = invoice[feature["key"]]
        if not isinstance(value, Real) or (isinstance(value, bool) and not feature["binary"]):
            raise ValueError(f"{feature['key']} must be a finite number")
        number = float(value)
        if not isfinite(number) or not feature["minimum"] <= number <= feature["maximum"]:
            raise ValueError(f"{feature['key']} must be within [{feature['minimum']}, {feature['maximum']}]")
        if feature["binary"] and number not in (0.0, 1.0):
            raise ValueError(f"{feature['key']} must be 0 or 1")
        values.append(number)
    return values


def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> int:
    """Validate and return an integer resource or reproducibility parameter.

    Args:
        name (str): Parameter name used in the error message.
        value (int): Candidate integer; bool is explicitly rejected.
        minimum (int): Inclusive lower bound.
        maximum (int): Inclusive upper bound.

    Returns:
        int: The original value when it is within the inclusive bounds.

    Raises:
        ValueError: value is boolean, non-integer or outside the stated bounds.

    This pure helper does not convert strings or floating-point numbers."""
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within [{minimum}, {maximum}]")
    return value


def _dependencies():
    """Lazily load the scientific implementations used by requested runs.

    Returns:
        dict: The local namespace containing np, shap, LimeTabularExplainer,
        scikit-learn estimators/preprocessing/split and metric callables,
        threadpool_limits, and the metadata status. Callers use these shared objects
        without importing the scientific stack at module-import time.

    Raises:
        ExplainabilityUnavailable: At least one required distribution is missing.
        ImportError: An installed dependency or one of its native dependencies fails
            to import; other dependency initialization errors may also propagate.

    Imports can initialize library state in this process. No packages are installed,
    no model is downloaded, and no external service or file output is created."""
    status = explainability_status()
    if not status["available"]:
        raise ExplainabilityUnavailable(
            "Optional explainability dependencies are missing: "
            + ", ".join(status["missing_dependencies"])
            + ". Use a separate environment with requirements-explainability.txt."
        )
    import numpy as np
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    return locals()


@lru_cache(maxsize=4)
def _fit_synthetic_model(seed: int, n_samples: int):
    """Generate one synthetic dataset and fit the cached linear baseline.

    Args:
        seed (int): NumPy/scikit-learn seed, validated by the public entry point.
        n_samples (int): Total generated rows, of which 25 percent are held out.

    Returns:
        dict: Scientific dependencies, the StandardScaler/LogisticRegression
        pipeline, train/test arrays and original row indices, a SHA-256 dataset
        fingerprint, and held-out accuracy, ROC AUC and Brier score.

    Feature distributions and the probabilistic label-generating logit are defined
    here. The stratified split separates training from evaluation; scaling is fitted
    only on training rows. Fitting and prediction use a one-thread numerical limit.
    The lru_cache retains up to four mutable bundles: treat returned arrays/models as
    read-only. Dependency, split or fitting errors propagate. There is no file/network
    I/O. Reproducibility assumes compatible library versions; the deliberately linear
    label rule makes this a teaching example, not a neutral model-family benchmark."""
    deps = _dependencies()
    np = deps["np"]
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(0.0, 0.65, n_samples), 0.05, 12.0)
    po_match = rng.binomial(1, 0.78, n_samples)
    history = rng.integers(0, 2001, n_samples)
    duplicate = rng.binomial(1, 0.12, n_samples)
    delivered = rng.binomial(1, 0.81, n_samples)
    bank_change = rng.binomial(1, 0.10, n_samples)
    x = np.column_stack((amount, po_match, history, duplicate, delivered, bank_change)).astype(float)
    generating_logit = (
        -1.8 + 0.85 * (amount - 1.0) + 2.3 * (1 - po_match)
        - 0.001 * history + 2.5 * duplicate + 1.35 * (1 - delivered) + 1.1 * bank_change
    )
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-generating_logit)))
    indices = np.arange(n_samples)
    train_ids, test_ids = deps["train_test_split"](indices, test_size=0.25, random_state=seed, stratify=y)
    model = deps["make_pipeline"](
        deps["StandardScaler"](),
        deps["LogisticRegression"](C=1.0, max_iter=1000, solver="lbfgs", random_state=seed),
    )
    with deps["threadpool_limits"](limits=1):
        model.fit(x[train_ids], y[train_ids])
        probabilities = model.predict_proba(x[test_ids])[:, 1]
        predictions = model.predict(x[test_ids])
    fingerprint = hashlib.sha256(x.astype("<f8").tobytes() + y.astype("i1").tobytes()).hexdigest()
    return {
        "deps": deps, "model": model, "x_train": x[train_ids], "x_test": x[test_ids],
        "y_train": y[train_ids], "y_test": y[test_ids], "train_ids": train_ids, "test_ids": test_ids,
        "fingerprint": fingerprint,
        "test_metrics": {
            "accuracy": float(deps["accuracy_score"](y[test_ids], predictions)),
            "roc_auc": float(deps["roc_auc_score"](y[test_ids], probabilities)),
            "brier_score": float(deps["brier_score_loss"](y[test_ids], probabilities)),
        },
    }


@lru_cache(maxsize=4)
def _fit_synthetic_tree_model(seed: int, n_samples: int):
    """Fit a cached boosted-tree classifier on the baseline's exact split.

    Args:
        seed (int): Seed shared with _fit_synthetic_model and the tree estimator.
        n_samples (int): Total synthetic rows used to identify the shared bundle.

    Returns:
        tuple: (fitted HistGradientBoostingClassifier, parameter dict, held-out
        metric dict containing accuracy, roc_auc and brier_score).

    Only baseline training rows are fitted; evaluation uses its untouched held-out
    rows. HistGradientBoosting is imported lazily, numerical work is thread-limited,
    and the fitted tuple is cached in process. Import/fitting errors propagate.
    The same-split comparison is descriptive synthetic evidence, not a claim that a
    more complex model improves operational quality."""
    bundle = _fit_synthetic_model(seed, n_samples)
    from sklearn.ensemble import HistGradientBoostingClassifier
    parameters = {
        "max_iter": 80, "max_leaf_nodes": 7, "learning_rate": 0.08,
        "l2_regularization": 1.0, "early_stopping": False, "random_state": seed,
    }
    model = HistGradientBoostingClassifier(**parameters)
    deps = bundle["deps"]
    with deps["threadpool_limits"](limits=1):
        model.fit(bundle["x_train"], bundle["y_train"])
        probabilities = model.predict_proba(bundle["x_test"])[:, 1]
        predictions = model.predict(bundle["x_test"])
    return model, parameters, {
        "accuracy": float(deps["accuracy_score"](bundle["y_test"], predictions)),
        "roc_auc": float(deps["roc_auc_score"](bundle["y_test"], probabilities)),
        "brier_score": float(deps["brier_score_loss"](bundle["y_test"], probabilities)),
    }


def _linear_coefficients(bundle, row):
    """Translate the fitted logistic rule into original feature units.

    Args:
        bundle (dict): The linear bundle from _fit_synthetic_model, including its
            fitted scaler/estimator and NumPy dependency.
        row (numpy.ndarray): One six-value input in original FEATURES order/units.

    Returns:
        dict: Standardized/original intercepts, the reconstructed class-1 log-odds
        margin, interpretation warnings, and per-feature training statistics,
        coefficients, interpretation increments, log-odds changes and odds ratios.

    Coefficients are divided by scaler.scale_; the original intercept subtracts
    coefficient times training mean. History uses a 365-day interpretation increment,
    other inputs one original unit. The caller supplies validated arrays; missing
    bundle members or incompatible shapes propagate errors. No inputs or files are
    modified. Associations hold other inputs fixed and do not establish causation."""
    deps, model = bundle["deps"], bundle["model"]
    np = deps["np"]
    scaler, estimator = model.steps[0][1], model.steps[1][1]
    standardized = estimator.coef_[0]
    raw = standardized / scaler.scale_
    raw_intercept = float(estimator.intercept_[0] - np.dot(raw, scaler.mean_))
    increments = (1.0, 1.0, 365.0, 1.0, 1.0, 1.0)
    return {
        "output_space": "class-1 log-odds",
        "intercept_standardized_space": float(estimator.intercept_[0]),
        "intercept_original_units": raw_intercept,
        "reconstructed_margin_original_units": raw_intercept + float(np.dot(raw, row)),
        "interpretation": "Holding other inputs fixed, the coefficient is an association learned from synthetic data, not a causal effect. Positive coefficients increase class-1 log-odds when that input increases.",
        "features": [
            {
                "feature_key": feature["key"], "feature": feature["label"], "unit": feature["unit"],
                "training_mean": float(scaler.mean_[index]),
                "training_standard_deviation": float(scaler.scale_[index]),
                "coefficient_per_training_standard_deviation": float(standardized[index]),
                "coefficient_per_original_unit": float(raw[index]),
                "interpretation_increment": increments[index],
                "log_odds_change_for_increment": float(raw[index] * increments[index]),
                "odds_multiplier_for_increment": float(np.exp(raw[index] * increments[index])),
                "binary_interpretation": "Change from 0 to 1" if feature["binary"] else None,
            }
            for index, feature in enumerate(FEATURES)
        ],
    }


def _explain_tree_shap(bundle, model, row):
    """Explain one boosted-tree margin and check numerical reconstruction.

    Args:
        bundle (dict): Shared training bundle supplying NumPy and SHAP.
        model: Fitted binary HistGradientBoostingClassifier.
        row (numpy.ndarray): One six-feature validated input in original units.

    Returns:
        dict: TreeExplainer configuration/background description, base value,
        per-feature contributions and sorted positive/negative contributors,
        model/reconstructed margins, absolute reconstruction error, 1e-8 tolerance
        and pass flag, plus sigmoid-converted reference/result probabilities.

    Raises:
        RuntimeError: SHAP does not return one contribution per declared feature.

    SHAP's own additivity check and dependency/model errors may also propagate. The
    path-dependent background uses training path counts stored in the trees, not
    held-out rows. Values add in raw class-1 margin/log-odds; converted base probability
    is not asserted to equal the average predicted probability. Inputs/files are not
    modified, and additivity does not validate usefulness, causality or release."""
    np, shap = bundle["deps"]["np"], bundle["deps"]["shap"]
    explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent",
        feature_names=[feature["label"] for feature in FEATURES],
    )
    explanation = explainer(row.reshape(1, -1), check_additivity=True)
    values = np.asarray(explanation.values, dtype=float).reshape(-1)
    if values.shape != (len(FEATURES),):
        raise RuntimeError("Unexpected SHAP output shape for binary HistGradientBoostingClassifier")
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    margin = float(model.decision_function(row.reshape(1, -1))[0])
    reconstructed = base + float(values.sum())
    error = abs(reconstructed - margin)
    contributions = [
        {"feature_key": feature["key"], "feature": feature["label"], "value": float(row[index]), "contribution_log_odds": float(values[index])}
        for index, feature in enumerate(FEATURES)
    ]
    return {
        "library": "shap", "method": "TreeExplainer with tree_path_dependent background",
        "explained_class": 1, "explained_class_label": CLASS_NAMES[1],
        "output_space": "class-1 raw margin, verified as log-odds by sigmoid against predict_proba",
        "background_source": "Training path counts stored in the fitted tree ensemble. No held-out background rows.",
        "base_value_log_odds": base,
        "base_probability_after_sigmoid": float(1.0 / (1.0 + np.exp(-base))),
        "contributions": contributions,
        "contributors_increasing_score": sorted((item for item in contributions if item["contribution_log_odds"] > 0), key=lambda item: item["contribution_log_odds"], reverse=True),  # Rank positive SHAP shares from largest to smallest.
        "contributors_decreasing_score": sorted((item for item in contributions if item["contribution_log_odds"] < 0), key=lambda item: item["contribution_log_odds"]),  # Rank negative SHAP shares from most negative upward.
        "model_margin_log_odds": margin, "reconstructed_margin_log_odds": reconstructed,
        "reconstruction_absolute_error": error, "reconstruction_tolerance": 1e-8,
        "reconstruction_passed": bool(error <= 1e-8),
        "probability_after_sigmoid": float(1.0 / (1.0 + np.exp(-reconstructed))),
    }


def _explain_shap(bundle, row, seed: int):
    """Explain a logistic margin with a training-only independent background.

    Args:
        bundle (dict): Fitted linear pipeline and shared training/test bundle.
        row (numpy.ndarray): One six-feature validated input in original units.
        seed (int): Seed whose offset selects up to 80 background training rows.

    Returns:
        dict: LinearExplainer/Independent-masker settings, background size, base and
        per-feature log-odds contributions, model/reconstructed margin, absolute
        error and 1e-8 reconstruction verdict, and sigmoid-derived probabilities.

    Raises:
        RuntimeError: SHAP returns an unexpected contribution shape.

    The background and explained row are transformed with the training-fitted scaler.
    The reported mean background probability is separate from sigmoid(base); these
    quantities need not be equal. Dependency/model errors propagate. The helper reads
    but does not mutate the bundle and performs no I/O. Independent-background SHAP
    is a model explanation under that assumption, not a causal or correlated-feature
    analysis."""
    deps, model = bundle["deps"], bundle["model"]
    np, shap = deps["np"], deps["shap"]
    scaler, estimator = model.steps[0][1], model.steps[1][1]
    count = min(80, len(bundle["x_train"]))
    background_ids = np.random.default_rng(seed + 1).choice(len(bundle["x_train"]), count, replace=False)
    background = scaler.transform(bundle["x_train"][background_ids])
    masker = shap.maskers.Independent(background, max_samples=count)
    explainer = shap.LinearExplainer(estimator, masker)
    explanation = explainer(scaler.transform(row.reshape(1, -1)))
    values = np.asarray(explanation.values, dtype=float).reshape(-1)
    if values.shape != (len(FEATURES),):
        raise RuntimeError("Unexpected SHAP output shape for the binary linear model")
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    margin = float(model.decision_function(row.reshape(1, -1))[0])
    reconstructed = base + float(values.sum())
    probability = float(1.0 / (1.0 + np.exp(-reconstructed)))
    error = abs(reconstructed - margin)
    return {
        "library": "shap", "method": "LinearExplainer with Independent masker",
        "explained_class": 1, "explained_class_label": CLASS_NAMES[1],
        "output_space": "log-odds of class 1, not additive probability points",
        "background_source": "80 rows sampled only from the training split",
        "background_rows": count,
        "base_value_log_odds": base,
        "base_probability_after_sigmoid": float(1.0 / (1.0 + np.exp(-base))),
        "background_mean_predicted_probability": float(model.predict_proba(bundle["x_train"][background_ids])[:, 1].mean()),
        "base_value_note": "The base is the mean linear output over the supplied background. Sigmoid(base) need not equal the mean background probability.",
        "contributions": [
            {"feature_key": feature["key"], "feature": feature["label"], "value": float(row[index]), "contribution_log_odds": float(values[index])}
            for index, feature in enumerate(FEATURES)
        ],
        "model_margin_log_odds": margin,
        "reconstructed_margin_log_odds": reconstructed,
        "reconstruction_absolute_error": error,
        "reconstruction_tolerance": 1e-8,
        "reconstruction_passed": bool(error <= 1e-8),
        "probability_after_sigmoid": probability,
    }


def _explain_lime(bundle, row, seed: int, lime_samples: int):
    """Fit a local probability surrogate and evaluate a separate neighbourhood.

    Args:
        bundle (dict): Linear-model bundle with training data and dependencies.
        row (numpy.ndarray): One six-feature validated case in original units.
        seed (int): Seed for LIME fitting; seed + 9301 generates evaluation neighbours.
        lime_samples (int): Requested fitting perturbations, bounded by the caller.

    Returns:
        dict: LIME configuration, class-1 model/surrogate values and case error,
        local coefficients, fitted-neighbourhood weighted R2, and independent metrics
        over 256 valid-domain neighbours: weighted probability MAE/RMSE, weighted R2
        and a count of surrogate predictions outside [0, 1].

    Binary inputs are treated categorically. Independent evaluation jitters continuous
    features, clips them to schema bounds, occasionally flips binary flags, and uses
    LIME's own scaling/kernel for the surrogate comparison. These are fresh artificial
    points, not LIME's fitting sample or real observations. LIME's linear output is
    not clipped into a probability; out-of-range values must remain visible. Scientific
    errors propagate. The helper has CPU cost but no file/network I/O, and neither R2
    nor a small local error establishes global accuracy or real-world fidelity."""
    deps, model = bundle["deps"], bundle["model"]
    np = deps["np"]
    categorical = [index for index, feature in enumerate(FEATURES) if feature["binary"]]
    explainer = deps["LimeTabularExplainer"](
        bundle["x_train"], mode="classification",
        feature_names=[feature["label"] for feature in FEATURES],
        class_names=list(CLASS_NAMES), categorical_features=categorical,
        categorical_names={
            1: ["not matched", "matched"], 3: ["not repeated", "repeated"],
            4: ["not confirmed", "confirmed"], 5: ["not changed", "changed"],
        },
        discretize_continuous=False, sample_around_instance=True, random_state=seed,
        kernel_width=0.75 * len(FEATURES) ** 0.5,
    )
    explanation = explainer.explain_instance(
        row, model.predict_proba, labels=(1,), num_features=len(FEATURES), num_samples=lime_samples,
    )
    true_probability = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    local_probability = float(np.asarray(explanation.local_pred).reshape(-1)[0])
    intercept = float(explanation.intercept[1])
    coefficients = {int(index): float(value) for index, value in explanation.local_exp[1]}

    # Independent valid-domain neighbourhood. It is not LIME's fitting sample.
    evaluation_rng = np.random.default_rng(seed + 9301)
    neighbourhood = np.tile(row, (256, 1)).astype(float)
    for index, feature in enumerate(FEATURES):
        if feature["binary"]:
            flips = evaluation_rng.random(256) < 0.10
            neighbourhood[flips, index] = 1.0 - row[index]
        else:
            scale = float(bundle["x_train"][:, index].std())
            neighbourhood[:, index] += evaluation_rng.normal(0.0, 0.25 * scale, 256)
            neighbourhood[:, index] = np.clip(neighbourhood[:, index], feature["minimum"], feature["maximum"])
    design = (neighbourhood - explainer.scaler.mean_) / explainer.scaler.scale_
    case_design = (row - explainer.scaler.mean_) / explainer.scaler.scale_
    for index in categorical:
        design[:, index] = (neighbourhood[:, index] == row[index]).astype(float)
        case_design[index] = 1.0
    distances = np.linalg.norm(design - case_design, axis=1)
    weights = explainer.base.kernel_fn(distances)
    surrogate = np.full(256, intercept)
    for index, coefficient in coefficients.items():
        surrogate += coefficient * design[:, index]
    actual = model.predict_proba(neighbourhood)[:, 1]
    independent_mae = float(np.average(np.abs(actual - surrogate), weights=weights))
    independent_rmse = float(np.sqrt(np.average((actual - surrogate) ** 2, weights=weights)))
    independent_r2 = float(deps["r2_score"](actual, surrogate, sample_weight=weights))
    return {
        "library": "lime", "method": "LimeTabularExplainer, local weighted linear surrogate",
        "explained_class": 1, "explained_class_label": CLASS_NAMES[1],
        "output_space": "class-1 probability approximated by a local linear surrogate",
        "num_samples": lime_samples, "discretize_continuous": False,
        "sample_around_instance": True,
        "kernel_width": float(0.75 * len(FEATURES) ** 0.5),
        "intercept": intercept,
        "contributions": [{"condition": str(condition), "weight": float(weight)} for condition, weight in explanation.as_list(label=1)],
        "model_probability": true_probability,
        "surrogate_probability_at_case": local_probability,
        "case_absolute_error": abs(true_probability - local_probability),
        "fit_neighbourhood_weighted_r2": float(explanation.score),
        "fit_metric_note": "LIME's score describes its fitted perturbation sample. It is not independent validation or global model accuracy.",
        "independent_neighbourhood": {
            "rows": 256, "seed": seed + 9301,
            "generation": "Valid-domain numeric jitter and occasional binary flips around this case, separate from LIME fitting samples",
            "weighted_mae_probability": independent_mae,
            "weighted_rmse_probability": independent_rmse,
            "weighted_r2": independent_r2,
            "surrogate_outside_probability_range_count": int(((surrogate < 0) | (surrogate > 1)).sum()),
        },
    }


def run_explainability_lab(
    invoice: Mapping[str, Real] | None = None, *, seed: int = 1729,
    n_samples: int = 800, lime_samples: int = 4000, case_index: int = 0,
) -> dict:
    """Fit a synthetic logistic baseline and explain one score with SHAP/LIME.

    Args:
        invoice (Mapping[str, Real] | None): Validated six-field fictional input, or
            None to explain the held-out row selected by case_index.
        seed (int): Inclusive range 0 through 2**31 - 1; default 1729.
        n_samples (int): Total synthetic rows, inclusive range 200 through 5000.
        lime_samples (int): LIME fitting perturbations, range 500 through 20000.
        case_index (int): Zero-based position within the 25-percent held-out split.
            Its range is checked even when invoice supplies the explained row.

    Returns:
        dict: Finite-JSON-compatible environment/parameters, split fingerprint and
        held-out metrics, case features/probability/class, SHAP and LIME reports,
        numerical checks, illustrative LIME assessment, timing_seconds and limits.

    Raises:
        ValueError: Inputs, budgets or held-out index violate the bounded schema.
        ExplainabilityUnavailable: Optional scientific distributions are missing.

    Other import/fitting/explainer failures propagate. Work runs on CPU with a
    one-thread numerical limit and may populate the in-process fitted-model cache.
    No files, remote services or MLflow registry are used. ok/execution_ok indicate
    execution, not an accepted explanation: LIME's illustrative gate may still fail.
    SHAP checks log-odds reconstruction, while LIME approximates probability; neither
    proves causality, useful real-invoice performance or production readiness."""
    started = time.perf_counter()
    _bounded_int("seed", seed, 0, 2**31 - 1)
    _bounded_int("n_samples", n_samples, 200, 5000)
    _bounded_int("lime_samples", lime_samples, 500, 20000)
    _bounded_int("case_index", case_index, 0, n_samples - 1)
    supplied = validate_invoice(invoice) if invoice is not None else None
    bundle = _fit_synthetic_model(seed, n_samples)
    deps, model = bundle["deps"], bundle["model"]
    np = deps["np"]
    if case_index >= len(bundle["x_test"]):
        raise ValueError(f"case_index must be less than the held-out size {len(bundle['x_test'])}")
    row = np.asarray(supplied, dtype=float) if supplied is not None else bundle["x_test"][case_index].copy()
    with deps["threadpool_limits"](limits=1):
        probability = float(model.predict_proba(row.reshape(1, -1))[0, 1])
        predicted_class = int(model.predict(row.reshape(1, -1))[0])
        shap_started = time.perf_counter()
        shap_result = _explain_shap(bundle, row, seed)
        shap_seconds = time.perf_counter() - shap_started
        lime_started = time.perf_counter()
        lime_result = _explain_lime(bundle, row, seed, lime_samples)
        lime_seconds = time.perf_counter() - lime_started
    return {
        "ok": True,
        "scenario": "Synthetic invoice-review score with two local explanation methods",
        "intended_use": "Engineering exercise only. The score supports review and is not proof of wrongdoing.",
        "environment": {"python_version": platform.python_version(), "packages": explainability_status()["package_versions"], "device": "CPU", "gpu_used": False},
        "parameters": {"seed": seed, "n_samples": n_samples, "test_fraction": 0.25, "lime_samples": lime_samples, "case_index": case_index, "threads_limit": 1},
        "training": {
            "data_origin": "Entirely synthetic features and labels generated with a seeded probabilistic rule",
            "dataset_sha256": bundle["fingerprint"],
            "model": "StandardScaler followed by sklearn LogisticRegression",
            "model_parameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
            "train_rows": len(bundle["train_ids"]), "test_rows": len(bundle["test_ids"]),
            "train_test_index_overlap": int(np.intersect1d(bundle["train_ids"], bundle["test_ids"]).size),
            "test_metrics": bundle["test_metrics"],
        },
        "case": {
            "origin": "caller-provided numeric demonstration input" if supplied is not None else "held-out synthetic record",
            "synthetic_record_index": None if supplied is not None else int(bundle["test_ids"][case_index]),
            "features": {feature["key"]: float(row[index]) for index, feature in enumerate(FEATURES)},
            "feature_labels": {feature["key"]: feature["label"] for feature in FEATURES},
            "positive_class": 1, "positive_class_label": CLASS_NAMES[1],
            "probability_review_suggested": probability,
            "predicted_class": predicted_class, "predicted_label": CLASS_NAMES[predicted_class],
            "decision_threshold": 0.5,
            "synthetic_reference_label": None if supplied is not None else int(bundle["y_test"][case_index]),
        },
        "shap": shap_result, "lime": lime_result,
        "checks": {
            "same_explained_class": shap_result["explained_class"] == lime_result["explained_class"] == 1,
            "shap_reconstruction_passed": shap_result["reconstruction_passed"],
            "shap_probability_matches_model": abs(shap_result["probability_after_sigmoid"] - probability) <= 1e-10,
            "train_test_separated": not np.intersect1d(bundle["train_ids"], bundle["test_ids"]).size,
        },
        "quality_assessment": {
            "execution_ok": True,
            "shap_numerical_reconstruction_passed": shap_result["reconstruction_passed"],
            "lime_case_in_probability_range": 0.0 <= lime_result["surrogate_probability_at_case"] <= 1.0,
            "lime_case_error": lime_result["case_absolute_error"],
            "lime_illustrative_error_threshold": 0.05,
            "lime_illustrative_gate_passed": (
                0.0 <= lime_result["surrogate_probability_at_case"] <= 1.0
                and lime_result["case_absolute_error"] <= 0.05
            ),
            "illustrative_only": True,
            "note": "The LIME threshold is a proposed teaching diagnostic, not production or business acceptance. SHAP reconstruction checks numerical additivity, not causal truth or model suitability.",
        },
        "timing_seconds": {"total": time.perf_counter() - started, "shap": shap_seconds, "lime_with_fidelity": lime_seconds},
        "limitations": [
            "Synthetic labels and limited numeric features do not establish usefulness on real invoices.",
            "The default classifier is a small transparent linear baseline, not XGBoost or a deployed decision system.",
            "SHAP values are additive in class-1 log-odds. LIME weights describe a local probability surrogate and cannot be compared as identical units.",
            "The SHAP background and the LIME perturbation distribution influence the explanations.",
            "LIME's fitted weighted R2 is not global predictor accuracy. Independent local metrics remain limited to the stated artificial neighbourhood.",
            "Continuous LIME fitting perturbations can be implausible, and local surrogate predictions need not stay within [0, 1].",
            "Neither method proves real-world causation, fairness, legal compliance or wrongdoing.",
            "This run uses CPU only. It does not claim that SHAP has no optional GPU implementation.",
            "No MLflow registry, tracking server or production release gate is exercised by this helper.",
            "Execution timings describe this local run and are not production performance benchmarks.",
        ],
    }


def run_explainability_comparison(
    invoice: Mapping[str, Real] | None = None, *, seed: int = 1729,
    n_samples: int = 800, lime_samples: int = 4000, case_index: int = 0,
) -> dict:
    """Compare a linear baseline and boosted trees on the same synthetic split.

    Args:
        invoice (Mapping[str, Real] | None): Six-field fictional input. None selects
            the explicit high-score demonstration invoice, not a held-out example.
        seed (int): Shared generation/fitting seed; validated by run_explainability_lab.
        n_samples (int): Total synthetic rows, range 200 through 5000.
        lime_samples (int): Linear-baseline LIME fitting rows, range 500 through 20000.
        case_index (int): Valid held-out position passed through to the baseline;
            it is still validated although the comparison always supplies an input.

    Returns:
        dict: The unchanged-schema report under linear_baseline, original-unit
        coefficients, positive linear contributors, tree_model/Tree SHAP results,
        same-split metric table, numerical checks, illustrative quality assessment,
        total timing_seconds and explicit production_approval=False.

    The comparison has no observed label for its explicit displayed input. Both fits
    use identical training and evaluation rows; LIME belongs to the linear model and
    Tree SHAP to the tree model. Their outcomes are not a direct same-model explainer
    comparison. Public-input/dependency/model errors propagate as in the baseline.
    Fitting and explanation use CPU/cache memory only; no files, registry or release
    state change. All numerical checks can pass while LIME's fidelity diagnostic
    fails. Synthetic data and the linear label rule cannot establish model-family
    superiority, causality or operational acceptance."""
    started = time.perf_counter()
    demonstration = {
        "amount_ratio": 2.4, "purchase_order_match": 0, "supplier_history_days": 90,
        "duplicate_reference": 1, "delivery_confirmed": 0, "recent_bank_change": 1,
    }
    linear = run_explainability_lab(
        demonstration if invoice is None else invoice, seed=seed, n_samples=n_samples,
        lime_samples=lime_samples, case_index=case_index,
    )
    bundle = _fit_synthetic_model(seed, n_samples)
    deps, np = bundle["deps"], bundle["deps"]["np"]
    row = np.asarray([linear["case"]["features"][feature["key"]] for feature in FEATURES], dtype=float)
    tree, tree_parameters, tree_metrics = _fit_synthetic_tree_model(seed, n_samples)
    with deps["threadpool_limits"](limits=1):
        tree_probability = float(tree.predict_proba(row.reshape(1, -1))[0, 1])
        tree_class = int(tree.predict(row.reshape(1, -1))[0])
        tree_shap = _explain_tree_shap(bundle, tree, row)
        coefficients = _linear_coefficients(bundle, row)
    linear_shap = linear["shap"]
    coefficient_error = abs(coefficients["reconstructed_margin_original_units"] - linear_shap["model_margin_log_odds"])
    checks = {
        "same_training_and_heldout_rows": True,
        "train_test_separated": linear["checks"]["train_test_separated"],
        "linear_shap_reconstruction_passed": linear_shap["reconstruction_passed"],
        "tree_shap_reconstruction_passed": tree_shap["reconstruction_passed"],
        "tree_sigmoid_matches_probability": abs(tree_shap["probability_after_sigmoid"] - tree_probability) <= 1e-10,
        "linear_raw_coefficients_reconstruct_margin": coefficient_error <= 1e-8,
        "same_explained_class": linear_shap["explained_class"] == tree_shap["explained_class"] == 1,
    }
    return {
        "ok": True, "scenario": "Two fitted synthetic invoice-review models with local explanations",
        "intended_use": "Engineering comparison only. Scores support review, not proof of wrongdoing.",
        "production_approval": False,
        "environment": linear["environment"], "parameters": linear["parameters"],
        "case": {
            "origin": "fixed fictional high-score example" if invoice is None else "caller-provided numeric demonstration input",
            "features": linear["case"]["features"], "feature_labels": linear["case"]["feature_labels"],
            "positive_class": 1, "positive_class_label": CLASS_NAMES[1],
            "decision_threshold": 0.5, "ground_truth_label": None,
            "ground_truth_note": "This explicit fictional input is not a labelled held-out observation.",
        },
        "linear_baseline": linear,
        "linear_coefficients": coefficients,
        "linear_contributors_increasing_score": sorted(
            (item for item in linear_shap["contributions"] if item["contribution_log_odds"] > 0),
            key=lambda item: item["contribution_log_odds"], reverse=True,  # Order the displayed positive local contributions by size.
        ),
        "tree_model": {
            "training": {
                "model": "sklearn HistGradientBoostingClassifier", "model_parameters": tree_parameters,
                "dataset_sha256": bundle["fingerprint"], "train_rows": len(bundle["train_ids"]),
                "test_rows": len(bundle["test_ids"]), "train_test_index_overlap": 0,
                "test_metrics": tree_metrics,
            },
            "case": {
                "probability_review_suggested": tree_probability,
                "predicted_class": tree_class, "predicted_label": CLASS_NAMES[tree_class],
                "decision_threshold": 0.5,
            },
            "shap": tree_shap,
        },
        "model_comparison": {
            "evaluation_rows": len(bundle["test_ids"]),
            "same_split_dataset_sha256": bundle["fingerprint"],
            "metrics": [
                {"model_id": "logistic_regression", "model": linear["training"]["model"], **linear["training"]["test_metrics"]},
                {"model_id": "hist_gradient_boosting", "model": "sklearn HistGradientBoostingClassifier", **tree_metrics},
            ],
            "metric_meaning": "Accuracy is thresholded correctness, AUROC measures ranking, and lower Brier score indicates better probability error on this split.",
            "interpretation": "One fixed synthetic split is descriptive only. The label-generating rule is linear, so this exercise is not a neutral benchmark of model families. No real-data superiority is established.",
        },
        "checks": checks,
        "quality_assessment": {
            "execution_ok": True, "numerical_checks_passed": all(checks.values()),
            "lime_assessment": linear["quality_assessment"],
            "model_quality_accepted_for_real_use": False, "illustrative_only": True,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
        "limitations": [
            "Entirely synthetic data, six limited numeric features and one fixed held-out split do not establish usefulness on real invoices.",
            "The displayed high-score input has no observed ground-truth label and must not be described as a detected wrong invoice.",
            "Both SHAP methods reconstruct class-1 log-odds. Their background assumptions differ, so their contribution sizes are not a direct causal comparison.",
            "Linear coefficients are global fitted associations holding other inputs fixed. Tree SHAP values are local contributions relative to the ensemble background.",
            "LIME is retained as a diagnostic counterexample. Its raw surrogate can exceed [0, 1] and must not replace the classifier's probability.",
            "No SHAP value, coefficient or LIME explanation proves causation, fairness, wrongdoing or production readiness.",
            "CPU execution, local model fitting and explanation are exercised. No MLflow tracking service, registry, real invoice ingestion or production deployment is exercised.",
        ],
    }
