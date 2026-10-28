# Explainability review: SHAP and LIME on the same models, cases and outputs — 28 October 2026

This package answers one critique of the 27 October explainability evidence: SHAP additivity, local fidelity, stability and usefulness to a reviewer are **different questions**, and a passed numerical check must not become a global validation. Here each quality is measured separately, for **both** fitted models, on the **same** 600/200 split, the **same** 20 test cases and the **same** output, with the scale of every number stated.

Everything is synthetic. No real invoice, person or organisation is involved. It is a small engineering study, not a model, explanation or production acceptance. The 27 October code and evidence ([`explainability-comparison.json`](../../ai-engineering-20261027/explainability-comparison.json)) are unchanged; the review reproduces its reference numbers exactly (LIME probability surrogate 1.4251149187993795, tree probability 0.909404959651305, same dataset SHA-256).

| File | Content |
| --- | --- |
| [`protocol.json`](protocol.json) | Questions, cases, scales, methods, checks and thresholds, written before the review was run, with three recorded amendments |
| [`explainability-review-20261028.json`](explainability-review-20261028.json) | Full result: per model, scale and case, every check, plus data, interpretable model, counterfactuals, environment, durations and source hashes |

Code: [`lab/explainability_review.py`](../../../../lab/explainability_review.py), [`scripts/review_explainability_engineering.py`](../../../../scripts/review_explainability_engineering.py), tests [`tests/test_explainability_review.py`](../../../../tests/test_explainability_review.py). The 27 October module [`lab/engineering_explainability.py`](../../../../lab/engineering_explainability.py) is reused unchanged for the data, models and split.

## Scales

Every number below names its scale.

- **log-odds**: class-1 log-odds, the raw margin returned by `decision_function`. For both models, sigmoid(log-odds) equals `predict_proba` within 0.0 on the 200 test rows.
- **probability**: class-1 probability, `predict_proba[:, 1]`.
- Rank correlations, Jaccard indices and pass counts are unitless. A rank or sign comparison is only made between two numbers on the same scale.

## What was compared

| Item | Setting |
| --- | --- |
| Data | 800 synthetic invoices, six independent features, label from a known linear logit; stratified split 600 train / 200 test, seed 1729 |
| Models | `StandardScaler` + `LogisticRegression(C=1)`; `HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7)`. Unchanged |
| Cases | 20 test rows, 5 per stratum: A label 1 both models correct, B label 0 both correct, C label 1 at least one model wrong, D label 0 at least one model wrong; `default_rng(1729 + 3000)`; no shortfall (strata had 20, 136, 30 and 14 rows). Both models explain the same 20 rows |
| Reference case | The 27 October fixed high-score input (amount ratio 2.4, no purchase order, 90 days, repeated reference, no delivery, bank change). No label; reported separately and excluded from every aggregate |
| SHAP | `shap.explainers.Exact` (all 64 coalitions) with an `Independent` (interventional) masker over 100 training rows, for both models and both scales. Log-odds: model function `decision_function`. Probability: `predict_proba[:, 1]` with the same interventional background |
| LIME | 27 October configuration (binary fields categorical, no discretisation, sampling around the case, kernel width 0.75 x sqrt(6), 4,000 samples). Log-odds: regression mode on `decision_function`. Probability: classification mode, label 1 |
| LIME for comparison | Each coefficient times (design value at the case minus its mean over the same SHAP background). This is the interventional SHAP value of LIME's linear surrogate, so it can be ranked and signed against SHAP on the same scale. Raw coefficient ranks are reported as a secondary view |

## 1. Predictive performance and calibration (200 test invoices)

Values with 95 percent percentile bootstrap intervals over 1,000 resamples of the 200 rows (seed 1729 + 2000). Accuracy is a proportion at threshold 0.5; ROC AUC is a ranking measure; Brier score, log loss and ECE use the predicted class-1 **probability**. Test positive rate 0.25.

| Model | Accuracy | ROC AUC | Brier score | Log loss | ECE, 10 equal-width bins | ECE, 5 equal-mass bins |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.860 [0.810, 0.910] | 0.819 [0.730, 0.886] | 0.128 [0.101, 0.160] | 0.417 [0.344, 0.512] | 0.072 [0.056, 0.131] | 0.050 |
| Boosted trees | 0.810 [0.755, 0.860] | 0.810 [0.730, 0.874] | 0.138 [0.110, 0.171] | 0.434 [0.359, 0.524] | 0.042 [0.041, 0.113] | 0.040 |

Accuracy, AUC and Brier score equal the 27 October values. The ECE intervals are wide and the tree point estimate sits at the lower edge of its interval: with 200 rows many bins hold fewer than 15 rows, so ECE is noisy, and its bootstrap distribution lies mostly above the point estimate. The two models' intervals overlap widely on this split.

Reliability table (probability scale; mean predicted probability against observed positive rate per bin):

| Bin | LR count | LR mean predicted | LR observed | Trees count | Trees mean predicted | Trees observed |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0-0.1 | 74 | 0.06 | 0.08 | 79 | 0.05 | 0.08 |
| 0.1-0.2 | 39 | 0.14 | 0.13 | 31 | 0.14 | 0.16 |
| 0.2-0.3 | 20 | 0.24 | 0.15 | 23 | 0.24 | 0.22 |
| 0.3-0.4 | 19 | 0.34 | 0.26 | 21 | 0.35 | 0.29 |
| 0.4-0.5 | 12 | 0.46 | 0.17 | 12 | 0.44 | 0.42 |
| 0.5-0.6 | 13 | 0.55 | 0.69 | 13 | 0.54 | 0.46 |
| 0.6-0.7 | 6 | 0.64 | 1.00 | 12 | 0.64 | 0.75 |
| 0.7-0.8 | 8 | 0.74 | 0.75 | 5 | 0.75 | 1.00 |
| 0.8-0.9 | 4 | 0.85 | 1.00 | 4 | 0.84 | 0.75 |
| 0.9-1.0 | 5 | 0.96 | 0.80 | 0 | n/a | n/a |

## 2. One check per quality

Pass counts are out of the 20 selected cases. A study-level verdict passes only when every case passes. Thresholds are the illustrative values fixed in the protocol before the run (they are not business, legal or production criteria).

| Check | Method | LR, log-odds | LR, probability | Trees, log-odds | Trees, probability |
| --- | --- | --- | --- | --- | --- |
| **C1 additivity**: max \|base + sum − output\|, tolerance 1e-8 | SHAP (Exact) | **pass** 6.2e-15 | **pass** 1.9e-15 | **pass** 4.9e-15 | **pass** 1.2e-15 |
| C1 additivity | LIME | not a design property | not a design property | not a design property | not a design property |
| **C1b implementation**: max \|Exact − independent enumeration\|, tolerance 1e-6 | SHAP | **pass** 2.8e-15 | **pass** 8.3e-16 | **pass** 2.6e-15 | **pass** 6.7e-16 |
| C1b model-specific explainer vs enumeration | SHAP | **pass** 7.8e-16 (LinearExplainer) | not available | **fail** 3.5e-3 (TreeExplainer) | **fail** 5.3e-2 (TreeExplainer) |
| **C2 fidelity at the case**: \|surrogate − output\|, max (median) | LIME | **pass** 20/20, 0.014 (0.002), limit 0.2 | **fail** 12/20, 0.187 (0.041), limit 0.05 | **fail** 5/20, 1.302 (0.493), limit 0.2 | **fail** 5/20, 0.205 (0.084), limit 0.05 |
| **C3 fidelity on fresh neighbours**: held-out weighted R², min (median), limit 0.8 | LIME | **pass** 20/20, 1.000 (1.000) | **fail** 17/20, 0.730 (0.916) | **fail** 7/20, 0.672 (0.759) | **fail** 7/20, 0.680 (0.748) |
| C2 and C3 together | LIME | **pass** 20/20 | **fail** 11/20 | **fail** 0/20 | **fail** 2/20 |
| **C4 range** diagnostic: surrogate at the case inside [0, 1] | LIME | not applicable | **pass** 20/20; 15.1 % of neighbours outside | not applicable | **pass** 20/20; 5.4 % of neighbours outside |
| **C5 stability**: mean pairwise top-3 Jaccard and Spearman ≥ 0.8, 10 seeds | LIME | **pass** 20/20 | **fail** 19/20 | **fail** 18/20 | **pass** 20/20 |
| **C5 stability**: same rule, 5 background samples | SHAP | **fail** 15/20 | **fail** 14/20 | **fail** 16/20 | **fail** 11/20 |
| **C6 reference sensitivity**: training sample vs median baseline, Jaccard and Spearman ≥ 0.8 | SHAP | **fail** 12/20 insensitive | **fail** 9/20 | **fail** 9/20 | **fail** 8/20 |
| **C7 SHAP–LIME agreement**: Spearman of \|attributions\| ≥ 0.8 and same sign on SHAP's top 3 | both | **pass** 20/20 | **fail** 19/20 | **fail** 8/20 | **fail** 7/20 |

Fresh neighbours for C3 are LIME perturbations drawn by the next seed (1730) and therefore not used to fit the evaluated surrogate. The fitting-sample weighted R² is close to the held-out one in every block (minimum 1.000, 0.722, 0.673 and 0.678 in column order). On a second, plausible neighbourhood (schema-clipped jitter of 0.25 standard deviations, 10 percent binary flips) the median weighted R² is 1.000, 0.881, 0.443 and 0.489, with minima down to −3.0 for the trees on log-odds.

The review rebuilds LIME's surrogate from its coefficients and its exact captured perturbations; the rebuilt prediction at the case and the rebuilt fitting R² equal LIME's own values (difference 0.0 in every block), and LIME drew identical perturbations for both models and both modes.

### What the checks show

- **Additivity says nothing about the other qualities.** Exact SHAP adds up to within 6.2e-15 in all four blocks, while fidelity, stability, background sensitivity and agreement fail in most of them.
- **The same library can fail additivity on the same model.** On 27 October, path-dependent Tree SHAP passed additivity (2.2e-15 log-odds). Interventional `TreeExplainer` on the same boosted trees misses it by **2.6e-3 log-odds** and **6.6e-4 probability** with its own base value, and differs from exact enumeration by 3.5e-3 log-odds. Nineteen background values (9 amount ratios, 10 supplier-history values) equal a split threshold exactly. After moving them by 1e-6, the log-odds difference falls to 1.5e-8 (additivity 4.0e-8). On probability the gap remains (5.3e-2), because `TreeExplainer` rescales each reference's margin attributions by (σ(f(x)) − σ(f(r))) / (f(x) − f(r)): it matches that reconstruction within 3.0e-9 but not the exact Shapley values of the probability. Both versions are additive, yet they attribute differently. These two diagnostics were added by protocol amendment and change no verdict.
- **LIME's 27 October failure is a scale effect on the linear model.** On the reference case the logistic log-odds surrogate is 6.242 against a model output of 6.263 (error 0.021, held-out R² 1.000), and its attributions match SHAP (Spearman 1.0). On probability the same case gives a surrogate of 1.425 against 0.998 (error 0.427, outside [0, 1]). Across the 20 test cases the probability surrogate stays inside [0, 1] at the case, but on average 15.1 percent of its predictions on the fitting neighbours (at most 25.8 percent) leave the range, and 9 cases fail fidelity.
- **LIME cannot follow the trees.** Boosted trees are piecewise constant in log-odds; no case passes both fidelity checks on log-odds and two pass on probability. Errors at the case reach 1.302 log-odds and 0.205 probability.
- **Stable is not the same as faithful.** LIME with 4,000 samples barely moves across seeds (median pairwise Jaccard 1.0 in every block), including on the trees where it is unfaithful.
- **SHAP top-3 ranks depend on the background sample.** With 100 background rows, 4 to 9 cases per block fail the stability rule, each because its top 3 changes between the five samples; counting every change, 7 to 11 of the 20 cases change their top 3 at least once. Example, case at test position 18 on logistic log-odds: amount ratio −0.415 and delivery confirmation −0.346 are close, so third place alternates (mean Jaccard 0.70).
- **The reference changes the explanation.** Against a single median baseline, every feature equal to the baseline gets exactly 0. The base value moves by −1.365 log-odds on the logistic model and −1.818 on the trees; single contributions change by up to 0.415 and 1.185 log-odds.
- **Agreement depends on how LIME is read.** On logistic log-odds, background-centred LIME attributions rank exactly like SHAP (median Spearman 1.0, 20/20), but raw LIME coefficients do not (median Spearman −0.20): a coefficient is a slope, not a contribution. On the trees the median Spearman is 0.71 (log-odds) and 0.77 (probability), median top-3 Jaccard 0.5.

## 3. Assumptions and data

- **Dependence.** On the 600 training rows the largest absolute off-diagonal correlation is 0.078 (Pearson) and 0.079 (Spearman); every variance inflation factor is between 1.004 and 1.016. The generator draws the features independently, so these are sampling noise.
- **Interventional SHAP** replaces masked features by background values independently of the kept features. It evaluates the model on hybrid rows, which can be impossible combinations when features are dependent. The assumption matches this generator; real invoices would need their dependence measured first. Here every hybrid row stays inside the schema bounds because each coordinate comes from a valid row.
- **LIME's neighbourhood.** Continuous features receive Gaussian noise with the training standard deviation around the case, without clipping. Binary features are resampled from the training frequencies, independently of the case. Weights are sqrt(exp(−d²/w²)) in LIME's design space. On the primary seed, a mean of **27.9 percent** of LIME's fitting perturbations per case (at most 63.5 percent) fall outside the schema: 15.8 percent have an amount ratio outside 0.05 to 12 and 14.3 percent have a supplier history outside 0 to 3,650 days. LIME is therefore fitted partly on impossible invoices.

## 4. Interpretable model: logistic regression

Preprocessing: `StandardScaler` fitted on the 600 training rows (means 1.206, 0.795, 1,052.59 days, 0.112, 0.822, 0.092; scales 0.836, 0.404, 579.50 days, 0.315, 0.383, 0.289). Coefficients are on the **log-odds** scale; odds ratios are exp(coefficient × increment). Intervals: 95 percent bootstrap over 200 refits on resampled training rows (seed 1729 + 4000). The generating coefficients are known only because the data are synthetic.

| Feature | Per training SD | Per original unit | Increment | Odds ratio per increment | Generating per unit | Inside interval |
| --- | --- | --- | --- | --- | --- | --- |
| Amount ratio | 0.853 [0.618, 1.240] | 1.020 [0.770, 1.368] | 1 | 2.77 [2.16, 3.93] | 0.85 | yes |
| Purchase order matched | −0.761 [−1.006, −0.537] | −1.885 [−2.458, −1.350] | 0 → 1 | 0.15 [0.09, 0.26] | −2.3 | yes |
| Supplier history | −0.596 [−0.860, −0.376] | −0.00103 per day [−0.00149, −0.00064] | 365 days | 0.69 [0.58, 0.79] | −0.001 | yes |
| Repeated reference | 0.638 [0.410, 0.880] | 2.026 [1.325, 2.769] | 0 → 1 | 7.59 [3.76, 15.95] | 2.5 | yes |
| Delivery confirmed | −0.602 [−0.829, −0.403] | −1.572 [−2.125, −1.069] | 0 → 1 | 0.21 [0.12, 0.34] | −1.35 | yes |
| Recent bank change | 0.311 [0.066, 0.497] | 1.077 [0.249, 1.745] | 0 → 1 | 2.94 [1.28, 5.73] | 1.1 | yes |

Intercept in original units 0.805 [0.171, 1.528]; generating intercept 1.0. Every generating value lies inside its interval. The estimates are shrunk by L2 regularisation (C = 1) and noisy with 600 rows: the repeated-reference odds ratio interval spans more than a factor of four.

- **No interactions on log-odds.** The mixed second difference f(x_ij) − f(x_i) − f(x_j) + f(x) over the 20 cases and 15 feature pairs is at most 1.1e-15 log-odds. On **probability** it reaches 0.279 (all 15 pairs above 1e-9): the sigmoid couples the features even without interaction terms. The boosted trees interact on both scales (1.865 log-odds, 14 of 15 pairs; 0.330 probability).
- **Correlated features.** A coefficient is the change with the other inputs held fixed. With correlated inputs that combination can be unrealistic and the coefficients become unstable. With VIF ≤ 1.016 this limit is minor here; it must be measured on real data.

## 5. Counterfactuals

The 27 October code produced none. A minimal bounded counterfactual flips the model's decision at 0.5 by changing only the purchase-order match, the delivery confirmation and the amount ratio (grid 0.05 to 12.00, step 0.01). Supplier history, repeated reference and bank change are treated as immutable facts. Cost = flipped flags + |amount change| / training SD of amount (0.836).

| Model | Cases | Found | Decision really flips (re-predicted) | Inside schema (`validate_invoice`) | Immutable fields unchanged | Median cost | Fields changed (cases) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 20 | 20 | 20 | 20 | 20 | 0.862 | amount 17, purchase order 8, delivery 3 |
| Boosted trees | 20 | 20 | 20 | 20 | 20 | 1.000 | amount 16, purchase order 4, delivery 5 |

Reference case: **no** bounded counterfactual exists for the logistic model (probability 0.998; even with a purchase order, delivery confirmation and the minimum amount, the immutable facts keep the decision). The trees flip from 0.909 to 0.463 (probability) by changing all three mutable fields (cost 3.11). On the logistic model, 17 of the 20 counterfactuals land between 0.497 and 0.503 in probability, just across the threshold, so they are fragile. A counterfactual describes the model's boundary; it is not advice, and lowering an invoice amount is not an action anyone should take to change a score.

## 6. Not measured: reviewer usefulness and causality

Neither is measured or claimed. SHAP, LIME, coefficients and counterfactuals describe the fitted model, not the world. A proposed reviewer study, not run:

- **Task:** reviewers see synthetic cases with the score and decide the route and whether the score looks wrong, under four conditions: score only, with SHAP, with LIME, with a counterfactual. Some cases come from a deliberately corrupted model (one feature sign inverted).
- **Participants:** at least 12 accounts-payable or audit reviewers who did not build the model; within-subject design, Latin-square counterbalancing, randomised cases.
- **Measures:** primary, detection rate of corrupted-model cases; secondary, decision accuracy against the synthetic truth, forward-simulation accuracy, over-reliance on wrong scores, time per case, and a short trust and workload questionnaire. Analysis by mixed-effects logistic regression, pre-registered.

## 7. Limits

- Synthetic data, six features and one split do not establish usefulness on real invoices. The label rule is linear, so the logistic model is correctly specified: this is not a neutral comparison of model families.
- Twenty stratified cases describe these cases, not every input. Thresholds are illustrative.
- SHAP–LIME agreement compares two explanations with each other, not with a known true attribution.
- The tie and rescaling findings concern shap 0.54.1 with scikit-learn 1.9.1 `HistGradientBoostingClassifier` on this data; other versions or models were not tested.
- Durations describe this local CPU run, not production performance. No MLflow service, registry, real data or deployment is exercised. `production_approval` is false.

## 8. Reproduction

Environment: CPython 3.12.10 in a separate virtual environment; numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, shap 0.54.1, lime 0.2.0.1, threadpoolctl 3.7.0, pandas 3.1.0, pytest 9.1.1 (all pinned versions installed as listed in `requirements-explainability.txt`). Windows 11, processor string "Intel64 Family 6 Model 158 Stepping 12, GenuineIntel", 16 logical CPUs, **one numerical thread, CPU only**, `CUDA_VISIBLE_DEVICES=-1`, no GPU used.

Seeds: data, split and models 1729; cases 1729 + 3000; SHAP backgrounds 1729 + 101 to 105; LIME 1729 + 0 to 9 (primary 1729, held-out sample from 1730); plausible neighbourhood 1729 + 9301; metric bootstrap 1729 + 2000; coefficient bootstrap 1729 + 4000.

Durations of the retained run: performance and calibration 9.3 s, logistic explanations 22.2 s, tree explanations 37.1 s, model and counterfactual reports 1.1 s, **total 73.0 s** (process 73.1 s), started 2026-10-28 19:53:28 UTC. The source SHA-256 hashes are in `provenance` in the JSON. A second run into a temporary file (78.7 s) gave a report identical to the retained one apart from `timing_seconds` and `provenance`.

From the repository root, in PowerShell:

```powershell
py -3.12 -m venv ..\multimodal-xai-venv
..\multimodal-xai-venv\Scripts\python.exe -m pip install -r requirements-explainability.txt
..\multimodal-xai-venv\Scripts\python.exe -B scripts\review_explainability_engineering.py --output "$env:TEMP\explainability-review-rerun.json"
..\multimodal-xai-venv\Scripts\python.exe -B -m pytest tests\test_explainability_review.py tests\test_engineering_explainability.py -q -p no:cacheprovider
```

The script reads the parameters and thresholds from `protocol.json` and refuses to run if they differ from what the code applies. It refuses an existing output file. Exit code 0 means every check passed, 1 means the report was written with at least one failed check (the case here), 2 means nothing was run (existing output, missing dependency or mismatched protocol). The retained report was produced with `--output docs/evidence/ai-engineering-review-20261028/explainability/explainability-review-20261028.json`. Apart from `timing_seconds` and `provenance`, a rerun with the same versions gave the same numbers on this machine.

## 9. Amendments and run history

The protocol was written before the review code existed. Three amendments are recorded in `protocol.json`; none changes a threshold, parameter, case or verdict rule:

1. After an API probe on three test rows, an independent brute-force enumeration and a threshold-tie diagnostic were added to C1b.
2. After a reduced smoke run (whose numbers are not reported), the rescaled-margin reconstruction was added to the tree diagnostic.
3. After a first protocol-sized run, the model-specific additivity field was changed to use each explainer's own base value. The retained run is identical to the first one except for the changed fields; the TreeExplainer base equals the Exact base within 2.2e-16.
