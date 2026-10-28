# Explainability experiments and acceptance gate reference

This guide explains how the synthetic invoice experiments produce a score and
explanations, and how the laboratory gate evaluates recorded evidence. It covers
five source files and every named function, class and nested helper in them.
The source links below open the relevant definition.

The important separation is between **executing an experiment**, **checking its
numerics**, and **approving an operational service**. These functions perform the
first two. The gate always returns `production_approved=False`; domain review and
an authorized human decision remain separate.

## Reading path

1. Start with `invoice_schema` and `validate_invoice` to understand the six numeric
   inputs and their allowed units and ranges.
2. Read `run_explainability_lab` for one logistic-regression score and its SHAP and
   LIME explanations. Its default input is a held-out synthetic row.
3. Read `run_explainability_comparison` for the same-split linear/tree comparison.
   Its default displayed input is a fixed fictional high-score example, without
   an observed ground-truth label.
4. Follow the fitting and explanation helpers to see the calculation details.
5. Read `evaluate_candidate_evidence` to see how recorded results can block a
   candidate while successful execution and human approval remain distinct.
6. Read the test functions to see which errors are exercised deliberately.

## Components and boundaries

| File | Responsibility | Observable effect |
|---|---|---|
| `lab/engineering_explainability.py` | Generate fictional inputs, fit classifiers, explain one score, and return measurements | CPU computation and an in-process cache; no explicit report-file, registry or network operation |
| `lab/engineering_acceptance.py` | Evaluate trusted in-memory report values and compute their hashes | Returns a technical verdict; no model calls or external state changes |
| `scripts/measure_explainability_engineering.py` | Command-line wrapper for the experiment/comparison | Prints JSON and optionally creates parent directories and writes/replaces the requested output file |
| `tests/test_engineering_explainability.py` | Input, import-isolation and numerical checks with optional real scientific libraries | Some tests execute bounded real CPU experiments; an import test starts an isolated Python subprocess |
| `tests/test_engineering_acceptance.py` | Gate behavior over artificial report fixtures | Mutates independent in-memory test data; does not claim an inference/load experiment occurred |

SHAP means SHapley Additive exPlanations and attributes an output to its inputs.
LIME means Local Interpretable Model-agnostic Explanations and fits a simpler model
near one input. Expanding the names does not establish their validity: the units,
background, numerical checks and limits below explain what the results mean.

## Invoice input contract

`validate_invoice` accepts a mapping containing exactly these six keys. Its result
is a list of floats in the order shown. No invoice files or serialized models are
accepted by this interface.

| Key | Unit and domain | Inclusive bounds |
|---|---|---|
| `amount_ratio` | Invoice amount divided by supplier median, unitless | 0.05 to 12.0 |
| `purchase_order_match` | Binary flag | 0 or 1 |
| `supplier_history_days` | Number of days, numeric within the teaching domain | 0 to 3650 |
| `duplicate_reference` | Binary flag | 0 or 1 |
| `delivery_confirmed` | Binary flag | 0 or 1 |
| `recent_bank_change` | Binary flag | 0 or 1 |

Finite real numbers are required. Booleans are permitted only for binary fields.
Missing/extra keys, numeric strings, invalid ranges and fractional binary values
raise `ValueError`. These bounds describe the exercise; they are not approved
business rules for real invoice processing.

## Running one explanation

The public parameters are keyword-bounded: `seed` is 0 through `2**31 - 1`,
`n_samples` is 200 through 5000, and `lime_samples` is 500 through 20000.
`case_index` must identify a row in the 25-percent held-out split, even when the
caller supplies an explicit invoice. The normal defaults are seed 1729, 800 total
rows, 4000 LIME fitting perturbations and held-out index 0.

The model is a training-fitted `StandardScaler` followed by
`LogisticRegression`. Generation uses a seeded probabilistic linear rule; the
stratified 75/25 split keeps training and evaluation indices separate. Numerical
work is limited to one thread. Fitted baseline and tree bundles each have an
`lru_cache` of at most four entries, keyed by seed/sample count. Their mutable
arrays and estimators are implementation objects to treat as read-only.

The result contains:

- `environment` and `parameters`, including package versions, CPU device and seed.
- `training`, with the dataset fingerprint, model parameters, split counts/overlap
  and held-out accuracy, ROC AUC and Brier score.
- `case`, with the input values, class-1 probability, 0.5 threshold, predicted
  label, and an observed synthetic label only when a held-out row was selected.
- `shap` and `lime`, with their different output units and diagnostics.
- `checks`, `quality_assessment`, `timing_seconds` and explicit limitations.

`ok=True` and `execution_ok=True` describe an executed calculation. They do not
replace the separate LIME diagnostics or validate a real operational use.

## What each explanation actually measures

### Linear SHAP

The linear model and background are expressed in standardized feature space.
Up to 80 training rows, selected with `seed + 1`, form an independent masker.
The base value plus six SHAP contributions reconstructs the class-1 margin in
**log-odds**, checked with absolute tolerance `1e-8`. Applying the sigmoid gives
a probability, compared with the model at `1e-10` tolerance by the public helper.

`sigmoid(base)` is recorded separately from the mean probability over the
background; these need not be equal. An independent background does not establish
a causal effect or resolve all correlated-feature interpretation questions.

### Tree SHAP

The comparison fits `HistGradientBoostingClassifier` on exactly the baseline's
training rows and evaluates it on the same held-out rows. `TreeExplainer` uses
`model_output="raw"` and a path-dependent background from the fitted trees.
The helper checks the six-value shape, reconstructs the model margin within
`1e-8`, and reports the sigmoid conversion. The report sorts positive contributions
from largest downward and negative contributions from most negative upward.

The tree and linear backgrounds differ. Comparing their individual contribution
sizes is not a direct causal comparison between models.

### LIME

LIME explains the **linear baseline's class-1 probability**. Binary fields are
categorical; the continuous fitting variables are not discretized. The report
keeps the surrogate's raw local output, including values outside `[0, 1]`.
It never silently clips a failed surrogate into a plausible probability.

There are two different diagnostic scopes:

- `fit_neighbourhood_weighted_r2` is LIME's score on its own fitting perturbations.
  It is not independent validation or global classifier accuracy.
- `independent_neighbourhood` evaluates 256 new artificial points, generated with
  `seed + 9301`. Continuous inputs receive bounded jitter and binary inputs
  occasional flips. LIME's scaling and distance kernel weight the comparison.
  The report gives probability MAE, RMSE, weighted R2 and the number of outputs
  outside the probability range.

The public helper's illustrative case gate requires an in-range output and an
absolute difference at most 0.05 from the classifier. This teaching threshold is
not approved business acceptance, a stability study or a representative-data test.

## Reading the model comparison

`run_explainability_comparison` embeds the unchanged single-model report under
`linear_baseline`, adds `linear_coefficients`, `tree_model`, `model_comparison`
and numerical `checks`, and always states `production_approval=False`.

Original-unit coefficients divide the standardized coefficients by the training
standard deviation. The original-unit intercept subtracts coefficient times
training mean. The history explanation uses a 365-day increment; the other inputs
use one original unit. Exponentiating an increment's log-odds change gives its odds
multiplier, holding other inputs fixed.

The deliberately high-score comparison preserves a failed linear-model LIME
surrogate as a diagnostic counterexample while Tree SHAP reconstructs the tree
margin. They explain **different fitted models**. Passing Tree SHAP addition and
failing a LIME range check therefore do not rank the methods on the same model.
The fixed synthetic split and linear label-generating rule cannot establish
superiority of a model family on real data.

## Pure evidence gate

The manifest accepts only `rag`, `vision` and `explainability` top-level keys.
Present values must be dictionaries/lists already loaded by a trusted caller.
The evaluator never opens supplied paths or URLs and never authenticates the
source of a report. Canonical UTF-8 JSON uses sorted keys, compact separators and
finite numbers, with a maximum size of 512 KiB. SHA-256 hashes identify those
content values; they are not signatures or proofs of measurement authenticity.

| Topic | What is checked | What remains outside this function |
|---|---|---|
| Retrieval-augmented generation (RAG) | Bounded case-scoped attempts/sources, nonrepeated query trace, exact numbered quotes, trusted per-trial outcome checks, and a required separate multi-step case | Re-running inference, proving entailment, representative coverage and completeness of all relevant runs |
| Vision | Finite ordered millisecond timing values, request accounting for an open-loop report, trained-weight metadata and held-out quality against the recorded threshold | A real object-detection deployment, realistic peak capacity, business latency acceptance and independent report authenticity |
| Explainability | Reported SHAP reconstruction error/tolerance at most `1e-8`; LIME's raw in-range probability and recomputed error against a recorded threshold at most 0.05 | Recomputing SHAP shares, independent LIME neighbourhood/stability acceptance, causality and usefulness to domain experts |
| Governance | Explicit pending human/domain/operational review entries | Recording or granting a person's production decision, registry release or regulatory/business approval |

RAG's `refinement` expectation preserves the historical requirement of more than
one search. `paraphrase` accepts a correct answer in one to three searches. The
separate `multi_step` protocol requires more than one search and passing trusted
expected-outcome observations. A missing or failed multi-step test blocks the
candidate; a valid quote from a different object cannot override a recorded failed
outcome. This helper does not independently infer whether a quote is relevant.

The report schema is:

- `schema_version`: 1.
- `decision`: `blocked` if any check has status `fail`, otherwise
  `technical_candidate`.
- `manual_review_required`: always true.
- `production_approved`: always false.
- `checks`: ordered `id`, `status`, `detail` dictionaries. `review_pending` entries
  preserve outstanding work and do not themselves grant or deny human approval.
- `evidence_hashes`: per-topic canonical content hashes.
- `limitations`: trust, completeness and release boundaries.

Input-contract errors raise `ValueError`; failed or missing measurements normally
become failed checks. All supplied explanation cases are considered, including
known failures, within the one-to-eight-case bound.

## Command-line report wrapper

After activating the separately prepared environment with the scientific extras,
run the script from the repository root, for example:

```text
python scripts/measure_explainability_engineering.py --compare --output results/explanation-comparison.json
```

`--invoice-json` optionally supplies the six numeric fields. `--output` creates
parent directories and writes the exact JSON also sent to stdout, replacing an
existing output file. Choose a new filename when preserving earlier evidence.

| Exit code | Meaning |
|---|---|
| 0 | All returned numerical `checks` are true |
| 1 | At least one returned numerical check is false |
| 2 | A caught dependency, `ValueError` or JSON input failure is represented in an `ok=False` error object; argparse also uses status 2 for invalid CLI usage |

Other import, scientific, serialization and filesystem errors can propagate.
A comparison with a known failing LIME diagnostic can still exit 0 when its
numerical reconstruction checks pass. The wrapper is not the acceptance gate.

## Function and class index

Every definition has an inline docstring describing its actual inputs, output,
side effects, failure behavior and limits. The following links identify the
current source definitions; helper entries nested inside the gate share its
local evaluation state.

### `lab/engineering_explainability.py`

| Definition | Purpose |
|---|---|
| [ExplainabilityUnavailable](../../lab/engineering_explainability.py#L40) | Signal that optional explainability distributions are unavailable. |
| [invoice_schema](../../lab/engineering_explainability.py#L50) | Return independent copies of the six demonstration field definitions. |
| [explainability_status](../../lab/engineering_explainability.py#L62) | Inspect distribution metadata for the optional local experiment. |
| [validate_invoice](../../lab/engineering_explainability.py#L89) | Validate one synthetic input and order its values for the classifiers. |
| [_bounded_int](../../lab/engineering_explainability.py#L127) | Validate and return an integer resource or reproducibility parameter. |
| [_dependencies](../../lab/engineering_explainability.py#L148) | Lazily load the scientific implementations used by requested runs. |
| [_fit_synthetic_model](../../lab/engineering_explainability.py#L184) | Generate one synthetic dataset and fit the cached linear baseline. |
| [_fit_synthetic_tree_model](../../lab/engineering_explainability.py#L242) | Fit a cached boosted-tree classifier on the baseline's exact split. |
| [_linear_coefficients](../../lab/engineering_explainability.py#L277) | Translate the fitted logistic rule into original feature units. |
| [_explain_tree_shap](../../lab/engineering_explainability.py#L325) | Explain one boosted-tree margin and check numerical reconstruction. |
| [_explain_shap](../../lab/engineering_explainability.py#L381) | Explain a logistic margin with a training-only independent background. |
| [_explain_lime](../../lab/engineering_explainability.py#L443) | Fit a local probability surrogate and evaluate a separate neighbourhood. |
| [run_explainability_lab](../../lab/engineering_explainability.py#L537) | Fit a synthetic logistic baseline and explain one score with SHAP/LIME. |
| [run_explainability_comparison](../../lab/engineering_explainability.py#L650) | Compare a linear baseline and boosted trees on the same synthetic split. |

### `lab/engineering_acceptance.py`

| Definition | Purpose |
|---|---|
| [_number](../../lab/engineering_acceptance.py#L16) | Return whether value is a finite built-in int or float, excluding bool. |
| [_object](../../lab/engineering_acceptance.py#L25) | Normalize an optional report node to a dictionary for guarded lookup. |
| [_canonical](../../lab/engineering_acceptance.py#L30) | Serialize one evidence value deterministically and enforce its size limit. |
| [evaluate_candidate_evidence](../../lab/engineering_acceptance.py#L54) | Check supplied lab reports and return a non-authorizing technical verdict. |
| [evaluate_candidate_evidence.check](../../lab/engineering_acceptance.py#L73) | Append one pass/fail measurement result to this evaluation's local list. |
| [evaluate_candidate_evidence.pending](../../lab/engineering_acceptance.py#L83) | Append a human or operational review requirement without approving it. |
| [evaluate_candidate_evidence.shap_check](../../lab/engineering_acceptance.py#L248) | Check reported SHAP reconstruction diagnostics for the current case. |

### `scripts/measure_explainability_engineering.py`

| Definition | Purpose |
|---|---|
| [main](../../scripts/measure_explainability_engineering.py#L21) | Parse CLI options, execute the selected experiment and emit its report. |

### `tests/test_engineering_explainability.py`

| Definition | Purpose |
|---|---|
| [result](../../tests/test_engineering_explainability.py#L21) | Provide one module-scoped real default explanation report to tests. |
| [valid_invoice](../../tests/test_engineering_explainability.py#L36) | Return a fresh six-field fictional positive-score input dictionary. |
| [test_module_import_does_not_load_optional_scientific_packages](../../tests/test_engineering_explainability.py#L45) | Verify lazy imports in a fresh isolated Python subprocess. |
| [test_invalid_inputs_are_rejected_before_computation](../../tests/test_engineering_explainability.py#L58) | Reject each parametrized schema violation without fitting a model. |
| [test_missing_and_unexpected_fields_are_rejected](../../tests/test_engineering_explainability.py#L74) | Reject a missing feature combined with an unrecognized file_path key. |
| [test_shap_reconstructs_the_model_in_the_declared_output_space](../../tests/test_engineering_explainability.py#L87) | Check default Linear SHAP addition and its probability conversion. |
| [test_both_explanations_target_the_same_score_and_class](../../tests/test_engineering_explainability.py#L104) | Check class alignment, held-out separation and the default LIME diagnostic. |
| [test_local_lime_fidelity_and_explanation_are_explicit](../../tests/test_engineering_explainability.py#L131) | Require explicit finite LIME outputs and independent-neighbourhood metadata. |
| [test_same_seed_reproduces_the_explanations](../../tests/test_engineering_explainability.py#L155) | Repeat the default run and compare deterministic scientific outputs. |
| [test_invalid_case_or_cost_parameters_are_bounded](../../tests/test_engineering_explainability.py#L172) | Reject sample, perturbation, seed and index values outside public limits. |
| [comparison](../../tests/test_engineering_explainability.py#L184) | Provide a module-scoped real linear/tree comparison for one fictional input. |
| [test_tree_shap_reconstructs_a_real_positive_margin_on_the_same_split](../../tests/test_engineering_explainability.py#L199) | Check a fitted tree explanation and preserve the failed LIME counterexample. |
| [test_linear_coefficients_recover_raw_unit_predictions](../../tests/test_engineering_explainability.py#L230) | Verify coefficient conversion from standardized features to original units. |

### `tests/test_engineering_acceptance.py`

| Definition | Purpose |
|---|---|
| [fixture_manifest](../../tests/test_engineering_acceptance.py#L10) | Build a fresh artificial manifest satisfying the technical gate branches. |
| [statuses](../../tests/test_engineering_acceptance.py#L43) | Index an evaluator's recorded check statuses by their stable ids. |
| [test_passing_technical_evidence_still_requires_separate_human_approval](../../tests/test_engineering_acceptance.py#L48) | Keep approval separate even when the artificial technical fixture passes. |
| [test_fast_random_weight_benchmark_does_not_pass_quality_or_release](../../tests/test_engineering_acceptance.py#L57) | Block random-weight timing evidence that omits training and quality facts. |
| [test_open_loop_probe_accounting_and_latency_do_not_release_random_weights](../../tests/test_engineering_acceptance.py#L70) | Validate open-loop accounting separately from model-quality acceptance. |
| [test_quality_failure_is_not_hidden_by_fast_timing_or_global_ok](../../tests/test_engineering_acceptance.py#L92) | Reject low held-out accuracy despite valid timing and a global ok flag. |
| [test_failed_lime_case_blocks_even_if_report_claims_success](../../tests/test_engineering_acceptance.py#L103) | Reject an out-of-range LIME surrogate despite optimistic report flags. |
| [test_all_supplied_explanation_cases_are_evaluated_without_hiding_failure](../../tests/test_engineering_acceptance.py#L114) | Retain a failure in the second of two supplied explanation reports. |
| [test_tree_comparison_checks_tree_shap_reconstruction_numerically](../../tests/test_engineering_acceptance.py#L127) | Reject a tree comparison whose reported SHAP error exceeds its tolerance. |
| [test_rag_counter_or_self_rating_does_not_override_failed_evidence](../../tests/test_engineering_acceptance.py#L140) | Block each parametrized RAG defect despite the inflated aggregate counter. |
| [test_missing_evidence_blocks_and_approval_fields_or_paths_are_rejected](../../tests/test_engineering_acceptance.py#L160) | Handle missing evidence as failure and reject unsupported manifest contracts. |
| [test_hashes_are_stable_for_key_order_and_change_with_evidence_without_file_io](../../tests/test_engineering_acceptance.py#L169) | Verify canonical content hashing and the evaluator's no-file-I/O boundary. |
| [test_nonfinite_or_oversized_manifest_is_rejected](../../tests/test_engineering_acceptance.py#L190) | Reject unsafe canonical-JSON values before evaluating individual evidence. |
| [two_step_trial](../../tests/test_engineering_acceptance.py#L202) | Return the mutable multi-step trial within an artificial manifest. |
| [test_missing_two_step_case_blocks_even_when_the_three_basic_cases_pass](../../tests/test_engineering_acceptance.py#L207) | Require a separate multi-step case instead of accepting only basic questions. |
| [test_wrong_two_step_answer_blocks_although_its_quote_is_exact](../../tests/test_engineering_acceptance.py#L217) | Block a failed multi-step outcome while preserving valid quoted text. |
| [test_two_step_answer_after_a_single_search_does_not_pass](../../tests/test_engineering_acceptance.py#L230) | Enforce the predefined loop-test requirement of more than one search. |
| [test_paraphrase_rule_accepts_one_search_but_historical_refinement_rule_does_not](../../tests/test_engineering_acceptance.py#L238) | Keep the historical loop rule distinct from the newer paraphrase answer rule. |

## Verification and study cautions

The explanation tests deliberately retain one failed LIME example; a passing
pytest result means the failure was detected and preserved. The acceptance tests
use wholly artificial manifests, including optimistic flags and an inflated
aggregate counter, to exercise rejection paths. They do not repeat the recorded
lab measurements or prove that synthetic fixture values describe a deployed model.

Tests also cover lazy import in an isolated interpreter, same-seed outputs in one
environment, invalid schema/cost options, exact coefficient algebra, canonical
hash order, no file-open calls by the gate, missing evidence, broken open-loop
request accounting, random weights, failed explanation cases and the distinct
paraphrase/multi-step rules. Review the individual test docstrings for each
fixture and expected outcome.

There are no production credentials, external model calls, model downloads or
MLflow tracking/registry operations in this documented path. A synthetic score
supports learning about engineering controls; it is never proof of wrongdoing.
