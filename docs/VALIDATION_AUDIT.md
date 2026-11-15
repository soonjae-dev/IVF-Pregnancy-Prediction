# Validation audit — 2026-11-10

## Scope and evidence

Reviewed the legacy pipeline code and saved baseline OOF outputs. This audit verifies code paths and saved metrics. The full baseline was rerun
with provenance on 2026-11-10 and all saved predictions matched the earlier run.
Historical complex-pipeline experiments were not rerun; clinical validity is untested.

| Finding | Evidence | Consequence / status |
|---|---|---|
| Target-based selection precedes validation | `build_cache.build` calls `run_feature_selection`; `src/feature_selection.py` computes correlation against all training labels | Validation labels influence the feature set used by downstream CV. Unresolved in the legacy cache path. |
| Learned imputation precedes validation | `build_cache.build` calls `run_gain_on_dataframes` before storing the matrix consumed by `evaluate_cv.load_dataset`, ensemble/resampling studies, and tuning | Validation features participate in representation learning. Removing the target from GAIN does not make this train-only preprocessing. |
| Train/test GAIN models are fitted separately | `src/run_imputation.py` calls `gain_impute` independently on train and test | This is not a reusable training-fitted imputer applied to unseen data. Requires an explicit fit/transform design for inductive evaluation. |
| Referee split is downstream of preprocessing | `tune.py` and `validate_tuning.py` read the shared cache | The split separates model search from scoring but does not undo feature-selection/imputation contamination. |
| Fold-local baseline exists | `baseline_cv.run_cv` creates a new pipeline from each training fold | Median statistics, category vocabulary and model fitting exclude the corresponding validation rows. |
| OOF files agree with summaries | Both 5,000-row and 256,351-row files have unique complete row positions and finite probabilities in [0,1]; recomputed pooled AUC/AP/F1 match JSON | Supports saved artifact consistency; does not independently prove training provenance. |

## Corrections to interpretation

A target encoding learned from validation labels remains leakage even if an
observed AUC difference is small or negative. An earlier version of the README
called it “not a leak”; that statement is withdrawn. Likewise, similar scores alone do not
prove model equivalence, and ROC-AUC's threshold invariance does not imply that
resampling cannot change ranking performance.

Historical claims that GAIN, ensembles, or resampling help or hurt need a
controlled fold-fitted comparison. No numerical leakage effect is attributed to
the remaining cache problems without such an experiment.

## Verified fixes and limits

| Fix | Code | Regression test |
|---|---|---|
| GAIN reconstruction loss computed on observed entries only | [`src/gain_imputer.py`](../src/gain_imputer.py) | [`test_gain_reconstruction.py`](../tests/test_gain_reconstruction.py) |
| Generator output detached during discriminator training | [`src/gain_imputer.py`](../src/gain_imputer.py) | None (code inspection only) |
| Final imputation pass in evaluation mode (no dropout, BatchNorm running statistics; added 2026-11-15) | [`src/gain_imputer.py`](../src/gain_imputer.py) | [`test_review_fixes.py`](../tests/test_review_fixes.py) |
| Cache reused only when data, source and build options match (added 2026-11-15) | [`build_cache.py`](../build_cache.py) | [`test_review_fixes.py`](../tests/test_review_fixes.py) |
| Adversarial loss computed on missing entries | [`src/gain_imputer.py`](../src/gain_imputer.py) | [`test_gain_adversarial.py`](../tests/test_gain_adversarial.py) |
| Fold-fitted LightGBM baseline | [`baseline_cv.py`](../baseline_cv.py) | [`test_baseline_cv.py`](../tests/test_baseline_cv.py) |

The synthetic-data suite runs in [CI](../.github/workflows/tests.yml).
See [recorded runs](RESULTS.md) for the source hash recorded by each run.
The documentation cleanup did not rerun GAIN studies.

These tests do not show a predictive benefit from corrected GAIN.

The baseline's 0.739219 pooled OOF ROC-AUC is the current reference result.
The exploratory full-OOF best F1 (approximately 0.5169 at threshold 0.2542) is
not an independently validated performance estimate or the competition objective.

## Remaining work

- New runs now record data/artifact hashes, source revision/dirty state, model settings
  and environment versions in unique directories. Historical summaries still lack this provenance.
- New OOF files include `source_row_index` for original CSV data-row positions.
  `row_index` continues to identify positions within the evaluated sample.
- Check group/time dependence and feature availability at the intended prediction
  time using dataset documentation; no patient-level or temporal claim is made.
- Refit every learned transformation within outer training folds for future
  comparisons; use inner folds for supervised selection or hyperparameter search.
- Keep legacy metrics labeled historical until revalidated. The legacy training
  and study scripts themselves have not been repaired by this documentation change.
