# Recorded results and source preservation

## Reference result

[`results/baseline_20261110T161422_601162Z/summary.json`](../results/baseline_20261110T161422_601162Z/summary.json)
is the reference run: 256,351 rows, 67 input features, five shuffled stratified
folds, seed 42, pooled OOF ROC-AUC **0.7392190176**, average precision
0.4486918168. Input feature counts precede one-hot encoding.

Its recorded `baseline_cv_sha256` equals the SHA-256 of the
[`baseline_cv.py`](../baseline_cv.py) published in this repository, so the
executed source can be checked directly:

```bash
shasum -a 256 baseline_cv.py   # fb0d971a315d7f5a2e0811572efa8ff4d58e250bddb1a8b30b2a4b60943a0531
```

The run was made before the repository's first commit, so it records no commit
(`commit: null`, `dirty: true`); the source hash is the identifier. It also records the
dataset hash, prediction-file hash and Python/package versions.

Some earlier records (the standalone full run and the missingness comparison)
contain a `commit` value from a private working history that was not published
with this repository. That commit cannot be looked up here; use the recorded
source hashes instead, and for the missingness comparison the preserved source copies.

## Repeatability

Earlier full runs produced with slightly older source versions (input validation
has since been tightened) give the same pooled metrics. The reference run's OOF
prediction file has the same SHA-256 as the
[earlier standalone run](../results/baseline_20261110T144314_638320Z/summary.json)
and the baseline variant of the
[missingness comparison](../results/missingness_20261110T144629_494691Z/comparison.json),
whose source copies are [preserved with a hash manifest](../results/missingness_20261110T144629_494691Z/source/manifest.json).
These checks show local repeatability on one macOS arm64 machine; they do not
establish cross-platform determinism, external validity or patient-level
independence.

| Record | Interpretation |
|---|---|
| [Reference run](../results/baseline_20261110T161422_601162Z/summary.json) | Current source; identical OOF predictions to the runs below. |
| [Standalone full run](../results/baseline_20261110T144314_638320Z/summary.json) | Earlier source version; identical predictions. |
| [Earlier full summary](../results/baseline_cv_full.json) | Same metrics; lacks environment/source provenance. |
| [5,000-row sample](../results/baseline_cv_sample_5000.json) | Three-fold smoke run; reproduced exactly by the current source. Not comparable with the full baseline. |

## Comparison with the competition pipeline

The README states only that the competition-time pipeline as a whole (in-fold SMOTE,
OOF AUC 0.719–0.730) scored below this baseline (0.7392). The gap cannot be attributed to
preprocessing alone, because the two set-ups differ in at least four places:

| | Competition pipeline (`evaluate_cv.py`, in-fold SMOTE) | Baseline (`baseline_cv.py`) |
|---|---|---|
| Preprocessing | Shared GAIN cache with target-based feature selection, learned before the split | Fold-fitted median imputation and one-hot encoding of all raw features |
| Hyperparameters | Tuned against the leaky score (~0.89), `configs/*_params.leaky.json` | Untuned fixed values (250 trees, learning rate 0.05, 31 leaves) |
| Resampling | SMOTE on the training part of each fold | None |
| Training size | 20% of each training fold held out for threshold search, so 64% of all rows | 80% of all rows |

**Fold assignment.** Both scripts use `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)`, whose split depends only on the number of rows and the label order.
The rebuilt GAIN cache (2026-11-14) and `data/train.csv` have identical label order, and
the two splits were checked to produce identical validation indices for all five folds.
The competition-time `train_df_imputed.pkl` behind `cv_results.json` (256,351 rows ×
90 features plus the target) is kept locally and not published, because it is derived
from the competition data. On 2026-11-15 its label order was checked to be identical to
`data/train.csv`, so that earlier run used the same five validation folds as well.

**Re-measurement of the SMOTE comparison.**
[`results/smote_placement_20261114.json`](../results/smote_placement_20261114.json)
reruns `evaluate_cv.py` with the competition-time parameters
(`--params-dir` pointing to copies of `configs/*_params.leaky.json`) on a GAIN cache
rebuilt by the current `build_cache.py` (256,351 rows × 104 features). Package versions:
scikit-learn 1.4.1.post1, xgboost 2.0.3, lightgbm 4.3.0, catboost 1.2.5,
imbalanced-learn 0.12.0, torch 2.2.1 (GAIN), numpy 1.26.4, pandas 2.2.1.

| Model | SMOTE before split | SMOTE in fold | Gap |
|---|---:|---:|---:|
| XGBoost | 0.9051 | 0.7232 | 0.182 |
| LightGBM | 0.9047 | 0.7296 | 0.175 |
| CatBoost | 0.9032 | 0.7199 | 0.183 |
| Random Forest | 0.9049 | 0.7289 | 0.176 |

The gaps match those in `cv_results.json` (0.178–0.186). This is a single-seed run; the
in-fold column still inherits the shared-cache leakage described in the
[validation audit](VALIDATION_AUDIT.md).

The two protocols differ in more than SMOTE placement. The leaky column cross-validates
SMOTE-resampled rows from an 80% split, so its validation folds contain synthetic rows,
and it reports the mean of fold AUCs. The in-fold column evaluates original validation
rows, trains on 64% of all rows after the calibration split, and reports pooled OOF AUC.
The gap therefore cannot be attributed solely to SMOTE placement.

## Legacy report index

These original reports are preserved without rewriting their metrics or metadata.
They describe earlier experiments with preprocessing learned before downstream
validation. They are not current baseline results or independently validated
performance estimates; the [validation audit](VALIDATION_AUDIT.md) applies.
Historical labels such as `honest`, `leaky` and `referee` identify comparisons in
that workflow, not the present validation status of the full pipeline. References
to `configs/*.json` in old metadata describe the repository state at that time.

| Report | Historical experiment |
|---|---|
| [cv_results.json](../cv_results.json) | SMOTE before CV versus within training folds; `honest` names only the latter estimator-stage protocol. |
| [results/smote_placement_20261114.json](../results/smote_placement_20261114.json) | The same comparison re-run on a rebuilt cache; see [above](#comparison-with-the-competition-pipeline). |
| [ensemble_study.json](../ensemble_study.json) | Single models and ensemble variants on the shared representation. |
| [resampling_study.json](../resampling_study.json) | Resampling and class weighting comparisons. |
| [target_encoding_study.json](../target_encoding_study.json) | Age encoding variants; small score differences do not establish the absence of leakage. |
| [column_study.json](../column_study.json) | Individual and combined categorical-column variants. |
| [tuning/referee.json](../tuning/referee.json) | Historical parameter adoption comparison downstream of shared preprocessing. |

## Recreating and inspecting artifacts

Raw data and OOF CSVs are local-only; a fresh clone contains summaries and source
copies, not the individual predictions. Follow [reproduction instructions](REPRODUCE.md)
to obtain data and generate a new timestamped run. Each new run records its own
source hash and Git state.

For controlled comparisons use identical data, folds and model settings, and
report paired differences. Do not substitute a legacy cache-based result for
this reference. The [missingness experiment](experiments/MISSINGNESS.md) is a
single-seed component comparison, with limitations stated alongside its results.
