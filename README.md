# Predicting Pregnancy Success for Infertility Patients

**English** | [한국어](README.ko.md)

[![Synthetic-data tests](https://github.com/soonjae-dev/IVF-Pregnancy-Prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/soonjae-dev/IVF-Pregnancy-Prediction/actions/workflows/tests.yml)

> **Summary**
>
> - **Diagnosis**: My competition result was in the lower half (Private 544/794, AUC 0.714). Re-running the pipeline after the competition showed that applying SMOTE before cross-validation inflated validation AUC by about 0.18, and that GAIN imputation and feature selection had also seen the validation rows ([5-2](#5-2-the-089-tuning-score-was-smote-leakage), [8](#8-retrospective)).
> - **Rebuild**: A baseline that learns only inside each fold, records data and source hashes, and reproduces its predictions byte for byte (cross-validation AUC 0.7392, [5-4](#5-4-a-leak-free-baseline)).
> - **What moves the score**: On the same folds, changing the missing-value handling, even type by type, moved AUC by at most 0.0003 (GAIN fitted inside each fold has not been compared yet), and hand-made ratio features lowered it. Nested tuning and a LightGBM–CatBoost rank blend raised it in every fold, to 0.7405 ([5-6](#5-6-what-actually-raised-the-score), [5-7](#5-7-handling-each-type-of-missing-value)).
> - **Limit**: Every post-competition score is cross-validation on the training data. Submissions are closed, so how these models would rank on the leaderboard cannot be checked, and they are not compared with leaderboard ranks here. The inputs include columns known only after treatment has progressed (such as the number of embryos transferred), so this is not the accuracy of a pre-treatment prediction; among rows with an embryo transferred, AUC is about 0.68 ([5-8](#5-8-where-the-model-is-right-and-where-it-fails)).

## 1. Project Overview

<p align="center">
  <img src="docs/images/competition_banner.png" alt="Banner of the online hackathon on predicting pregnancy success for infertility patients" width="100%">
</p>

**Task**

The LG Aimers 6th online hackathon (DACON) asked participants to predict pregnancy
success from infertility treatment records and to identify the factors that drive it.

**Competition result** (metric: ROC-AUC)

| | Public | Private |
|---|---:|---:|
| My score | 0.71516 | 0.71446 |
| Rank | 545 / 794 | 544 / 794 |
| 1st place | 0.74350 | 0.74377 |

## 2. Dataset Overview

| Item | Value |
|---|---|
| Training data | 256,351 rows × 69 columns (including ID and target) |
| Test data | 90,067 rows × 68 columns |
| Input features | 67 (20 categorical, 47 numeric) |
| Target | `임신 성공 여부` (pregnancy success), 25.8% positive |
| Missing values | 10.3% of input-feature cells, in 31 columns |

The figures above describe the raw training data before any preprocessing. How to obtain
the data is described in [docs/DATA.md](docs/DATA.md); it is not redistributed here. The
analysis of missing values and feature distributions was done after the competition and
is in [5-3](#5-3-post-competition-eda-and-what-the-missing-values-mean).

## 3. Preprocessing & Feature Engineering at the Time of the Competition

> Sections 3 and 4 describe the competition-time approach. The leak-free baseline rebuilt after the competition is in [5-4](#5-4-a-leak-free-baseline).

The competition notebook is [`notebooks/archive/baseline_modeling.ipynb`](notebooks/archive/baseline_modeling.ipynb).

**Derived features**

- Pregnancies per treatment
- Mean pregnancy rate by age group

**Preprocessing**

- **Feature selection**: dropped features whose absolute correlation with the target was
  below 0.02, and dropped all categorical columns except one string column.
- **Imputation**: implemented GAIN, a generative imputation method, in PyTorch. I did not
  check why the values were missing.
- **Class imbalance**: the positive rate is about 26%, so SMOTE oversampled to 50:50.

Some of these steps pulled validation information into training. What went wrong is
summarised in [section 8](#8-retrospective).

## 4. Modeling & Validation Strategy at the Time of the Competition

- **Models**: XGBoost, LightGBM, CatBoost, Random Forest, TabTransformer
- **Tuning**: Optuna search on cross-validation score for each model
- **Ensemble**: stacking and a weighted average, with weights searched by Optuna
- **Threshold**: search for the F1-maximising classification threshold
- **Validation**: 20% of the training data held out as a validation set

## 5. Post-competition Analysis & Improvements

### 5-1. Scores during the competition

| Stage | ROC-AUC |
|---|---:|
| Cross-validation during tuning | ~0.89–0.91 |
| Validation set (20% holdout) | 0.760 |
| Private leaderboard | 0.714 |

The score dropped sharply at each stage. I strongly suspected validation leakage and
tested it below.

### 5-2. The 0.89 tuning score was SMOTE leakage

Hyperparameters were chosen by cross-validating on the SMOTE-augmented training data.
SMOTE creates synthetic rows by interpolating between neighbouring samples, so the
training folds contained synthetic rows derived from samples in the validation fold.

After the competition I moved the pipeline into modules and, keeping the
competition-time parameters, compared two evaluation protocols
([`evaluate_cv.py`](evaluate_cv.py)). They differ in more than where SMOTE is applied:

- **SMOTE before split (leaky)**: SMOTE is applied to 80% of the training data and that
  resampled data is cross-validated. Validation folds also contain synthetic rows, and the
  mean of the fold AUCs is reported.
- **SMOTE in fold**: the original data is cross-validated and SMOTE is applied only inside
  each training fold. 20% of each training fold is held out for threshold search, so the
  models train on 64% of all rows, and pooled OOF AUC on the original validation rows is
  reported.

I measured this comparison twice.

- **First measurement**: on the competition-time GAIN-imputed data
  (`train_df_imputed.pkl`, 90 features) ([record](cv_results.json)). The pkl is kept
  locally and not published, because it is derived from the competition data.
- **Re-measurement**: on a GAIN cache rebuilt with the code as of 2026-11-14 (104 features,
  after the GAIN loss fixes and before the later evaluation-mode fix), running the same
  comparison again
  ([record](results/smote_placement_20261114.json)). The extra features come from a change
  in categorical handling after the competition, not from GAIN: nine categorical columns
  that were discarded at the time (egg source, treatment period code and others) are now
  one-hot encoded, and the leaky age-group success-rate feature was replaced by an
  age-group one-hot.

| Model | SMOTE before split (first) | SMOTE in fold (first) | SMOTE before split (re-run) | SMOTE in fold (re-run) |
|---|---:|---:|---:|---:|
| XGBoost | 0.906 | 0.721 | 0.905 | 0.723 |
| LightGBM | 0.905 | 0.726 | 0.905 | 0.730 |
| CatBoost | 0.904 | 0.719 | 0.903 | 0.720 |
| Random Forest | 0.909 | 0.723 | 0.905 | 0.729 |

With different input data, the gap between the two protocols stayed almost the same,
0.175–0.186. The re-measurement had already dropped the leaky age-group success-rate
feature, yet the gap did not change, so the 0.18 does not come from that feature. Both
measurements reuse a shared GAIN cache built before the split, so the "SMOTE in fold"
columns are not leak-free either. In both, GAIN and the label-based feature selection
were fitted on all rows before the split ([8-1](#8-1-why-the-validation-score-was-optimistic)).
The first measurement's cache also had the target inside GAIN
([8-2](#8-2-training-and-test-data-were-processed-differently)); the rebuilt cache holds
the target out.

The "~0.89–0.91" in 5-1 is the best score Optuna recorded during the competition (four
models): 0.885–0.891 by model in the parameter files saved during the competition, and
0.913–0.915 in the final notebook run that produced the 0.760 validation score. The
outputs of that final run are kept only in a local working copy of the notebook; the
published notebook has its outputs cleared, so these two figures cannot be checked from
this repository. The "SMOTE
before split" columns (both first and re-run) were measured again after the competition
with the saved parameter files, so the numbers differ slightly. All are scores with SMOTE
applied before the split.

An AUC gap of about **0.18** was observed between the leaky evaluation with SMOTE applied
before cross-validation and evaluation on original validation rows. The gap came from the
validation setup, not the model, and the hyperparameters had been chosen against the
inflated score. Because the evaluation populations and split procedures also differ, the
entire gap cannot be attributed solely to SMOTE placement.

### 5-3. Post-competition EDA and what the missing values mean

During the competition I filled every missing value with GAIN without asking why it was
missing. Below is a re-analysis of the raw training data done after the competition.

**Missing values are concentrated in a few columns.** Five columns are more than 90%
missing (days since egg thawing 99.4%, PGS 99.2%, PGD 99.2%, pre-implantation genetic
testing 98.9%, years since last pregnancy attempt 96.3%).

**Part of the missingness is caused by the treatment type.** 97.5% of records are IVF
(in vitro fertilisation) and 2.5% are DI (donor insemination). DI creates no embryos, so
29 embryo- and egg-related columns are 100% empty for DI rows. DI rows are 2.5% of the
data but hold 10.7% of all missing cells. These values are "not applicable", not "unknown".

| Treatment type | Share | Pregnancy rate |
|---|---:|---:|
| IVF | 97.5% | 26.2% |
| DI | 2.5% | 12.9% |

**Success falls with age, then rises again at 45+.**

| Age at treatment | Rows | Pregnancy rate | Donor-egg share |
|---|---:|---:|---:|
| 18–34 | 102,476 | 32.3% | 2.8% |
| 35–37 | 57,780 | 27.8% | 3.2% |
| 38–39 | 39,247 | 21.7% | 4.1% |
| 40–42 | 37,348 | 15.9% | 8.8% |
| 43–44 | 12,253 | 11.8% | 20.1% |
| 45–50 | 6,918 | 16.8% | 53.2% |
| Unknown | 329 | 0.0% | 0.0% |

More than half of the 45–50 group used donor eggs. Treatments with donor eggs succeed
31.5% of the time, versus 25.8% with the patient's own eggs. Age alone does not explain
success; egg source has to be considered with it. At the time of the competition I dropped
the egg-source column while trimming categorical features.

The 329 rows with unknown age contain no successes and no donor-egg use. They are few,
but their pattern is clearly different from the other groups.

**How the missing values should have been handled**

- Gaps that mean "not applicable", such as the embryo and egg columns of DI treatments,
  should not be filled with plausible values. Keep them as 0 or a separate category so that
  the "not applicable" information survives.
- Before filling missing values, check why they are missing.
- Do not drop categorical columns, such as egg source, that explain the effect of other
  features.

The effect of these choices on the score is checked in
[5-5](#5-5-missing-value-indicator-experiment),
[5-6](#5-6-what-actually-raised-the-score) and
[5-7](#5-7-handling-each-type-of-missing-value).

### 5-4. A leak-free baseline

Before adding anything complex, I first secured a score I could trust.
[`baseline_cv.py`](baseline_cv.py) starts from the raw CSV and, in every training fold,
fits a median imputer, a one-hot encoder and LightGBM from scratch. It uses every string
column and does no resampling, no label-based feature selection and no tuning.

**How the baseline handles missing values.** Of the 31 input columns with gaps, 29 are
numeric and 2 are strings (specific treatment type, main reason for embryo creation).

| Step | Numeric columns (47) | String columns (20) |
|---|---|---|
| Before the split (stateless, row by row) | `inf` is turned into a missing value | A missing value becomes the literal category `__MISSING__` |
| Fitted on the training fold | `SimpleImputer(strategy="median")`: each column's median over the training fold only | `OneHotEncoder`: the category list comes from the training fold only |
| Applied to the validation fold | Gaps filled with the training-fold median | `__MISSING__` is its own indicator column; unseen categories become all zeros |

- Nothing that looks at the label touches missing values, and no value from the
  validation fold reaches the median or the category list. This is what replaces
  the competition-time GAIN, which was fitted on all rows (and the target) before the
  split.
- No missingness indicator is added for numeric columns, so after imputation the model
  cannot tell a filled median from a real one. The experiment in
  [5-5](#5-5-missing-value-indicator-experiment) adds those indicators.
- The "not applicable" gaps of DI rows are also filled with medians. The model can
  still separate DI rows through the treatment-type column.
- All numeric gaps get the same treatment, whatever they mean. In three yes/no columns
  (PGS, PGD, pre-implantation genetic testing) only 1 is ever recorded, so the training
  median is 1 and every blank becomes 1. A blank there probably means "no", but the
  official data dictionary has not been checked to confirm it. The effect of this and of each
  other type of gap is measured in [5-7](#5-7-handling-each-type-of-missing-value).
- The ten count columns (for example total treatments, stored as `0회` … `6회 이상`)
  are strings in the raw file, so the baseline one-hot encodes them as categories rather
  than numbers.

| Rows | Features | Validation | OOF ROC-AUC | Average precision |
|---:|---:|---|---:|---:|
| 256,351 | 67 | Stratified 5-fold, seed 42 | **0.7392** | 0.4487 |

The random-ranking baseline for average precision is the positive rate, about 0.258. Fold
AUCs range from 0.7370 to 0.7420, with a standard deviation of about 0.002.

Each run saves the data hash, source hash, package versions and every out-of-fold
prediction. Re-running on the same machine reproduces the predictions byte for byte
([reference run](results/baseline_20261110T161422_601162Z/summary.json),
[run log](docs/RESULTS.md)).

The reference run predates the repository's first commit, so it records no commit
(`commit: null`, `dirty: true`).
Its recorded source hash matches this repository's `baseline_cv.py`, so the executed code
can be checked directly.

The competition-style pipeline ("SMOTE in fold" columns of 5-2, 0.719–0.730) scored below
this simple baseline. The two pipelines differ in several places, so the gap cannot be
traced to a single cause (the list of differences and the fold check are in the
[run log](docs/RESULTS.md#comparison-with-the-competition-pipeline)).

**The leaderboard score is unknown.** Submissions closed with the competition, so the
submission file for this baseline (`predict_test.py`) was never scored. The score above is
cross-validation within the training data and cannot be compared directly with the
leaderboard.

### 5-5. Missing-value indicator experiment

With the same folds and settings, I added only numeric missing-value indicators. AUC moved
from 0.73922 to 0.73937 (+0.00015). Per-fold differences average +0.00014 with a standard
deviation of 0.00017, and all five are positive. The observed gain was very small, and a
single-seed experiment was insufficient to establish a reliable improvement, so I retained
the simpler baseline ([details](docs/experiments/MISSINGNESS.md)).

The baseline's median imputer also fills "not applicable" gaps, such as the DI embryo and
egg columns from [5-3](#5-3-post-competition-eda-and-what-the-missing-values-mean), with
medians. The treatment-type column is still an input, so the model can tell DI rows apart.
One possible explanation for the nearly unchanged score is that the missingness
information was already carried by other columns; this experiment alone cannot establish
the cause. Setting DI gaps to 0 is compared on the same folds in
[5-6](#5-6-what-actually-raised-the-score).

### 5-6. What actually raised the score

Fixing the leak made the score trustworthy, but not higher. To see whether the
remaining gap was about missing values, I added one change at a time to the baseline
([`improve_cv.py`](improve_cv.py)) on the same five folds. Everything that learns from
data (imputation, column choice, hyperparameter search, early stopping) is fitted inside
each training fold; tuning uses an inner split of the training fold.

| Step (cumulative) | OOF ROC-AUC | vs previous | Folds up |
|---|---:|---:|---:|
| Baseline (median imputation) | 0.73922 | | |
| Leave numeric gaps as NaN for LightGBM | 0.73953 | +0.00032 | 4/5 |
| DI "not applicable" gaps as 0 | 0.73952 | −0.00002 | 2/5 |
| Count and age strings as ordered numbers | 0.73935 | −0.00017 | 1/5 |
| Eleven domain ratios (embryos per egg and others) | 0.73900 | −0.00035 | 0/5 |
| Nested Optuna tuning of LightGBM | 0.74002 | +0.00102 | 5/5 |
| CatBoost with native categories (same features) | 0.73981 | −0.00021 | 1/5 |
| Rank average of tuned LightGBM and CatBoost | **0.74034** | +0.00053 | 5/5 |

- **Missing-value handling was not the lever.** The best missing-value change moved AUC
  by +0.0003; setting DI gaps to 0 added nothing once the model saw NaN and the treatment
  type. These experiments did not include GAIN. The competition pipeline fitted GAIN
  before the split and with the target, but separating the effect of GAIN itself from
  that leakage needs a controlled comparison with GAIN fitted inside each fold, which
  has not been run.
- **Tuning and combining two different models helped**, by +0.0010 and +0.0005, in every
  fold. The final score is +0.0011 above the baseline.
- **The domain ratios I built did not help.** Tree models already combine the raw counts.

0.7403 is cross-validation AUC on the training data, not a leaderboard score, so it
cannot be compared with the 1st-place public score of 0.74350. One seed was used, so the
small differences are descriptive. Details and the full protocol:
[docs/experiments/IMPROVEMENT.md](docs/experiments/IMPROVEMENT.md).

### 5-7. Handling each type of missing value

Not every gap means the same thing. Checked against the other columns, the 31 columns
with gaps fall into five types
([details and counts](docs/experiments/MISSING_TYPES.md)):

| Type | Example | Size |
|---|---|---|
| 1. DI not applicable | Embryo and egg counts of DI treatments | 21 columns, all 6,291 DI rows |
| 2. Step not performed | Transfer day when no embryo was transferred | 5 elapsed-day columns; e.g. 36,535 IVF rows with no transfer |
| 3. Truly missing | Embryos were transferred but the transfer day is blank | About 1,600 IVF rows |
| 4. Blank in a flag that records only 1 (probably "no") | PGS, PGD and genetic-testing flags | 3 columns, 99% blank |
| 5. Unknown | Years since the last pregnancy attempt | 96.3% blank, no structural reason |

[`missing_types_cv.py`](missing_types_cv.py) encodes each type explicitly: 0 for
not-applicable counts and for blanks in those flags, −1 for days of steps that did not happen,
and an indicator for truly missing days. Every rule is row-wise or chosen from the training
fold. It was compared on the same five folds with the baseline and with leaving every gap
as NaN.

| Comparison | Difference in OOF AUC | Folds up |
|---|---:|---:|
| Treat flag blanks as 0 (median imputation otherwise) vs baseline | +0.00018 | 5/5 |
| Leave every gap as NaN vs baseline | +0.00032 | 4/5 |
| Type rules vs NaN, untuned LightGBM | +0.00002 | 2/5 |
| Type rules vs NaN, tuned LightGBM | +0.00003 | 3/5 |
| Type rules vs NaN, CatBoost | −0.00024 | 1/5 |
| Type rules vs NaN, LightGBM–CatBoost blend | −0.00014 | 0/5 |

- **Treating flag blanks as 0 helped a little.** AUC rose by +0.00018, in every fold. With
  the median, all rows of these columns become 1; with 0, only the few rows where 1 was
  actually recorded stand out. Why the gain is this small cannot be settled by this
  experiment alone.
- **Telling the types apart added nothing beyond NaN.** When gaps stay as NaN, the model
  learns inside each training fold where to send them, and the type is already visible in
  other columns (transferred embryos = 0, frozen embryo used = 0, treatment type).
- **The best result is the simplest input.** Gaps left as NaN, no hand-made ratios, and a
  rank blend of tuned LightGBM and CatBoost: **OOF AUC 0.74052**, +0.0013 above the
  baseline in every fold, and slightly above the 5-6 blend that included the ratios.

**How to read 0.7405.** It is the best of several variants compared on the same five
folds with one seed, so it is slightly optimistic. It is cross-validation AUC on the
training data, and submissions closed with the competition, so it cannot be placed on the
leaderboard; leaderboard scores near the top are only thousandths apart, and a
cross-validation score cannot say which side of those gaps a model would land on. What
these experiments support is the direction (which changes help, consistently across
folds), not a rank.

### 5-8. Where the model is right and where it fails

One overall AUC does not say which patients gained and where the model is weak. Without
retraining, the out-of-fold predictions saved in 5-7 were split by subgroup to compare the
baseline (0.7392) and the best blend (0.7405) on the same rows
([full tables and scope](docs/experiments/ERROR_ANALYSIS.md)).

| Rows | Count | Positive rate | AUC baseline → blend | Difference (95% interval) |
|---|---:|---:|---|---|
| All | 256,351 | 25.8% | 0.7392 → 0.7405 | +0.0013 (+0.0010 to +0.0017) |
| Embryo transferred, or DI | 219,807 | 30.1% | 0.6776 → 0.6792 | +0.0016 (+0.0012 to +0.0020) |

- **Much of the 0.74 comes from rows with no embryo transferred.** Of 36,544 IVF rows with
  zero embryos transferred, 30 (0.08%) are pregnancies, so the model gets these rows almost
  for free. Without them, both models rank at about 0.68. The improvement does not shrink
  when they are removed.
- **The blend gains a little in almost every group and does not clearly lose in any.** The
  gain is larger where the baseline is weaker: DI +0.0065, frozen-embryo cycles +0.0046,
  age 40–42 +0.0031. For age 45–50, donor eggs and 3 or more embryos transferred, the
  interval includes 0.
- **Hard for both models**: frozen-embryo cycles (AUC 0.65), two embryos transferred
  (0.65), donor eggs (0.68), DI (0.68–0.69).
- **When the inputs become known.** Some inputs, such as the number of embryos transferred
  and the transfer day, are known only well into the treatment, and the number transferred
  on its own ranks better than age (AUC 0.615 vs 0.596). So 0.74 is the score of a model
  that sees a cycle's full record, not the accuracy of a prediction made before treatment.
  A model trained only on pre-treatment columns was not measured. Without patient or clinic
  identifiers, generalization to new patients or clinics is not measured either.

## 6. Repository Layout

| Path | Contents |
|---|---|
| `notebooks/archive/` | Competition-time notebook |
| `src/`, `main.py`, `build_cache.py`, `evaluate_cv.py`, `*_study.py`, `tune.py`, `validate_tuning.py` | The notebook moved into modules after the competition, plus code that measured each cause. Some issues from 8-1 remain |
| `baseline_cv.py`, `compare_missingness.py`, `analyze_oof.py` | Leak-free workflow rebuilt after the review |
| `improve_cv.py`, `missing_types_cv.py` | Improvement and missing-value-type experiments on the same folds (5-6, 5-7) |
| `error_analysis.py` | Subgroup comparison from the saved out-of-fold predictions (5-8) |
| `predict_test.py` | Trains the same baseline on all training data and writes a submission file |
| `tests/` | Synthetic-data tests (run in CI) |
| `results/` | Run records (summary.json with hashes) |
| `docs/` | Validation audit, run log, reproduction guide |

## 7. How to Run

Python 3.11. Download the competition data as described in [docs/DATA.md](docs/DATA.md)
and place it at `data/train.csv` and `data/test.csv`.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python baseline_cv.py --sample 5000 --folds 3   # quick check
python baseline_cv.py --folds 5                 # full cross-validation
python predict_test.py                          # write a submission file
```

The improvement experiment in 5-6 also needs CatBoost and Optuna:

```bash
python -m pip install -r requirements-improve.txt
python improve_cv.py                            # about 17 minutes on a 15-core laptop
python missing_types_cv.py                      # about 45 minutes
python error_analysis.py                        # subgroup analysis, about a minute
```

Tests run without the competition data:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

More options: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## 8. Retrospective

### 8-1. Why the validation score was optimistic

Besides the SMOTE leakage in [5-2](#5-2-the-089-tuning-score-was-smote-leakage), rereading the notebook revealed more problems.

- **Ensemble weights were chosen on the validation set.** Optuna maximised validation AUC
  and the same validation set was used to report the score.
- **Feature selection and GAIN had already seen the validation set.** Both were computed
  on the full data before the train/validation split.
- **The age-group success rate was computed with validation labels.** This feature is a
  target mean, so computing it before the split put validation labels inside a feature.

### 8-2. Training and test data were processed differently

- **GAIN received the target column.** Missing values in the training data were filled
  while looking at the label. The test data has no label, so patterns the model learned
  probably did not hold on the test set.
- **GAIN was trained separately on training and test data.** The same missing value was
  filled by different rules in the two sets.

The validation set was filled the same way as the training data, so this problem stayed
invisible until the leaderboard. I think it is the most likely explanation for the gap
between validation 0.760 and leaderboard 0.714, but I have not tested it with the
competition-time version (how to test it: [8-5](#8-5-open-questions)).

### 8-3. Time went to things the metric ignores, and the implementation had bugs

- The metric was ROC-AUC, yet I spent effort finding the F1-optimal threshold. ROC-AUC
  does not depend on the threshold, so this work had no effect on the score.
- My GAIN implementation had three bugs: the reconstruction loss, an adversarial loss
  computed on the wrong entries, and a missing `detach`. All are fixed after the
  competition, and the two loss functions are covered by [regression tests](tests/).

Details with code references are in the [validation audit](docs/VALIDATION_AUDIT.md).

### 8-4. Comparison with top solutions, and what to improve

After the competition I read the Private 3rd- and 13th-place code shared on the DACON
board and compared it with my approach. The Private scores are from the leaderboard;
everything else was checked directly in the two notebooks. The table compares methods.
My post-competition scores are cross-validation on the training data, not on this scale,
so they are left out of it.

| | My approach | [Private 3rd](https://dacon.io/competitions/official/236452/codeshare/12308) | [Private 13th](https://dacon.io/competitions/official/236452/codeshare/12296) |
|---|---|---|---|
| Private score | 0.71446 | 0.74281 | 0.74228 |
| Class imbalance | SMOTE (50:50) | Effectively none (class weight ≈ 1.0) | Class weights |
| Missing values | GAIN for everything | DI rows filled with 0; MICE fitted on training data and applied to test | Missing values in DI-related columns as 0 or a separate category (`Not Answer(DI)`); special values such as 1 or 999 for elapsed days |
| Categorical features | All dropped except one string column | All kept; CatBoost native handling | All kept (two columns split into keywords); native handling |
| Features | Pregnancies per treatment, and age-group mean success rate (a target mean) | Many: per-stage embryo/egg ratios, egg age, combinations with treatment timing | Differences between treatment/pregnancy/birth counts, keyword splits |
| Validation | 20% holdout; tuning on SMOTE data | Stratified 25-fold | Stratified 5-fold |
| Models | 5 models + stacking and weighted ensemble | A single CatBoost | Weighted rank average of LightGBM, XGBoost and CatBoost (top 3 folds per model) |

Three differences stood out.

- **The top teams looked at what the missingness meant first.** Both treated the structural
  DI missingness from [5-3](#5-3-post-competition-eda-and-what-the-missing-values-mean) as 0 or a
  separate category. I filled in plausible embryo counts with GAIN.
- **The top teams put domain knowledge into the features.** I spent my time adding models
  and ensembles; they spent theirs understanding the treatment process and building
  features.
- **The top teams skipped SMOTE and matched the metric.** The 13th-place team used the fact that
  ROC-AUC only looks at order, converting predictions to ranks before a weighted average.

Testing these differences on my own leak-free folds ([5-6](#5-6-what-actually-raised-the-score),
[5-7](#5-7-handling-each-type-of-missing-value)) refined the first two. Once nothing was fitted
before the split, the way gaps were filled (median, NaN, 0 for DI, a rule per type) changed
AUC by at most 0.0003. GAIN was not among the methods compared: the competition pipeline
fitted it before the split and with the target, and telling the effect of GAIN itself apart
from that leakage would need GAIN fitted inside each fold. The ratio features I built did not help either, so I cannot confirm from my own
runs how much the top teams' features contributed. What did help, in every fold, was
tuning inside the fold and a rank average of two different models, close to what the
13th-place team did.

Putting 8-1 to 8-3 together with this comparison, these are the points to improve.

- **Question the validation score first.** Before changing the model, I should have checked
  whether the score could be trusted.
- **Every step that "learns" from training data belongs inside the fold.** That includes
  imputation, feature selection and oversampling.
- **Know what the metric responds to.** For ROC-AUC, the order of predicted probabilities
  matters, not the threshold.
- **Before filling missing values, ask why they are missing.** Missingness itself can be
  information.

### 8-5. Open questions

- **How would the best post-competition model score on the hidden test set?** Submissions
  are closed, so this cannot be measured, and checks without labels are limited. A
  classifier trained to tell training rows from test rows can reveal distribution
  differences that are detectable in the observed inputs. Chance-level separation does not
  show that the two distributions are the same or that test performance would hold: the
  classifier can miss differences, and a change in the relation between inputs and target,
  like the actual test score, cannot be checked this way.
- **How much did the train/test processing mismatch cost on the leaderboard?** ([8-2](#8-2-training-and-test-data-were-processed-differently))
  This can be estimated without the leaderboard. Hold out part of the training data and
  score it twice with the same model: once filled by GAIN with the target, as the
  competition validation set was, and once by a GAIN trained separately without the
  target, as the test set was. The difference estimates the effect of the processing
  mismatch in a held-out setting. Target inclusion and separate training change together in
  this comparison, and it does not reproduce the private test distribution, so it cannot
  identify the exact loss on the private leaderboard.
- **Does GAIN help once it is fitted inside each fold?** The old results cannot answer
  this. I plan to compare fold-fitted GAIN with median imputation on identical folds,
  over several seeds, and report the paired differences.
- **Do rows from the same patient or clinic fall into different folds?** The data's `ID`
  is unique per row and there is no patient- or clinic-level ID, so this cannot be checked
  yet. If several treatments of one patient are split across folds, the cross-validation
  score may still be optimistic.
- **Is every feature available at prediction time?** Some features, such as the number of
  transferred embryos or days to embryo transfer, are only known once treatment is under
  way. For predicting success before treatment they could not be used, and the score could
  be lower than reported here.

## Author

Lee Soon-jae ([soonjae-dev](https://github.com/soonjae-dev)). The competition data has its
own terms ([docs/DATA.md](docs/DATA.md)).
