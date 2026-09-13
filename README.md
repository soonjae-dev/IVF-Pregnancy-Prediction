# IVF Pregnancy Success Prediction (LG Aimers 6th Hackathon)

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-blue?style=flat&logo=optuna&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Machine learning pipeline for the Dacon IVF Pregnancy Success Prediction competition. 
This repository refactors the initial single-file Jupyter Notebook experiments into a modular, production-ready structure.

## Project Background & Overview

**The Challenge**
Infertility is a growing global health issue, placing significant physical, emotional, and financial burdens on couples. For patients undergoing fertility treatments, minimizing the number of procedures while maximizing the chances of pregnancy is highly critical. 

**The AI Solution**
To address these challenges, healthcare institutions are increasingly turning to Artificial Intelligence (AI). AI-driven predictive models can analyze vast amounts of infertility treatment data to support optimal clinical decision-making and establish personalized treatment plans. This not only alleviates the burden on patients but also serves as a key competitive advantage for medical providers offering differentiated care.

**Hackathon Objective**
Developed for the LG Aimers Hackathon, this project focuses on building an AI model to predict "Pregnancy Success" using real-world infertility patient data. The primary goal is to identify the optimal features that determine pregnancy outcomes and construct a robust predictive model. Through this data-driven approach, we aim to explore innovative ways to enhance the overall efficiency of infertility treatments.

## Repository Structure
```text
├── configs/               # Hyperparameters and experiment configurations
├── data/                  # Raw dataset (Download from Dacon, ignored by git)
├── notebooks/             # Initial EDA and experiment notebook
├── src/                   # Core pipeline modules
│   ├── __init__.py
│   ├── config.py          # Environment setup, seed fixing, and device config (MPS/CUDA/CPU)
│   ├── preprocessing.py   # Data loading and basic preprocessing
│   ├── features.py        # Domain-based feature engineering (e.g., success rates)
│   ├── feature_selection.py # Correlation-based feature dropping
│   ├── encoding.py        # One-Hot encoding and train/test column synchronization
│   ├── gain_imputer.py    # PyTorch implementation of GAIN for missing values
│   ├── run_imputation.py  # Wrapper for GAIN execution
│   ├── data_splitting.py  # Train/Val split and CV strategy
│   ├── optimization.py    # Hyperparameter tuning using Optuna (no torch)
│   ├── tab_transformer.py # TabTransformer; the only module that needs torch
│   ├── ensemble.py        # Stacking and Weighted Ensemble logic
│   └── evaluation.py      # F1 threshold optimization and submission generation
├── main.py                # Main execution script
├── build_cache.py         # Steps 1-6 once, cached; GAIN isolated in its own process
├── evaluate_cv.py         # Leakage-free CV evaluation of the SMOTE protocols
├── resampling_study.py    # Imbalance strategies compared on identical folds
├── ensemble_study.py      # What the ensembles add over the best single model
├── target_encoding_study.py # Whether the age target encoding leaked, and what it was worth
├── column_study.py        # What is in the object columns the pipeline discarded
├── tune.py                # Resumable hyperparameter search, leakage-free protocol
├── validate_tuning.py     # Referee split: do the tuned parameters actually win?
├── TUNING.md              # How to run the re-tuning on a laptop, over days
├── requirements.txt       # Project dependencies
└── .gitignore
```

## Tech Stack
- **Data Processing & EDA**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning (Ensemble)**: `XGBoost`, `CatBoost`, `LightGBM`
- **Deep Learning**: `PyTorch` (MPS acceleration for macOS)
- **Optimization**: `Optuna`

## Key Engineering Strategies
- **Domain-Knowledge Feature Engineering**: Derived a feature calculating the pregnancy success rate per attempt from the total number of procedures and previous pregnancies. Every categorical column is one-hot encoded, including the ten the pipeline used to discard unread — the age bracket among them, which alone was worth 0.0148 AUC. See [Results](#results).
- **Advanced Missing Value Imputation**: Generative Adversarial Imputation Nets (GAIN) implemented in PyTorch. The target column is held out of the imputation and both generators are seeded — see [Results](#results) for why both matter.
- **Imbalanced Data Handling**: none. SMOTE was originally applied to the whole training set; it was first moved inside each cross-validation fold, then measured against no resampling at all, found to change nothing, and removed. The measurement is kept as a script — see [Results](#results).
- **Hyperparameter Optimization**: Utilized Optuna to fine-tune critical parameters for the four tree-based models the pipeline uses (XGBoost, LightGBM, RandomForest, CatBoost). A TabTransformer is also implemented and tunable in `src/tab_transformer.py`, but `main.py` does not call it.
- **Ensemble Strategy**: Predictions are combined through a Stacking Classifier and a weight-searched blend, with the F1 threshold chosen on a held-out calibration split. Both are worth about +0.003 AUC over the best single model — real and consistent across folds, but small; see [Results](#results).

## Results

Evaluated on the full training set: **256,351 rows × 104 features** after feature
engineering, selection, one-hot encoding and GAIN imputation, with a **25.8%
positive rate**.

**Evaluation protocol.** 5-fold `StratifiedKFold`. Each training fold is split
80/20 into a fit part and a calibration part; base models are fitted on the fit
part, the ensembles and the F1 threshold are built from calibration
probabilities alone, and everything is then scored on the validation fold, which
no fitting or tuning decision has touched. ROC-AUC is computed on pooled
out-of-fold probabilities.

| Model | OOF ROC-AUC | OOF F1 |
|---|---|---|
| RandomForest | 0.7319 | 0.5113 |
| LightGBM | 0.7311 | 0.5109 |
| CatBoost | 0.7239 | 0.5050 |
| XGBoost | 0.7236 | 0.5057 |
| **Stacked (logistic meta)** | **0.7346** | **0.5139** |
| **Weighted ensemble** | **0.7348** | **0.5142** |

Blend weights, averaged over folds: RandomForest 0.55 · LightGBM 0.33 ·
CatBoost 0.10 · XGBoost 0.02. Either ensemble is worth about **+0.003 AUC and
+0.003 F1** over the best single model — small, but consistent across all five
folds. The two are tied to within 0.0002, well inside the fold-to-fold spread;
nothing here says one beats the other. For reference, labelling every case
positive yields F1 = 0.4106.

Reproduce with `python ensemble_study.py`.

### Imbalance handling: measured, then dropped

The pipeline was built around SMOTE. On the representation it actually produces,
SMOTE does nothing:

| Model | no resampling | SMOTE | `scale_pos_weight` | SMOTE cost |
|---|---|---|---|---|
| RandomForest | 0.7319 | 0.7307 | 0.7278 | 1.8× time |
| LightGBM | 0.7311 | 0.7309 | 0.7303 | 2.9× |
| CatBoost | 0.7239 | 0.7208 | 0.7224 | 2.7× |
| XGBoost | 0.7236 | 0.7265 | 0.7192 | 3.2× |

Averaged over the four models, SMOTE is worth **−0.0004 AUC**. It helps exactly
one model, XGBoost, by +0.0029 — and XGBoost is the weakest of the four and
carries 2% of the blend weight, so that gain does not reach the final
prediction. It costs the other three between 0.0002 and 0.0031, and costs
everyone 1.8× to 3.2× the training time. `scale_pos_weight` came in below
no-resampling for all four.

The positive rate is 25.8%, which is 2.87:1 — not the regime SMOTE was designed
for, and one that gradient-boosted trees handle unaided. ROC-AUC is also
threshold-invariant, so most of what resampling does is invisible to it by
construction. Reproduce with `python resampling_study.py`.

The pipeline therefore no longer resamples anywhere: `prepare_training_data`
returns the split untouched and `src/optimization.py` scores a plain
`sklearn.pipeline.Pipeline`. That also removes the last place resampled rows
could reach a meta learner — `StackingClassifier` runs its own internal
cross-validation, which was previously being handed a SMOTE'd training set.

### The age target encoding: a leak that wasn't, hiding a feature that was

`src/features.py` used to target-encode the age bracket: group the training set
on `시술 당시 나이`, take the mean of the label, map it back as a feature. That
has the exact shape of a leak. Every training row's own label sat inside the
group mean it received, and the mapping was fitted once on the whole training
set, before any cross-validation split — so validation-fold labels reached the
training rows too.

Shape is not size. The bracket takes **7 values** and the smallest group holds
**329 rows**, so no single row can move its own group mean. Measured against an
out-of-fold encoding — mapping fitted on the outer training portion only, and
within it each row's value drawn from a nested split it was held out of:

| Model | as shipped | out-of-fold | column dropped | one-hot instead |
|---|---|---|---|---|
| LightGBM | 0.7284 | 0.7292 | 0.7137 | 0.7281 |
| XGBoost | 0.7181 | 0.7183 | 0.7043 | 0.7186 |

All four were measured on the 33-to-39-feature representation of the time, which
is why none of them match the table at the top — the discarded columns found in
the next section were added afterwards. What matters here is the comparison
between the columns, and all four share a representation and a set of folds.

The leak is worth **−0.0005 AUC**. It is not a leak; it is a leak-shaped thing
that never had room to leak.

The interesting column is the third one. Dropping the feature costs **0.0148**,
far more than a redundant convenience should. The reason is that `시술 당시
나이` is an *object* column, so `src/encoding.py` discarded it — and this
derived feature was the only path by which age reached the model at all. Age is
the strongest single predictor in the dataset: success rate runs 0.323 in the
18–34 bracket down to 0.118 at 43–44.

And the fourth column settles it. Replacing the encoding with a plain one-hot of
the same 7 brackets — same grouping, no labels anywhere — scores within
**0.0004**. Everything the feature contributed came from the grouping. The
labels were along for the ride.

So the bracket is now one-hot encoded and the target encoding is gone. Identical
accuracy, and a category of error that no longer has to be argued about. The
same argument would not survive a higher-cardinality key — target-encoding a
clinic ID or a patient ID here would leak properly, and the fix would have to be
the out-of-fold version rather than a one-hot.

Reproduce with `python target_encoding_study.py`.

### The rest of the discarded columns

Age had been thrown away by accident, not by judgement — so the same question
was worth asking of everything else on that side of the line.
`drop_unused_object_columns` keeps only what is on `columns_to_encode` and
discards the rest, which was nine columns: `특정 시술 유형`, `배란 유도 유형`,
`배아 생성 주요 이유`, `난자 출처`, `정자 출처`, `난자 기증자 나이`,
`정자 기증자 나이`, `시술 시기 코드`, `클리닉 내 총 시술 횟수`. (`시술 유형`
looks like a tenth but survives as the `시술 유형_원본` copy.)

Each was added to the encoding on its own and the **whole pipeline re-run** —
feature engineering, selection, encoding and GAIN — because adding a column
changes what the imputer sees and therefore every value in the matrix. Scored
by LightGBM on the same five folds:

| Added column | new cols | OOF ROC-AUC | Δ AUC |
|---|---|---|---|
| *(baseline, 39 features)* | — | 0.7281 | — |
| `시술 시기 코드` | 7 | 0.7289 | +0.0009 |
| `정자 출처` | 4 | 0.7288 | +0.0008 |
| `배아 생성 주요 이유` | 12 | 0.7288 | +0.0007 |
| `정자 기증자 나이` | 7 | 0.7288 | +0.0007 |
| `배란 유도 유형` | 2 | 0.7285 | +0.0005 |
| `난자 기증자 나이` | 5 | 0.7284 | +0.0004 |
| `클리닉 내 총 시술 횟수` | 7 | 0.7281 | +0.0000 |
| `난자 출처` | 3 | 0.7280 | −0.0001 |
| `특정 시술 유형` | 18 | 0.7275 | −0.0006 |

Nothing here is another age bracket. Every individual effect sits inside the
fold-to-fold spread, and reading this table as a ranking would be reading noise.

The obvious move — keep the six that helped — is the wrong one, and the study
checks it rather than assuming. Selecting on these deltas means selecting on
noise, so the six-column combination was scored against simply taking **all
nine, unselected**:

| | features | OOF ROC-AUC | OOF F1 |
|---|---|---|---|
| baseline | 39 | 0.7281 | 0.5079 |
| the six with positive deltas | 83 | 0.7303 | 0.5106 |
| **all nine, no selection** | **104** | **0.7311** | **0.5109** |

Not selecting wins. The filtering was fitting noise, and the simpler rule is
also the better one. Confirmed on a second model: XGBoost goes 0.7167 → 0.7236,
**+0.0069**, on the same change.

So all nine are now encoded. For scale, that is worth about as much as the
entire stacking-and-blending apparatus (+0.003), obtained by not discarding data.

Adding them broke `main.py`, which is worth recording. One-hot names inherit
whatever was in the category, and these categories contain `:` and `,` —
`특정 시술 유형_ICSI:IVF`, `배아 생성 주요 이유_기증용, 현재 시술용` — which
LightGBM rejects outright. The studies never saw it because they hand over bare
NumPy arrays; only `main.py`'s `StackingClassifier` passes a named DataFrame
through. `src/encoding.py` now sanitizes the generated names, which leaves the
feature set at 104 and the numbers above unchanged.

Reproduce with `python column_study.py`.

### Two leaks that were real

#### 1. SMOTE before the cross-validation split

An earlier version of this pipeline reported CV ROC-AUC ≈ 0.89 — still recorded as `best_value` in `configs/*.json`. That number was wrong, and working out why was the most instructive part of the project.

The original code applied SMOTE to the entire training set and passed the resampled data straight into `cross_val_score`. SMOTE interpolates between neighbouring minority samples, so a synthetic point derived from sample *A* could land in a training fold while *A* itself sat in the validation fold. The model had effectively already seen the neighbourhood of the answer.

Re-running that exact protocol on the same data reproduces the inflated figure and isolates the size of the effect:

| Model | Leaky protocol | Corrected protocol | Gap |
|---|---|---|---|
| XGBoost | 0.9061 ± 0.0006 | 0.7205 ± 0.0016 | −0.186 |
| LightGBM | 0.9046 ± 0.0009 | 0.7262 ± 0.0019 | −0.178 |
| CatBoost | 0.9039 ± 0.0008 | 0.7190 ± 0.0023 | −0.185 |
| RandomForest | 0.9088 ± 0.0009 | 0.7229 ± 0.0015 | −0.186 |

The first fix was to make SMOTE a step in an `imblearn.pipeline.Pipeline` in `src/optimization.py`, so that it is refitted within each fold. The section above is what happened when that corrected version was then measured against no resampling at all.

#### 2. The target column inside the imputer

Fixing the first leak made the pipeline runnable end to end for the first time —
`src/run_imputation.py` had been calling `gain_impute()` with its import
commented out, so `main.py` had been dying at step 6 of 10. With GAIN finally
executing, ROC-AUC jumped from 0.732 to 0.83. Imputation does not do that.

At the point GAIN is called, the training frame carries **34 columns** and the
test frame **33**. The extra one is the label. GAIN learns the joint
distribution of everything it is handed and uses it to fill gaps, so all
**978,332 missing cells** in the training matrix — 11.2% of it — were imputed by
a generator that had seen that row's outcome. Train and test were also imputed
by separately trained generators on different column counts, leaving the two in
different representations.

| Imputation | CV ROC-AUC |
|---|---|
| None — `fillna(-1)` | 0.7320 ± 0.0015 |
| GAIN with the target | 0.7943 ± 0.0007 |
| GAIN without the target | 0.7302 ± 0.0016 |

The leak was worth **+0.064 AUC**. Once it cannot see the label, GAIN's own
contribution is **−0.0019** — indistinguishable from skipping it.

`gain_impute` also drew its noise and mini-batches from unseeded generators.
Three runs on identical input gave 0.8696, 0.8592 and 0.8034: a spread as wide
as the leak, which made the reported figure a property of the run rather than of
the method. Both generators are now seeded, and two consecutive runs reproduce
exactly at 0.7300 ± 0.0017.

This one was introduced by the refactor, not present in the original work: the
February 2025 notebook's cached imputation scores 0.7294, in the same band as
the corrected pipeline. The label started riding along with the features when
the notebook was reorganised into modules.

**Known limitation.** The hyperparameters in `configs/*.json` were selected under
the leaky protocol and are reused as-is above, so the corrected figures retain a
small residual optimism.

`tune.py` and `validate_tuning.py` are the fix, and [TUNING.md](TUNING.md) is
the runbook. The search runs on 75% of the data under the corrected protocol and
is resumable, because it takes days on a laptop; the remaining 25% is a referee
split the search never sees, used to decide whether the new parameters really
beat the incumbents before anything is written to `configs/`. A search returns
the best of everything it tried, and that maximum is inflated by however many
things it tried — so it cannot be compared against anything, and the referee
split is what makes the comparison decidable.

Early evidence that this is worth doing: after five LightGBM trials and three
XGBoost trials, the referee split already shows **+0.0068** and **+0.0130**
ROC-AUC over the incumbents. The parameters carried forward from the leaky
search were not merely justified by a wrong number — they were the wrong
parameters.

Every `best_value` field in `configs/*.json` is a number from that protocol and
should be read as a record of what was searched, not as performance. Two of the
files are not read by any code: `tab_params.json` (the TabTransformer in
`src/tab_transformer.py` is implemented and tunable, but `main.py` never calls it)
and `cs_xgb_params.json` (a cost-sensitive XGBoost variant that did not make it
into the pipeline). They are kept because they record what was tried.

## How to Run
```bash
git clone https://github.com/soonjae-dev/IVF-Pregnancy-Prediction.git
cd IVF-Pregnancy-Prediction
pip install -r requirements.txt

# Place train.csv / test.csv from Dacon under data/
python main.py
```

## Author
- **Lee Soon-jae** ([@soonjae-dev](https://github.com/soonjae-dev))
