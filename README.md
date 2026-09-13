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
│   ├── config.py          # Seeding and device selection (torch imported lazily)
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
├── main.py                # Main execution script (reads the cache; no torch)
├── build_cache.py         # Steps 1-6 once, cached; GAIN isolated in its own process
├── evaluate_cv.py         # Reproduces the SMOTE-before-CV leak against its fix
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
- **Imbalanced Data Handling**: none. SMOTE was originally applied to the whole training set; it was first moved inside each cross-validation fold, then measured against no resampling at all and found to lose to it in all four models, and removed. The measurement is kept as a script — see [Results](#results).
- **Hyperparameter Optimization**: Optuna, re-run under a leakage-free protocol on a 75% tuning split, with the remaining 25% held back to decide whether the new parameters actually beat the old ones. Worth +0.0148 AUC to XGBoost and +0.0142 to CatBoost — see [Results](#results) and [TUNING.md](TUNING.md). A TabTransformer is also implemented and tunable in `src/tab_transformer.py`, but `main.py` does not call it.
- **Ensemble Strategy**: A Stacking Classifier and a weight-searched blend, with the F1 threshold chosen on a held-out calibration split. Both are implemented and both are now worth **nothing** — +0.0002 AUC over a single LightGBM, and behind it on F1. [Results](#results) explains why.

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
| **LightGBM** | **0.7385** | 0.5156 |
| XGBoost | 0.7384 | 0.5154 |
| CatBoost | 0.7381 | **0.5161** |
| RandomForest | 0.7333 | 0.5126 |
| Stacked (logistic meta) | 0.7384 | 0.5159 |
| Weighted ensemble | 0.7386 | 0.5156 |

Fold-to-fold F1 spread is ±0.0012, and every figure in the top five rows sits
inside a 0.0005 band of ROC-AUC. Read that as: after tuning, they are the same
model. Labelling every case positive yields F1 = 0.4106.

Blend weights, averaged over folds: LightGBM 0.57 · CatBoost 0.29 ·
XGBoost 0.10 · RandomForest 0.04.

Reproduce with `python ensemble_study.py`.

### The ensemble stopped being worth anything

Before the hyperparameters were re-tuned, the weighted blend beat the best
single model by +0.0038 AUC, consistently across folds. That was the number
this README reported and the reason the ensemble was there.

After re-tuning, on the same protocol:

| | best single | weighted | stacked |
|---|---|---|---|
| ROC-AUC | 0.7385 | 0.7386 (+0.0002) | 0.7384 (−0.0000) |
| OOF F1 | 0.5161 | 0.5156 (−0.0005) | 0.5159 (−0.0002) |

It buys **+0.0002 AUC**, and on F1 the best single model — CatBoost — is ahead
of both ensembles. Against a fold spread of ±0.0012 there is nothing here.

The explanation is in the base models. Before tuning they ranged over 0.0083 of
AUC; now they sit within 0.0004 of each other. Averaging models that disagree
recovers something; averaging models that agree recovers their common answer.
The ensemble's value had been compensating for badly tuned base models, and the
tuning removed the thing it was compensating for. The weights show the same
collapse from the other side: RandomForest went from 0.55 of the blend to 0.04.

The code stays — it runs, and `main.py` still emits both submissions — but a
single LightGBM is the defensible default now, and this section is here so that
nobody reads the ensemble as load-bearing.

### Imbalance handling: measured, then dropped

The pipeline was built around SMOTE. On the representation it actually produces,
with parameters tuned honestly, SMOTE is worse than doing nothing:

| Model | no resampling | SMOTE | `scale_pos_weight` | SMOTE cost |
|---|---|---|---|---|
| LightGBM | 0.7385 | 0.7369 | 0.7383 | 2.2× time |
| XGBoost | 0.7384 | 0.7357 | 0.7383 | 2.3× |
| CatBoost | 0.7381 | 0.7375 | 0.7380 | 2.4× |
| RandomForest | 0.7333 | 0.7313 | 0.7326 | 1.8× |

SMOTE loses to plain training in **all four**, by −0.0017 on average, at
1.8× to 2.4× the training time. `scale_pos_weight` is nearly free and nearly
neutral — between −0.0001 and −0.0007 — but never better.

An earlier run of this table, before the hyperparameters were re-tuned, had one
exception: XGBoost gained +0.0029 from SMOTE. That exception is gone, and where
it went is the interesting part. XGBoost was the worst-tuned of the four under
the leaky-protocol parameters, and resampling was partly making up for it. Given
parameters chosen honestly, XGBoost now loses the most of any model (−0.0027).

That is the same shape as what happened to the ensemble: a technique that
appeared to earn its place was compensating for a defect elsewhere, and stopped
earning it once the defect was fixed. Neither was measured against a properly
tuned baseline when it was adopted.

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

So all nine are now encoded. For scale, at the time it was measured that was
worth about as much as the entire stacking-and-blending apparatus, obtained by
not discarding data. The apparatus has since been re-measured at nothing, which
makes the comparison less flattering to the apparatus than it was meant to be.

Adding them broke `main.py`, which is worth recording. One-hot names inherit
whatever was in the category, and these categories contain `:` and `,` —
`특정 시술 유형_ICSI:IVF`, `배아 생성 주요 이유_기증용, 현재 시술용` — which
LightGBM rejects outright. The studies never saw it because they hand over bare
NumPy arrays; only `main.py`'s `StackingClassifier` passes a named DataFrame
through. `src/encoding.py` now sanitizes the generated names, which leaves the
feature set at 104 and the numbers above unchanged.

Every number in this section was measured before the hyperparameters were
re-tuned, so none of them match the table at the top. The comparison is between
columns, and all of it shares one set of parameters and one set of folds.

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

### Re-tuning the hyperparameters

The parameters in `configs/*.json` were chosen by the leaky search. Fixing the
protocol did not re-pick them, so every figure above until recently was produced
by parameters selected to be good at exploiting a leak.

`tune.py` searches again under the corrected protocol, and
[TUNING.md](TUNING.md) is the runbook: resumable to a SQLite study, because it
runs over days on a laptop. It sees 75% of the data. The other 25% is a referee
split, used to choose nothing during the search, and `validate_tuning.py` fits
both parameter sets on the tuning split and scores them there before anything is
written to `configs/`. That separation is the whole design — a search returns
the best of everything it tried, and that maximum is inflated by however many
things it tried, so it cannot be compared against an incumbent's single score.

On the referee split, with 164 completed LightGBM trials, 121 XGBoost, and
around 50 each for RandomForest and CatBoost:

| Model | leaky-protocol params | re-tuned | Δ AUC |
|---|---|---|---|
| XGBoost | 0.7250 | 0.7404 | **+0.0154** |
| CatBoost | 0.7255 | 0.7406 | **+0.0151** |
| LightGBM | 0.7323 | 0.7406 | +0.0083 |
| RandomForest | 0.7336 | 0.7352 | +0.0016 |

All four were adopted. The published figures at the top of this section are
measured on the result.

Two things are worth reading off that table. The first is that the leaky search
did not merely justify itself with a wrong number — it picked the wrong
parameters, and the models it hurt most were the ones that had looked weakest.
The second is the convergence: after tuning, the three boosting models land
within 0.0002 of each other on the referee split. What had looked like
differences between model families was mostly differences in how badly each had
been mis-tuned.

A further 50 trials each for RandomForest and CatBoost afterwards improved
nothing — every model came back `keep incumbent`. That is the signal to stop:
three independent searches converging on the same number is a property of the
data, not of the search.

**What is still optimistic.** The referee split settles *which* parameters to
use. The figures above are still cross-validated over all the data, including
the 75% the search saw, so a little optimism remains. Removing it takes nested
cross-validation — a separate search inside each outer fold — which is five
searches instead of one.

**On `configs/`.** Each replaced file is kept beside its replacement as
`<model>_params.leaky.json`, so what the leaky search chose is still in the
repository. `best_value` in those files is a number from that protocol and
records what was searched, not performance. Two configs are read by no code:
`tab_params.json` (the TabTransformer in `src/tab_transformer.py` is implemented
and tunable, but `main.py` never calls it) and `cs_xgb_params.json` (a
cost-sensitive XGBoost variant that did not make it into the pipeline). They are
kept because they record what was tried.

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
