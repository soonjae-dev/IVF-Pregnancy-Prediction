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
│   ├── data_splitting.py  # Train/Val split and SMOTE application
│   ├── optimization.py    # Hyperparameter tuning using Optuna
│   ├── ensemble.py        # Stacking and Weighted Ensemble logic
│   └── evaluation.py      # F1 threshold optimization and submission generation
├── main.py                # Main execution script
├── evaluate_cv.py         # Leakage-free CV evaluation of the SMOTE protocols
├── resampling_study.py    # Imbalance strategies compared on identical folds
├── ensemble_study.py      # What the ensembles add over the best single model
├── requirements.txt       # Project dependencies
└── .gitignore
```

## Tech Stack
- **Data Processing & EDA**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning (Ensemble)**: `XGBoost`, `CatBoost`, `LightGBM`
- **Deep Learning**: `PyTorch` (MPS acceleration for macOS)
- **Optimization**: `Optuna`

## Key Engineering Strategies
- **Domain-Knowledge Feature Engineering**: Derived a new feature calculating the pregnancy success rate based on the total number of procedures and previous pregnancies. Also mapped the average pregnancy success rate by age group using target encoding to capture demographic patterns.
- **Advanced Missing Value Imputation**: Generative Adversarial Imputation Nets (GAIN) implemented in PyTorch. The target column is held out of the imputation and both generators are seeded — see [Results](#results) for why both matter.
- **Imbalanced Data Handling**: SMOTE is applied through an `imblearn.pipeline.Pipeline` step so that oversampling happens **inside each cross-validation fold**, never across fold boundaries. Measured against no resampling at all it turns out to change nothing — see [Results](#results).
- **Hyperparameter Optimization**: Utilized Optuna to fine-tune critical parameters for tree-based models (XGBoost, LightGBM, RandomForest, CatBoost) and a deep learning model (TabTransformer).
- **Ensemble Strategy**: Maximized predictive performance by combining predictions through a Stacking Classifier and an Optuna-optimized Weighted Ensemble, targeting the optimal F1 threshold.

## Results

Evaluated on the full training set: **256,351 rows × 33 features** after feature
engineering, selection and GAIN imputation, with a **25.8% positive rate**.

**Evaluation protocol.** 5-fold `StratifiedKFold`. Each training fold is split
80/20 into a fit part and a calibration part; base models are fitted on the fit
part, the ensembles and the F1 threshold are built from calibration
probabilities alone, and everything is then scored on the validation fold, which
no fitting or tuning decision has touched. ROC-AUC is computed on pooled
out-of-fold probabilities.

| Model | OOF ROC-AUC | OOF F1 |
|---|---|---|
| LightGBM | 0.7284 | 0.5082 |
| RandomForest | 0.7273 | 0.5078 |
| CatBoost | 0.7225 | 0.5046 |
| XGBoost | 0.7181 | 0.5019 |
| Stacked (logistic meta) | 0.7311 | 0.5096 |
| **Weighted ensemble** | **0.7312** | **0.5105** |

Blend weights, averaged over folds: RandomForest 0.45 · LightGBM 0.41 ·
CatBoost 0.14 · XGBoost 0.01. The ensemble is worth **+0.0028 AUC and +0.0023
F1** over the best single model — small, but consistent across all five folds.
For reference, labelling every case positive yields F1 = 0.4106.

Reproduce with `python ensemble_study.py`.

### Imbalance handling: measured, then dropped

The pipeline was built around SMOTE. On the representation it actually produces,
SMOTE does nothing:

| Model | no resampling | SMOTE | `scale_pos_weight` | SMOTE cost |
|---|---|---|---|---|
| XGBoost | 0.7181 | 0.7201 | 0.7136 | 1.9× time |
| LightGBM | 0.7284 | 0.7283 | 0.7280 | 2.3× |
| CatBoost | 0.7225 | 0.7207 | 0.7213 | 1.5× |
| RandomForest | 0.7273 | 0.7255 | 0.7232 | 1.7× |

Averaged over the four models, SMOTE is worth **−0.0004 AUC**; the individual
differences all sit inside the fold-to-fold spread. `scale_pos_weight` was worst
or tied-worst in three of four.

The positive rate is 25.8%, which is 2.87:1 — not the regime SMOTE was designed
for, and one that gradient-boosted trees handle unaided. ROC-AUC is also
threshold-invariant, so most of what resampling does is invisible to it by
construction. Reproduce with `python resampling_study.py`.

### Two leaks, found and measured

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

The fix lives in `src/optimization.py`: SMOTE is now a step in an `imblearn.pipeline.Pipeline`, so it is refitted within each fold.

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
small residual optimism. Re-running the Optuna search under the corrected
protocol is the natural next step.

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
