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
├── evaluate_cv.py         # Leakage-free CV evaluation (reproduces the Results table)
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
- **Advanced Missing Value Imputation**: Implemented Generative Adversarial Imputation Nets (GAIN) using PyTorch to recover missing values realistically, upgrading from standard constant imputation.
- **Imbalanced Data Handling**: SMOTE is applied through an `imblearn.pipeline.Pipeline` step so that oversampling happens **inside each cross-validation fold**, never across fold boundaries. An earlier version resampled before splitting, which leaked synthetic neighbours into validation — see [Results](#results).
- **Hyperparameter Optimization**: Utilized Optuna to fine-tune critical parameters for tree-based models (XGBoost, LightGBM, RandomForest, CatBoost) and a deep learning model (TabTransformer).
- **Ensemble Strategy**: Maximized predictive performance by combining predictions through a Stacking Classifier and an Optuna-optimized Weighted Ensemble, targeting the optimal F1 threshold.

## Results

Evaluated on the full training set: **256,351 rows × 90 features** after feature engineering and GAIN imputation, with a **25.8% positive rate**.

**Evaluation protocol.** 5-fold `StratifiedKFold`. SMOTE is fitted on the training portion of each fold only; the validation fold keeps its original class distribution. The F1 decision threshold is chosen on a 20% calibration split carved out of that training portion, then applied to the validation fold, which is never used for any fitting or tuning decision. ROC-AUC is computed on pooled out-of-fold probabilities.

| Model | OOF ROC-AUC | OOF F1 | Avg. Precision | Threshold |
|---|---|---|---|---|
| LightGBM | **0.7262** ± 0.0019 | **0.5032** ± 0.0021 | 0.4407 | 0.258 |
| RandomForest | 0.7229 ± 0.0015 | 0.5022 ± 0.0017 | 0.4318 | 0.326 |
| XGBoost | 0.7205 ± 0.0016 | 0.4993 ± 0.0015 | 0.4359 | 0.220 |
| CatBoost | 0.7190 ± 0.0023 | 0.4973 ± 0.0022 | 0.4327 | 0.229 |
| **Weighted Ensemble** | **0.7304** | **0.5068** | — | 0.281 |

Ensemble weights: LightGBM 0.40 · RandomForest 0.31 · CatBoost 0.17 · XGBoost 0.11.
For reference, labelling every case positive yields F1 = 0.411.

Reproduce with `python evaluate_cv.py --data data/train_df_imputed.pkl`.

### A note on an earlier, inflated result

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

**Known limitation.** The hyperparameters in `configs/*.json` were selected under the leaky protocol and are reused as-is above, so the corrected figures retain a small residual optimism. Re-running the Optuna search under the corrected protocol is the natural next step.

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
