# IVF Pregnancy Success Prediction (LG Aimers 6th Hackathon)

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
- **Imbalanced Data Handling**: Applied SMOTE strictly on the training set to resolve class imbalance and oversample the positive class, carefully structured to prevent data leakage into the validation set.
- **Hyperparameter Optimization**: Utilized Optuna to fine-tune critical parameters for tree-based models (XGBoost, LightGBM, RandomForest, CatBoost) and a deep learning model (TabTransformer).
- **Ensemble Strategy**: Maximized predictive performance by combining predictions through a Stacking Classifier and an Optuna-optimized Weighted Ensemble, targeting the optimal F1 threshold.

## How to Run
```bash
git clone https://github.com/soonjae-dev/ivf-pregnancy-prediction.git
cd ivf-pregnancy-prediction
pip install -r requirements.txt
python src/train.py
```

## Author
- **Lee Soon-jae** ([@soonjae-dev](https://github.com/soonjae-dev))
