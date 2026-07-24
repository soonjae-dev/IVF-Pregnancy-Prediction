# 🧬 IVF Pregnancy Success Prediction (LG Aimers 5th Hackathon)

## 📌 Project Background & Overview

**The Challenge**
Infertility is a growing global health issue, placing significant physical, emotional, and financial burdens on couples. For patients undergoing fertility treatments, minimizing the number of procedures while maximizing the chances of pregnancy is highly critical. 

**The AI Solution**
To address these challenges, healthcare institutions are increasingly turning to Artificial Intelligence (AI). AI-driven predictive models can analyze vast amounts of infertility treatment data to support optimal clinical decision-making and establish personalized treatment plans. This not only alleviates the burden on patients but also serves as a key competitive advantage for medical providers offering differentiated care.

**Hackathon Objective**
Developed for the LG Aimers Hackathon, this project focuses on building an AI model to predict "Pregnancy Success" using real-world infertility patient data. The primary goal is to identify the optimal features that determine pregnancy outcomes and construct a robust predictive model. Through this data-driven approach, we aim to explore innovative ways to enhance the overall efficiency of infertility treatments.

## 📁 Repository Structure
```text
├── data/               # Raw and preprocessed data (Excluded via .gitignore)
├── notebooks/          # Jupyter notebooks for EDA and initial experiments
├── src/                # Modularized Python scripts
│   ├── preprocessing.py# Data loading and missing value imputation
│   ├── features.py     # Domain-based feature engineering
│   └── train.py        # Model training and Optuna optimization
├── models/             # Saved model weights
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🛠️ Tech Stack
- **Data Processing & EDA**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning (Ensemble)**: `XGBoost`, `CatBoost`, `LightGBM`
- **Deep Learning**: `PyTorch` (MPS acceleration for macOS)
- **Optimization**: `Optuna`

## 💡 Key Engineering Strategies
1. **Domain-Knowledge Feature Engineering**
   - **Success Rate per Attempt**: Derived a new feature calculating the pregnancy success rate based on the total number of procedures and previous pregnancies.
   - **Age-Group Mapping**: Mapped the average pregnancy success rate by age group to capture demographic patterns.
2. **Handling Missing Values & Imbalance**
   - **Imputation**: Replaced non-informative missing values with `-1` to reduce noise.
   - **Class Imbalance**: Applied `scale_pos_weight: 2.0` (in XGBoost) and equivalent techniques to heavily weight the positive class (pregnancy success).
3. **Hyperparameter Optimization**
   - Utilized **Optuna** to fine-tune critical parameters such as tree depth (`max_depth`) and iterations, maximizing predictive performance.

## 🚀 How to Run
```bash
git clone https://github.com/soonjae-dev/ivf-pregnancy-prediction.git
cd ivf-pregnancy-prediction
pip install -r requirements.txt
python src/train.py
```

## 👤 Author
- **Lee Soon-jae** ([@soonjae-dev](https://github.com/soonjae-dev))
