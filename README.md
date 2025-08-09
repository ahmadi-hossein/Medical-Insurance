Predicting medical insurance charges using classic ML regressors and ensembles on the popular **Medical Cost Personal Dataset**.  
Focus: clean preprocessing, solid evaluation, and **interpretable** results (feature importance / partial dependence).

> ⚠️ **Disclaimer:** This repository uses a public educational dataset and is **not** intended for clinical or actuarial decision-making.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Notebook at a Glance](#notebook-at-a-glance)
- [Quickstart](#quickstart)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Interpretability](#interpretability)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

---

## Overview
This project builds a regression pipeline to estimate **`charges`** based on features such as **age, sex, BMI, children, smoker, region**. The notebook covers EDA → feature engineering → model training (Linear/Tree/Ensemble) → evaluation → interpretation.

**Highlights**
- End-to-end in a single Jupyter Notebook (`medical-insurance.ipynb`)
- Robust validation (train/validation/test or k-fold)
- Strong baseline + tree-based ensembles (XGBoost/LightGBM/RandomForest)
- Clear metrics: **R²**, **MAE**, **RMSE**, (**MAPE** optional)
- Emphasis on interpretability and practical insights

---

## Dataset
- **Source:** Kaggle — *Medical Cost Personal Datasets* (`insurance.csv`)  
- **Target:** `charges`  
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`  

> Download the CSV and place it at `./data/insurance.csv` (you can change the path inside the notebook).

---

## Notebook at a Glance
`medical-insurance.ipynb` walks through:
1. **EDA**: distributions, outliers, correlations  
2. **Preprocessing**: missing checks, categorical encoding, scaling when needed  
3. **Modeling**: Linear Regression, Regularized (Ridge/Lasso), RandomForest, XGBoost/LightGBM  
4. **Evaluation**: R², MAE, RMSE (+ calibration / residual analysis)  
5. **Interpretation**: feature importance + partial dependence (and SHAP if enabled)

---

