# 🏚️ Ames Housing Price Prediction Pipeline

An end-to-end Machine Learning regression project using the Ames Housing dataset to predict residential home prices. Built in Python using Scikit-Learn, this repository covers data preprocessing, domain-specific imputation, feature encoding, and regularization.

---

## 📌 Project Overview

Predicting property values involves handling high-dimensional categorical features, skewed distributions, and domain-specific missing values. Standard linear models tend to overfit when dealing with numerous dummy variables. This project addresses these challenges by implementing **Ridge Regression ($L_2$ regularization)** to stabilize coefficient weights and maintain reliable predictions.

---

## ⚙️ Key Technical Steps

- **Domain-Specific Data Imputation:** Handled missing values (`NaN`) systematically—replacing missing structural records (e.g., basement square footage, garage space) with `0` to accurately represent non-existent amenities rather than using crude mean statistics.
- **Categorical Feature Encoding:** Applied One-Hot Encoding across high-cardinality categorical features.
- **Model Selection & Regularization:** Transitioned from standard Ordinary Least Squares (OLS) to **Ridge Regression**, controlling multicollinearity and preventing negative price predictions.
- **Evaluation & Deployment:** Evaluated models locally using $R^2$, MAE, and RMSE before generating test predictions for formal submission on Kaggle.

---

## 📊 Model Performance

| Metric | Validation Score |
| :--- | :--- |
| **$R^2$ Score** | **0.8366** (83.7% variance explained) |
| **MAE** | **$20,586.80** |
| **RMSE** | **$29,048.71** |
| **Kaggle Leaderboard (RMSLE)** | **0.22591** *(Baseline)* |

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Environment:** Google Colab / Jupyter Notebooks

---

## 🚀 Future Enhancements

- [ ] Apply **Target Log-Transformation** (`np.log1p`) to optimize performance directly against Kaggle's RMSLE metric.
- [ ] Implement **`StandardScaler`** across continuous numerical features.
- [ ] Fine-tune the regularization hyperparameter ($\alpha$) using `RidgeCV`.
