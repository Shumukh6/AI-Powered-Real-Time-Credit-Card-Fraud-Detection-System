# AI-Powered Credit Card Fraud Detection System

An end-to-end machine learning project for detecting fraudulent credit card transactions using XGBoost, Explainable AI, and an interactive Streamlit dashboard.

## Project Overview

This project simulates a real-world banking fraud detection system that analyzes transaction data and predicts whether a transaction is legitimate or fraudulent.

The system provides:

- Fraud probability score
- Risk level classification
- Recommended action
- AI-powered analyst explanation
- Live transaction feed simulation
- Interactive dashboard
- System architecture visualization

---

## Demo

Live App: (https://ai-powered-real-time-credit-card-fraud-detection-system-juny9r.streamlit.app/)

---

## Dataset

The project uses the Credit Card Transactions Fraud Detection dataset from Kaggle.

The dataset contains transaction-related information such as:

- Transaction amount
- Merchant category
- Transaction time
- Customer location
- Merchant location
- Customer profile
- Fraud label

Target column:

```text
is_fraud
```

---

## Machine Learning Pipeline

1. Data Loading  
2. Data Understanding  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Data Preprocessing  
6. XGBoost Model Training  
7. Model Evaluation  
8. Explainability using SHAP  
9. Streamlit Deployment  

---

## Feature Engineering

To improve fraud detection performance, several new features were engineered:

- Transaction Hour
- Customer Age
- Weekday
- Weekend Indicator
- Customer-Merchant Distance

These features helped the model capture temporal and behavioral fraud patterns.

---

## Model

Final model used:

```text
XGBoost Classifier
```

XGBoost was selected because it performs exceptionally well on large tabular datasets and handles imbalanced classification effectively using `scale_pos_weight`.

---

## Evaluation Results

| Metric | Result |
|--------|--------|
| Fraud Recall | 94% |
| ROC-AUC | 0.999 |
| Fraud F1-score | 0.80 |
| Fraud Precision | 0.70 |

The model achieved very high fraud detection capability while maintaining strong overall performance.

---

## Dashboard Features

The deployed Streamlit dashboard includes:

- Single transaction fraud analysis
- Fraud probability prediction
- Risk score gauge meter
- Recommended action
- AI analyst explanation
- Live transaction feed simulation
- System overview dashboard
- Architecture diagram

---

## Risk Levels

| Fraud Probability | Risk Level | Action |
|-------------------|------------|--------|
| 0% – 30% | Low | Approve |
| 30% – 70% | Medium | Manual Review |
| 70% – 100% | High | Block or Manual Review |

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  
- Plotly  
- GitHub  
- Streamlit Community Cloud  

---

## Project Architecture

```text
Transaction Input
        ↓
Feature Engineering
        ↓
XGBoost Fraud Model
        ↓
Fraud Probability
        ↓
Risk Level
        ↓
Recommended Action
        ↓
Analyst Dashboard
```

---

## Author

Developed by **Shumukh Alotaibi**.
