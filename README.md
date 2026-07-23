# 🚆 Railway Delay Prediction & Power BI Dashboard

## 📌 Project Overview

This project develops an end-to-end machine learning pipeline to predict railway station delay risks using real-world transportation data from **5,287 stations**.

The goal is to transform raw railway operational data into actionable insights through:
- Data preprocessing and feature engineering
- Time-based machine learning prediction
- Model evaluation
- Interactive Power BI dashboard visualization

---

## 🔄 Machine Learning Pipeline

The project follows a complete ML workflow:

1. **Data Loading**
   - Import large-scale railway datasets (Parquet format)

2. **ETL & Feature Engineering**
   - Data cleaning
   - Daily aggregation per station
   - Creation of lag features:
     - Previous delay rates
     - Rolling delay statistics

3. **Time-Based Data Split**
   - Training and testing split based on time order
   - Prevents data leakage in time-series prediction problems

4. **Model Training**
   - Random Forest
   - XGBoost
   - Class balancing for imbalanced delay events

5. **Model Evaluation**
   - ROC-AUC
   - Precision
   - Recall
   - Confusion Matrix

6. **Prediction Export**
   - Generate station-level risk predictions
   - Export results for Power BI visualization

---

## 🤖 Models

Two supervised learning models are compared:

| Model | Purpose |
|---|---|
| Random Forest | Robust baseline model |
| XGBoost | Gradient boosting model for improved prediction performance |

The target variable represents whether a station experiences a high-risk delay day.

---

## 📊 Power BI Dashboard

The dashboard contains three main views:

### 1. Model Performance
- ROC-AUC
- Precision / Recall
- Confusion Matrix
- Prediction quality overview

### 2. Risk Ranking
- Ranking of stations by predicted delay risk
- Identification of critical locations
- Focus on actionable insights instead of displaying all 5,287 stations

### 3. Station Details
- Individual station analysis
- Historical delay patterns
- Predicted risk probability

---

## 📈 Key Results

The final model achieved approximately:

- Precision: **~74%**
- Recall: **~53%**

Interpretation:
- When the model predicts a risky day, the prediction is usually correct.
- However, some risky days are still missed.

This represents the typical precision-recall trade-off in operational risk prediction systems.

---

## 🧠 Technical Highlights

- Avoided random train/test splitting to prevent **data leakage**
- Used lag-based features to simulate real forecasting scenarios
- Applied class weighting to handle imbalanced target classes
- Built an end-to-end pipeline from raw data to business dashboard

---

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- XGBoost
- Power BI
- Parquet
- Git

---

