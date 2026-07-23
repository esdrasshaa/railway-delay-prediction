import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from step3_split import FEATURE_COLS, TARGET_COL

# ============================================================
# STEP 5: Model evaluation
# ============================================================

def evaluate_model(model, test: pd.DataFrame, name: str = "Model") -> dict:
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class "1" (bad delay day)

    metrics = {
        "model": name,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n=== {name} ===")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


if __name__ == "__main__":
    from step2_etl import clean_raw_data, build_daily_station_features
    from step3_split import time_based_split
    from step4_train import train_random_forest, train_xgboost

    raw = pd.read_parquet("data-2026-06.parquet")
    daily = build_daily_station_features(clean_raw_data(raw))
    train, test = time_based_split(daily)

    rf_model = train_random_forest(train)
    xgb_model = train_xgboost(train)

    rf_metrics = evaluate_model(rf_model, test, name="Random Forest")
    xgb_metrics = evaluate_model(xgb_model, test, name="XGBoost")

    comparison = pd.DataFrame([rf_metrics, xgb_metrics])
    print("\n=== Comparison ===")
    print(comparison)
