import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from step3_split import FEATURE_COLS, TARGET_COL

# ============================================================
# STEP 4: Train Random Forest & XGBoost
# ============================================================

def train_random_forest(train: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",  # handles imbalance between "bad" vs "normal" days
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    return model


def train_xgboost(train: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    return model


if __name__ == "__main__":
    from step2_etl import clean_raw_data, build_daily_station_features
    from step3_split import time_based_split

    raw = pd.read_parquet("data-2026-06.parquet")
    daily = build_daily_station_features(clean_raw_data(raw))
    train, test = time_based_split(daily)

    rf_model = train_random_forest(train)
    xgb_model = train_xgboost(train)

    print("Random Forest feature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, rf_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<25} {imp:.3f}")

    print("\nModels trained successfully.")
