import pandas as pd

from step3_split import FEATURE_COLS

# ============================================================
# STEP 6: Predictions -- next-day delay probability
# ============================================================

def generate_predictions(model, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Score EVERY row (train+test) so Power BI can show both
    historical predictions and the most recent forecast per station.
    """
    daily = daily.copy()
    daily["predicted_bad_delay_probability"] = model.predict_proba(daily[FEATURE_COLS])[:, 1]
    daily["predicted_bad_delay_flag"] = (daily["predicted_bad_delay_probability"] >= 0.5).astype(int)
    return daily


def export_for_powerbi(daily_with_preds: pd.DataFrame, path: str = "predictions_for_powerbi.csv"):
    export_cols = [
        "station_name", "date", "weekday",
        "avg_delay", "delay_rate", "cancellation_rate",
        "target_bad_delay_day",              # actual outcome (for validating past predictions)
        "predicted_bad_delay_probability",   # the model's forecast
        "predicted_bad_delay_flag",
    ]
    daily_with_preds[export_cols].to_csv(path, index=False)
    print(f"Exported {len(daily_with_preds)} rows to {path}")


if __name__ == "__main__":
    from step2_etl import clean_raw_data, build_daily_station_features
    from step3_split import time_based_split
    from step4_train import train_xgboost

    raw = pd.read_parquet("data-2026-06.parquet")
    daily = build_daily_station_features(clean_raw_data(raw))
    train, test = time_based_split(daily)

    # In practice: retrain on ALL data (train+test) once you're happy
    # with the evaluation, so the final model sees the most recent days too.
    final_model = train_xgboost(daily)

    daily_with_preds = generate_predictions(final_model, daily)
    export_for_powerbi(daily_with_preds)

    print(daily_with_preds[
        ["station_name", "date", "predicted_bad_delay_probability", "predicted_bad_delay_flag"]
    ].tail(10))
