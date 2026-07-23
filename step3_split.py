import pandas as pd

# ============================================================
# STEP 3: Train/test split -- TIME-BASED, not random!
# ============================================================
#
# WICHTIG: Bei Zeitreihendaten NIEMALS random_state-basiertes
# train_test_split() verwenden! Das wuerde "die Zukunft" (spaetere
# Tage) versehentlich ins Training mischen und dir kuenstlich zu
# gute Metriken vorgaukeln (Data Leakage). Stattdessen: alles VOR
# einem Stichtag = Training, alles DANACH = Test.

def time_based_split(daily: pd.DataFrame, test_frac: float = 0.2):
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    cutoff_idx = int(len(daily) * (1 - test_frac))
    cutoff_date = daily["date"].sort_values().iloc[cutoff_idx]

    train = daily[daily["date"] < cutoff_date].copy()
    test = daily[daily["date"] >= cutoff_date].copy()

    print(f"Cutoff date: {cutoff_date.date()}")
    print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")

    return train, test


FEATURE_COLS = [
    "avg_delay", "max_delay", "n_trains", "delay_rate", "cancellation_rate",
    "weekday", "lag1_avg_delay", "lag1_delay_rate", "lag1_cancellation_rate",
    "roll3_delay_rate",
]
TARGET_COL = "target_bad_delay_day"


if __name__ == "__main__":
    from step2_etl import clean_raw_data, build_daily_station_features

    raw = pd.read_parquet("data-2026-06.parquet")
    daily = build_daily_station_features(clean_raw_data(raw))
    train, test = time_based_split(daily)
