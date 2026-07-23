import pandas as pd
import numpy as np

# ============================================================
# STEP 2: ETL -- cleaning & feature engineering
# ============================================================
#
# GOAL: our raw data has one row PER TRAIN STOP (very granular).
# For "next-day delay probability", we need one row PER STATION
# PER DAY, with features describing that day, and a TARGET
# describing whether TOMORROW was delayed.

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix types, drop obviously broken rows."""
    df = df.copy()

    # Ensure 'time' is a proper datetime (should already be, but be safe)
    df["time"] = pd.to_datetime(df["time"])

    # Drop rows with missing essential fields
    df = df.dropna(subset=["station_name", "delay_in_min", "time"])

    # Negative delays don't make physical sense here -> clip at 0
    df["delay_in_min"] = df["delay_in_min"].clip(lower=0)

    return df


def build_daily_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw per-train rows into one row per (station, date),
    with engineered features + the next-day target.
    """
    df = df.copy()
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.dayofweek  # 0=Monday

    # --- Daily aggregation per station ---
    daily = (
        df.groupby(["station_name", "date"])
        .agg(
            avg_delay=("delay_in_min", "mean"),
            max_delay=("delay_in_min", "max"),
            n_trains=("delay_in_min", "count"),
            n_delayed=("delay_in_min", lambda x: (x > 5).sum()),
            n_canceled=("is_canceled", "sum"),
            weekday=("weekday", "first"),
        )
        .reset_index()
    )

    daily["delay_rate"] = daily["n_delayed"] / daily["n_trains"]
    daily["cancellation_rate"] = daily["n_canceled"] / daily["n_trains"]

    # --- Sort so lag/lead features are computed correctly per station ---
    daily = daily.sort_values(["station_name", "date"]).reset_index(drop=True)

    # --- Lag features: yesterday's stats (used as PREDICTORS) ---
    group = daily.groupby("station_name")
    daily["lag1_avg_delay"] = group["avg_delay"].shift(1)
    daily["lag1_delay_rate"] = group["delay_rate"].shift(1)
    daily["lag1_cancellation_rate"] = group["cancellation_rate"].shift(1)

    # 3-day rolling average of delay rate (smoother trend feature)
    daily["roll3_delay_rate"] = (
        group["delay_rate"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- TARGET: will TOMORROW have a "bad" delay day? ---
    # Defined as: more than 30% of tomorrow's trains delayed >5min
    daily["next_day_delay_rate"] = group["delay_rate"].shift(-1)
    daily["target_bad_delay_day"] = (daily["next_day_delay_rate"] > 0.30).astype(int)

    # Drop rows where we don't have enough history (first day per station)
    # or don't have tomorrow's outcome yet (last day per station)
    daily = daily.dropna(
        subset=["lag1_avg_delay", "lag1_delay_rate", "next_day_delay_rate"]
    ).reset_index(drop=True)

    return daily


if __name__ == "__main__":
    raw = pd.read_parquet("data-2026-06.parquet")
    clean = clean_raw_data(raw)
    daily = build_daily_station_features(clean)

    print(f"Daily station-level rows: {len(daily)}")
    print(daily.head(10))
    print("\nTarget distribution:")
    print(daily["target_bad_delay_day"].value_counts(normalize=True))
