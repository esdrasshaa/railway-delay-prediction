import pandas as pd

# ============================================================
# STEP 1: Load raw railway data
# ============================================================
def load_raw_data(path: str) -> pd.DataFrame:
    """Load the monthly Deutsche Bahn parquet file."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(df.dtypes)
    return df

if __name__ == "__main__":
    df = load_raw_data("data-2026-06.parquet")
    print(df.head())
