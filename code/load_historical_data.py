
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"



# --------------------------------------------------
# 2. Load historical data
# --------------------------------------------------

def load_historical_data():
    DATA_FILE =DATA_DIR / "API_DA_Price.csv"

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical data file not found: {DATA_FILE}"
        )

    

    df = pd.read_csv(DATA_FILE)

    print("Historical data loaded successfully.")


    # --------------------------------------------------
    # 3. Validate dataset structure
    # --------------------------------------------------

    if "timestamp_utc" not in df.columns:
        raise ValueError(
            "Required column 'timestamp_utc' is missing."
        )


    # --------------------------------------------------
    # 4. Convert timestamp to datetime
    # --------------------------------------------------

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    # Remove rows with invalid timestamps
    invalid_timestamps = df["timestamp_utc"].isna().sum()

    if invalid_timestamps > 0:
        print(
            f"Warning: {invalid_timestamps} invalid timestamps found."
        )

        df = df.dropna(subset=["timestamp_utc"])


    # --------------------------------------------------
    # 5. Sort data chronologically
    # --------------------------------------------------

    df = df.sort_values(
        by="timestamp_utc"
    ).reset_index(drop=True)

    # --------------------------------------------------
# ==========================================
# INVESTIGATE INCOMPLETE DAYS AND EXTREME PRICES
# ==========================================

    # Convert delivery timestamp to datetime
    df["dtm"] = pd.to_datetime(df["dtm"], utc=True)

    # 1. Inspect incomplete days
    daily_counts = df.groupby("settlement_date").size()

    print("\nIncomplete days:")
    print(daily_counts[daily_counts != 48])

    # 2. Inspect the first and last dates
    print("\nFirst date records:")
    print(
        df[df["settlement_date"] == daily_counts.index.min()]
        .sort_values("dtm")
    )

    print("\nLast date records:")
    print(
        df[df["settlement_date"] == daily_counts.index.max()]
        .sort_values("dtm")
    )

    # 3. Examine the 10 highest electricity prices
    print("\nTop 10 highest electricity prices:")
    print(
        df.nlargest(10, "price")[
            ["timestamp_utc", "settlement_date",
            "settlement_period", "price", "dtm"]
        ].to_string(index=False)
    )

    # 4. Examine the 10 lowest electricity prices
    print("\nTop 10 lowest electricity prices:")
    print(
        df.nsmallest(10, "price")[
            ["timestamp_utc", "settlement_date",
            "settlement_period", "price", "dtm"]
        ].to_string(index=False)
    )
        
    # ==========================================
    # SETTLEMENT PERIOD AND TIMESTAMP INVESTIGATION
    # ==========================================

    # Sort chronologically
    df = df.sort_values("dtm").reset_index(drop=True)

    # 1. Inspect duplicated settlement period numbers
    print("\nDuplicated settlement periods on each day:")

    for date in ["2020-10-25", "2021-03-28", "2023-11-30"]:

        daily_df = df[df["settlement_date"] == date]

        duplicated = daily_df[
            daily_df["settlement_period"].duplicated(keep=False)
        ]

        print(f"\nDate: {date}")
        print(duplicated[
            ["settlement_period", "dtm", "price"]
        ].to_string(index=False))


    # 2. Inspect delivery timestamp gaps
    df["time_difference"] = df["dtm"].diff()

    gaps = df[df["time_difference"] != pd.Timedelta(minutes=30)]

    print("\nUnexpected delivery timestamp intervals:")
    print(
        gaps[["dtm", "time_difference"]].to_string(index=False)
    )


    # 3. Inspect the final settlement periods
    print("\nFinal records of the dataset:")

    print(
        df.tail(10)[
            ["timestamp_utc", "settlement_date",
            "settlement_period", "dtm", "price"]
        ].to_string(index=False)
    )


    # 4. Compare timestamp_utc and dtm
    print("\nTimestamp comparison:")

    print(
        df[["timestamp_utc", "dtm"]].head(10).to_string(index=False)
    )


    
# ==========================================
# UK LOCAL TIME AND SETTLEMENT CALENDAR CHECK
# ==========================================

    # Convert UTC delivery timestamps to UK local time
    df["dtm_uk"] = df["dtm"].dt.tz_convert("Europe/London")

    # Dates to investigate
    dates_to_check = [
        "2020-10-25",
        "2021-03-28",
        "2023-11-30"
    ]

    for date in dates_to_check:

        print(f"\n========== {date} ==========")

        # Select records using the UK local delivery date
        daily_df = df[
            df["dtm_uk"].dt.strftime("%Y-%m-%d") == date
        ].copy()
  
        print("Number of observations:", len(daily_df))

        print(
            daily_df[
                ["settlement_date", "settlement_period",
                "dtm", "dtm_uk", "price"]
            ].to_string(index=False)
        )

        print("\nRepeated UK local delivery times:")

        repeated_times = daily_df[
            daily_df["dtm_uk"].dt.strftime("%Y-%m-%d %H:%M")
            .duplicated(keep=False)
        ]

        print(
            repeated_times[
                ["settlement_period", "dtm", "dtm_uk"]
            ].to_string(index=False)
        )



    return df

def inspect_data_quality(df):

    print("\n========== DATA QUALITY ANALYSIS ==========")

    # 1. Check data types
    print("\nData types:")
    print(df.dtypes)

    # 2. Convert delivery-time marker
    df["dtm"] = pd.to_datetime(
        df["dtm"],
        utc=True,
        errors="coerce"
    )

    # 3. Check missing delivery timestamps
    print("\nInvalid delivery timestamps:")
    print(df["dtm"].isna().sum())

    # 4. Check duplicate delivery timestamps
    print("\nDuplicate delivery timestamps:")
    print(df["dtm"].duplicated().sum())

    # 5. Check settlement periods per day
    periods_per_day = (
        df.groupby("settlement_date")["settlement_period"]
        .agg(["count", "min", "max", "nunique"])
    )

    print("\nSettlement periods per day:")
    print(periods_per_day["count"].describe())

    print("\nDays with unusual period counts:")
    print(
        periods_per_day[
            periods_per_day["count"] != 48
        ].head(20)
    )

    # 6. Check price distribution
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    print("\nPrice statistics:")
    print(df["price"].describe())

    # 7. Check delivery interval gaps
    sorted_dtm = df["dtm"].dropna().sort_values()

    intervals = sorted_dtm.diff()

    print("\nDelivery interval distribution:")
    print(intervals.value_counts().head(10))

    print("\n========== QUALITY ANALYSIS COMPLETE ==========")

    return periods_per_day

    # --------------------------------------------------
    # 6. Check for duplicate records
    # --------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)


    # --------------------------------------------------
    # 7. Display dataset information
    # --------------------------------------------------

    print("\n========== DATASET OVERVIEW ==========")

    print(f"Total observations: {len(df)}")

    print(f"Total columns: {len(df.columns)}")

    print(f"First timestamp: {df['timestamp_utc'].min()}")

    print(f"Last timestamp: {df['timestamp_utc'].max()}")

    print("\nColumn names:")

    print(df.columns.tolist())

    print("\nMissing values:")

    print(df.isnull().sum())

    print("\nDataset preview:")

    print(df.head())


    return df


# --------------------------------------------------
# 8. Execute data loading
# --------------------------------------------------

if __name__ == "__main__":

    historical_df = load_historical_data()
    periods_per_day = inspect_data_quality(historical_df)


    print("\nHistorical dataset is ready for analysis.")
