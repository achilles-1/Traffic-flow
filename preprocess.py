import os
import pandas as pd

DATA_PATH = "Metro_Interstate_Traffic_Volume.csv"
OUTPUT_PATH = "preprocess.csv"


def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Make sure "
            "Metro_Interstate_Traffic_Volume.csv "
            "is in the same folder as preprocess.py."
        )

    return pd.read_csv(DATA_PATH)


def preprocess_data(df):

    df = df.copy()

    # Convert date_time to datetime
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    # Remove rows with invalid date_time or traffic_volume
    df = df.dropna(
        subset=["date_time", "traffic_volume"]
    )

    # Sort data by date and time
    df = df.sort_values(
        "date_time"
    ).reset_index(drop=True)

    # Create hour feature
    df["hour"] = df["date_time"].dt.hour

    # Create day of week feature
    # 0 = Monday, 6 = Sunday
    df["day_of_week"] = df["date_time"].dt.dayofweek

    # Create month feature
    df["month"] = df["date_time"].dt.month

    # Create weekend feature
    # 0 = Weekday, 1 = Weekend
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Create peak-hour feature
    # Morning: 7, 8, 9
    # Evening: 16, 17, 18
    df["is_peak_hour"] = df["hour"].isin(
        [7, 8, 9, 16, 17, 18]
    ).astype(int)

    # Convert holiday into binary values
    # 0 = No holiday
    # 1 = Holiday
    df["holiday"] = df["holiday"].fillna("None")

    df["holiday"] = (
        df["holiday"]
        .apply(lambda x: 0 if x == "None" else 1)
    )

    # Select features required by the ML model
    columns = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "traffic_volume"
    ]

    df = df[columns]

    # Remove remaining missing values
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":

    # Load raw dataset
    data = load_data()

    # Preprocess dataset
    processed = preprocess_data(data)

    # Save processed dataset
    processed.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Display results
    print("Preprocessing completed successfully.")
    print("Rows:", len(processed))
    print("Columns:", len(processed.columns))
    print("Saved:", OUTPUT_PATH)
    print("\nColumns:")
    print(processed.columns.tolist())