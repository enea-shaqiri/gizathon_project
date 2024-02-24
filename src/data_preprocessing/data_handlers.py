import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def load_data():
    return pd.read_csv(os.path.join(os.environ["DATA_DIR"], "dataset.csv"))


def create_date_features(df):
    months, days, hours = [], [], []
    s = pd.to_datetime(df.date)
    # Not using pd.to_datetime as it works with utc timezone.
    for j in range(len(s)):
        months.append(s.iloc[j].month)
        days.append(s.iloc[j].day)
        hours.append(s.iloc[j].hour)
    df["month"] = months
    df["day"] = days
    df["hour"] = hours
    df.drop(columns=["date"], inplace=True)
    return df


def create_rolling_features(df):
    columns = [col for col in df.columns if col != "date"]
    windows = [6, 12, 18, 24]
    for col in columns:
        for window in windows:
            df[f"rolling{window}_{col}_diff"] = df[col].shift(window) - df[col]
            df[f"rolling{window}_{col}_median"] = df[col].rolling(window).median()
            df[f"rolling{window}_{col}_std"] = df[col].rolling(window).std()
            df[f"rolling{window}_{col}_quantile25"] = df[col].rolling(window).quantile(0.25)
            df[f"rolling{window}_{col}_quantile75"] = df[col].rolling(window).quantile(0.75)
    df.dropna(inplace=True)
    rolling_columns = [col for col in df.columns if "rolling" in col]
    return df[["date"] + rolling_columns]

