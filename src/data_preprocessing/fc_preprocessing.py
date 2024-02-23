import numpy as np
import pandas as pd
from giza_actions.task import task
from src.data_preprocessing.data_handlers import load_data


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


#@task(name="Get train test split for FC")
def get_train_test(df, window=12, train_size=0.8):
    """
    It returns train, validation and test data.
    """
    df = create_date_features(df)
    X, y = [], []
    for j in range(len(df) - window):
        X.append(df[j: j + window])
        y.append(df["price actual"].iloc[j + window])
    X = np.array(X)
    X = X.reshape(len(X), -1)
    y = np.array(y)
    x_train = X[:int(len(X) * train_size)]
    y_train = y[:int(len(y) * train_size)]
    valid_size = int(len(X) - len(x_train) / 2)
    x_valid = X[int(len(X) * train_size): int(len(X) * train_size) + valid_size]
    y_valid = y[int(len(y) * train_size): int(len(y) * train_size) + valid_size]
    x_test = X[int(len(X) * train_size) + valid_size:]
    y_test = y[int(len(y) * train_size) + valid_size:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test

