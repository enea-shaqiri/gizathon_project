from warnings import simplefilter
import pandas as pd

from gizathon_project.src.data_preprocessing.data_handlers import create_date_features, create_rolling_features
import numpy as np
from giza_actions.task import task
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

def get_train_test(df, train_size=0.8):
    """
    It returns train, validation and test data.
    """
    df = create_rolling_features(df)
    df = create_date_features(df)
    X, y = [], []
    for j in range(len(df) - 1):
        X.append(df.iloc[j])
        y.append(df["price actual"].iloc[j + 1])
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    x_train = X[:int(len(X) * train_size)]
    y_train = y[:int(len(y) * train_size)]
    valid_size = int((len(X) - len(x_train)) / 2)
    x_valid = X[int(len(X) * train_size): int(len(X) * train_size) + valid_size]
    y_valid = y[int(len(y) * train_size): int(len(y) * train_size) + valid_size]
    x_test = X[int(len(X) * train_size) + valid_size:]
    y_test = y[int(len(y) * train_size) + valid_size:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


@task(name="Get train test split for Random Forest")
def get_train_test_task(df, train_size=0.8):
    return get_train_test(df, train_size)