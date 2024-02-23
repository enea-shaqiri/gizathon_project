import numpy as np
from giza_actions.task import task
from src.data_preprocessing.data_handlers import load_data


#@task(name="Get train test split for LSTM")
def get_train_test(df, window=24, train_size=0.8):
    """
    It returns train, validation and test data.
    """
    X, y = [], []
    for j in range(len(df) - window):
        X.append(df["price actual"][j: j + window])
        y.append(df["price actual"].iloc[j + window])
    X = np.array(X)
    y = np.array(y)
    x_train = X[:int(len(X) * train_size)]
    y_train = y[:int(len(y) * train_size)]
    valid_size = int(len(X) - len(x_train) / 2)
    x_valid = X[int(len(X) * train_size): int(len(X) * train_size) + valid_size]
    y_valid = y[int(len(y) * train_size): int(len(y) * train_size) + valid_size]
    x_test = X[int(len(X) * train_size) + valid_size:]
    y_test = y[int(len(y) * train_size) + valid_size:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


