import os

import pandas as pd
from dotenv import load_dotenv, find_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from giza_actions.model import GizaModel

from src.data_preprocessing.baseline_lstm_preprocessing import get_train_test
from src.data_preprocessing.data_handlers import load_data

load_dotenv(find_dotenv())

def plot_histograms(df):
    f, axes = plt.subplots(5, 4, figsize=(24, 20))
    for i, col in enumerate(df.columns):
        if col == "date":
            continue
        sns.histplot(df[col], ax=axes[int(i // 4), int(i % 4)])
    plt.show()


def plot_prediction_lstm():
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test(df)
    model = GizaModel(model_path=os.path.join(os.environ["ONNX_DIR"], "lstm_model.onnx"))
    result = model.predict(input_feed={"input": np.expand_dims(X_test, -1)}, verifiable=False)
    df_result = pd.DataFrame(np.hstack([np.arange(len(result)).reshape(-1, 1), y_test.reshape(-1, 1), result]), columns=["x", "target", "prediction"])
    sns.lineplot(data=df_result[-200:], x="x", y="target")
    sns.lineplot(data=df_result[-200:], x="x", y="prediction")
    plt.show()
    ((df_result["target"] - df_result["prediction"]) ** 2).mean(axis=0)
    # Convert result to a PyTorch tensor
    result_tensor = torch.tensor(result)

plot_prediction_lstm()

