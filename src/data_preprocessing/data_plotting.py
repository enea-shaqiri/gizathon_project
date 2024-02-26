import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from dotenv import load_dotenv, find_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from giza_actions.model import GizaModel

from gizathon_project.src.data_preprocessing.baseline_lstm_preprocessing import get_train_test as get_train_test_lstm
from gizathon_project.src.data_preprocessing.fc_preprocessing import get_train_test as get_train_test_fc
from gizathon_project.src.data_preprocessing.rf_preprocessing import get_train_test as get_train_test_rf
from gizathon_project.src.data_preprocessing.data_handlers import load_data

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
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_lstm(df)
    model = GizaModel(model_path=os.path.join(os.environ["ONNX_DIR"], "lstm_model.onnx"))
    train_results = model.predict(input_feed={"input": np.expand_dims(X_train, -1)}, verifiable=False)
    df_train_results = pd.DataFrame(np.hstack([np.arange(len(train_results)).reshape(-1, 1), y_train.reshape(-1, 1), train_results]), columns=["x", "target", "prediction"])
    test_results = model.predict(input_feed={"input": np.expand_dims(X_test, -1)}, verifiable=False)
    df_test_results = pd.DataFrame(np.hstack([np.arange(len(test_results)).reshape(-1, 1), y_test.reshape(-1, 1), test_results]),columns=["x", "target", "prediction"])
    print(f'MSE on training data {((df_train_results["target"] - df_train_results["prediction"]) ** 2).mean(axis=0)}')
    print(f'MSE from test data {((df_test_results["target"] - df_test_results["prediction"]) ** 2).mean(axis=0)}')
    print("10 days training samples")
    for j in [5000, 10000, 15000, 20000, 25000]:
        sns.lineplot(data=df_train_results[j-200: j], x="x", y="target", label="Target").set_title("Training")
        sns.lineplot(data=df_train_results[j-200: j], x="x", y="prediction", label="Prediction")
        plt.show()
    print("10 days testing samples")
    for j in [700, 1400, 2100, 2800, 3400]:
        sns.lineplot(data=df_test_results[j-200: j], x="x", y="target", label="Target").set_title("Test")
        sns.lineplot(data=df_test_results[j-200: j], x="x", y="prediction", label="Prediction")
        plt.show()

def plot_prediction_fc():
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_fc(df)
    model = GizaModel(model_path=os.path.join(os.environ["ONNX_DIR"], "fc_model.onnx"))
    train_results = model.predict(input_feed={"input": X_train}, verifiable=False)
    df_train_results = pd.DataFrame(np.hstack([np.arange(len(train_results)).reshape(-1, 1), y_train.reshape(-1, 1), train_results]),columns=["x", "target", "prediction"])
    test_results = model.predict(input_feed={"input": X_test}, verifiable=False)
    df_test_results = pd.DataFrame(np.hstack([np.arange(len(test_results)).reshape(-1, 1), y_test.reshape(-1, 1), test_results]),columns=["x", "target", "prediction"])
    print(f'MSE on training data {((df_train_results["target"] - df_train_results["prediction"]) ** 2).mean(axis=0)}')
    print(f'MSE from test data {((df_test_results["target"] - df_test_results["prediction"]) ** 2).mean(axis=0)}')
    print("10 days training samples")
    for j in [5000, 10000, 15000, 20000, 25000]:
        sns.lineplot(data=df_train_results[j-200: j], x="x", y="target", label="Target").set_title("Training")
        sns.lineplot(data=df_train_results[j-200: j], x="x", y="prediction", label="Prediction")
        plt.show()
    print("10 days testing samples")
    for j in [700, 1400, 2100, 2800, 3400]:
        sns.lineplot(data=df_test_results[j-200: j], x="x", y="target", label="Target").set_title("Test")
        sns.lineplot(data=df_test_results[j-200: j], x="x", y="prediction", label="Prediction")
        plt.show()


def plot_prediction_rf():
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_rf(df)
    model = GizaModel(model_path=os.path.join(os.environ["ONNX_DIR"], "rf_model.onnx"))
    train_results = model.predict(input_feed={"float_input": X_train}, verifiable=False)
    df_train_results = pd.DataFrame(np.hstack([np.arange(len(train_results)).reshape(-1, 1), y_train.reshape(-1, 1), train_results]),columns=["x", "target", "prediction"])
    test_results = model.predict(input_feed={"float_input": X_test}, verifiable=False)
    df_test_results = pd.DataFrame(np.hstack([np.arange(len(test_results)).reshape(-1, 1), y_test.reshape(-1, 1), test_results]),columns=["x", "target", "prediction"])
    print(f'MSE on training data {((df_train_results["target"] - df_train_results["prediction"]) ** 2).mean(axis=0)}')
    print(f'MSE from test data {((df_test_results["target"] - df_test_results["prediction"]) ** 2).mean(axis=0)}')
    print("10 days training samples")
    for j in [5000, 10000, 15000, 20000, 25000]:
        sns.lineplot(data=df_train_results[j-200: j], x="x", y="target", label="Target").set_title("Training")
        sns.lineplot(data=df_train_results[j-200: j], x="x", y="prediction", label="Prediction")
        plt.show()
    print("10 days testing samples")
    for j in [700, 1400, 2100, 2800, 3400]:
        sns.lineplot(data=df_test_results[j-200: j], x="x", y="target", label="Target").set_title("Test")
        sns.lineplot(data=df_test_results[j-200: j], x="x", y="prediction", label="Prediction")
        plt.show()

if __name__ == "__main__":
    plot_prediction_lstm()
    plot_prediction_fc()
    plot_prediction_rf()

