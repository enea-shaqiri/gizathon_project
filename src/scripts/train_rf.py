import os.path
import numpy as np
from giza_actions.action import Action, action
from giza_actions.task import task
from dotenv import load_dotenv, find_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from config.rf_config import configs
from src.data_preprocessing.data_handlers import load_data
from src.data_preprocessing.rf_preprocessing import get_train_test_task

load_dotenv(find_dotenv())

@task(name='Convert To ONNX')
def convert_to_onnx(model, input_size, filename):
    initial_type = [('float_input', FloatTensorType([None, input_size]))]
    path = os.path.join(os.environ["ONNX_DIR"], filename)
    # # Zipmap should be always turned off as it's not implemented in TF3800
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model has been converted to ONNX and saved as {path}")


@task(name="Training!")
def train(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    return model


@action(name="Train Random forest", log_prints=True)
def main():
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_task(df)
    model = RandomForestRegressor(n_estimators=400, max_depth=4, min_samples_split=10, max_features=30, verbose=True)
    best_model = train(model, X_train, y_train, X_valid, y_valid)
    convert_to_onnx(best_model, len(X_train[0]), configs["filename"])


if __name__ == "__main__":
    main()
