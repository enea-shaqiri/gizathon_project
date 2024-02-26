import random
import sys
sys.path.append("/home/enea/PycharmProjects/")
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel

from gizathon_project.src.data_preprocessing.data_handlers import load_data
from gizathon_project.src.data_preprocessing.fc_preprocessing import get_train_test_task

MODEL_ID = 358
VERSION_ID = 3


@task(name=f'Prediction with Cairo')
def prediction(data, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    (result, request_id) = model.predict(
        input_feed={"input": data}, verifiable=True, output_dtype="Tensor<FP16x16>"
    )

    return result, request_id


@action(name=f'Execution: Prediction with Cairo', log_prints=True)
def execution():
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_task(df)
    (result, request_id) = prediction(X_test[random.randint(0, len(X_test))], MODEL_ID, VERSION_ID)
    print("The prediction for you sample is: ", result)
    print("Request id: ", request_id)

    return result, request_id


execution()