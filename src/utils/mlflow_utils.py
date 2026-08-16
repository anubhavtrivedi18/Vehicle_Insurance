import mlflow


def setup_mlflow():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5001"
    )

    mlflow.set_experiment(
        "Vehicle-Insurance"
    )