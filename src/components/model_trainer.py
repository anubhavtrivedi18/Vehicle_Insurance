import sys
from typing import Tuple

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from src.exception import MyException
from src.logger import logging

from src.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object
)

from src.entity.config_entity import ModelTrainerConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

from src.entity.estimator import MyModel


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """
        :param data_transformation_artifact:
            Output reference of data transformation artifact stage

        :param model_trainer_config:
            Configuration for model training
        """

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self,
        train: np.array,
        test: np.array
    ) -> Tuple[object, object]:

        """
        Method Name : get_model_object_and_report

        Description :
            This function trains a RandomForestClassifier
            with specified parameters and calculates
            evaluation metrics.

        Output :
            Returns metric artifact object and trained model object.

        On Failure :
            Write an exception log and raise an exception.
        """

        try:

            logging.info(
                "Training RandomForestClassifier with specified parameters"
            )

            # ---------------------------------------------------------
            # Splitting train and test data
            # ---------------------------------------------------------

            x_train = train[:, :-1]
            y_train = train[:, -1]

            x_test = test[:, :-1]
            y_test = test[:, -1]

            logging.info("Train-test split done.")

            # ---------------------------------------------------------
            # Initialize RandomForestClassifier
            # ---------------------------------------------------------

            model = RandomForestClassifier(

                n_estimators=self.model_trainer_config._n_estimators,

                min_samples_split=self.model_trainer_config._min_samples_split,

                min_samples_leaf=self.model_trainer_config._min_samples_leaf,

                max_depth=self.model_trainer_config._max_depth,

                criterion=self.model_trainer_config._criterion,

                random_state=self.model_trainer_config._random_state
            )

            # ---------------------------------------------------------
            # MLflow - Log Model Parameters
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.log_params({
                    "model": "RandomForestClassifier",

                    "n_estimators":
                        self.model_trainer_config._n_estimators,

                    "min_samples_split":
                        self.model_trainer_config._min_samples_split,

                    "min_samples_leaf":
                        self.model_trainer_config._min_samples_leaf,

                    "max_depth":
                        self.model_trainer_config._max_depth,

                    "criterion":
                        self.model_trainer_config._criterion,

                    "random_state":
                        self.model_trainer_config._random_state
                })

            # ---------------------------------------------------------
            # Train Model
            # ---------------------------------------------------------

            logging.info("Model training going on...")

            model.fit(x_train, y_train)

            logging.info("Model training done.")

            # ---------------------------------------------------------
            # Predictions
            # ---------------------------------------------------------

            y_pred = model.predict(x_test)

            # ---------------------------------------------------------
            # Evaluation Metrics
            # ---------------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            logging.info(
                f"Test Accuracy: {accuracy}"
            )

            logging.info(
                f"Test F1 Score: {f1}"
            )

            logging.info(
                f"Test Precision: {precision}"
            )

            logging.info(
                f"Test Recall: {recall}"
            )

            # ---------------------------------------------------------
            # MLflow - Log Metrics
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.log_metrics({

                    "test_accuracy": accuracy,

                    "test_f1_score": f1,

                    "test_precision": precision,

                    "test_recall": recall
                })

            # ---------------------------------------------------------
            # Create Metric Artifact
            # ---------------------------------------------------------

            metric_artifact = ClassificationMetricArtifact(

                f1_score=f1,

                precision_score=precision,

                recall_score=recall
            )

            return model, metric_artifact

        except Exception as e:

            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        logging.info(
            "Entered initiate_model_trainer method of ModelTrainer class"
        )

        try:

            print(
                "------------------------------------------------------------------------------------------------"
            )

            print(
                "Starting Model Trainer Component"
            )

            # ---------------------------------------------------------
            # Load transformed train and test data
            # ---------------------------------------------------------

            train_arr = load_numpy_array_data(
                file_path=
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                file_path=
                self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info(
                "Train-test data loaded"
            )

            # ---------------------------------------------------------
            # Train model and get metrics
            # ---------------------------------------------------------

            trained_model, metric_artifact = (
                self.get_model_object_and_report(
                    train=train_arr,
                    test=test_arr
                )
            )

            logging.info(
                "Model object and artifact loaded."
            )

            # ---------------------------------------------------------
            # Load preprocessing object
            # ---------------------------------------------------------

            preprocessing_obj = load_object(
                file_path=
                self.data_transformation_artifact.transformed_object_file_path
            )

            logging.info(
                "Preprocessing object loaded."
            )

            # ---------------------------------------------------------
            # Calculate training accuracy
            # ---------------------------------------------------------

            train_predictions = trained_model.predict(
                train_arr[:, :-1]
            )

            train_accuracy = accuracy_score(
                train_arr[:, -1],
                train_predictions
            )

            logging.info(
                f"Training Accuracy: {train_accuracy}"
            )

            # ---------------------------------------------------------
            # MLflow - Log Training Accuracy
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.log_metric(
                    "train_accuracy",
                    train_accuracy
                )

            # ---------------------------------------------------------
            # Check model accuracy against expected threshold
            # ---------------------------------------------------------

            if (
                train_accuracy
                < self.model_trainer_config.expected_accuracy
            ):

                logging.info(
                    "No model found with score above the base score"
                )

                if mlflow.active_run():

                    mlflow.set_tag(
                        "model_status",
                        "REJECTED"
                    )

                raise Exception(
                    "No model found with score above the base score"
                )

            # ---------------------------------------------------------
            # Create final MyModel object
            # ---------------------------------------------------------

            logging.info(
                "Saving new model as performance is better than previous one."
            )

            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model
            )

            # ---------------------------------------------------------
            # Save final model
            # ---------------------------------------------------------

            save_object(
                self.model_trainer_config.trained_model_file_path,
                my_model
            )

            logging.info(
                "Saved final model object that includes both "
                "preprocessing and trained model"
            )

            # ---------------------------------------------------------
            # MLflow - Log RandomForest Model
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    name="random_forest_model"
                )

                logging.info(
                    "RandomForest model logged to MLflow."
                )

            # ---------------------------------------------------------
            # MLflow - Log Final MyModel Artifact
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.log_artifact(
                    self.model_trainer_config.trained_model_file_path,
                    artifact_path="final_model"
                )

                logging.info(
                    "Final MyModel artifact logged to MLflow."
                )

            # ---------------------------------------------------------
            # Create ModelTrainerArtifact
            # ---------------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(

                trained_model_file_path=
                    self.model_trainer_config.trained_model_file_path,

                metric_artifact=metric_artifact
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:

            raise MyException(e, sys) from e