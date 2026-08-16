from src.entity.config_entity import ModelEvaluationConfig

from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)

from sklearn.metrics import f1_score

from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging

from src.utils.main_utils import load_object

import sys
import pandas as pd
import mlflow

from typing import Optional

from src.entity.s3_estimator import Proj1Estimator

from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:

            self.model_eval_config = model_eval_config

            self.data_ingestion_artifact = data_ingestion_artifact

            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:

            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:

        """
        Method Name : get_best_model

        Description :
            This function is used to get model from production stage.

        Output :
            Returns model object if available in S3 storage.

        On Failure :
            Write an exception log and then raise an exception.
        """

        try:

            bucket_name = self.model_eval_config.bucket_name

            model_path = self.model_eval_config.s3_model_key_path

            proj1_estimator = Proj1Estimator(
                bucket_name=bucket_name,
                model_path=model_path
            )

            if proj1_estimator.is_model_present(
                model_path=model_path
            ):
                return proj1_estimator

            return None

        except Exception as e:

            raise MyException(e, sys) from e

    def _map_gender_column(self, df):

        """
        Map Gender column to:
        Female -> 0
        Male   -> 1
        """

        logging.info(
            "Mapping 'Gender' column to binary values"
        )

        df['Gender'] = (
            df['Gender']
            .map({
                'Female': 0,
                'Male': 1
            })
            .astype(int)
        )

        return df

    def _create_dummy_columns(self, df):

        """
        Create dummy variables for categorical features.
        """

        logging.info(
            "Creating dummy variables for categorical features"
        )

        df = pd.get_dummies(
            df,
            drop_first=True
        )

        return df

    def _rename_columns(self, df):

        """
        Rename specific columns and ensure integer
        types for dummy columns.
        """

        logging.info(
            "Renaming specific columns and casting to int"
        )

        df = df.rename(
            columns={
                "Vehicle_Age_< 1 Year":
                    "Vehicle_Age_lt_1_Year",

                "Vehicle_Age_> 2 Years":
                    "Vehicle_Age_gt_2_Years"
            }
        )

        for col in [
            "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes"
        ]:

            if col in df.columns:

                df[col] = df[col].astype('int')

        return df

    def _drop_id_column(self, df):

        """
        Drop the '_id' column if it exists.
        """

        logging.info(
            "Dropping '_id' column"
        )

        if "_id" in df.columns:

            df = df.drop(
                "_id",
                axis=1
            )

        return df

    def evaluate_model(self) -> EvaluateModelResponse:

        """
        Method Name : evaluate_model

        Description :
            Evaluate newly trained model against
            the production model and determine
            whether the new model should be accepted.

        Output :
            EvaluateModelResponse
        """

        try:

            # ---------------------------------------------------------
            # Load test data
            # ---------------------------------------------------------

            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            x = test_df.drop(
                TARGET_COLUMN,
                axis=1
            )

            y = test_df[TARGET_COLUMN]

            logging.info(
                "Test data loaded and now transforming it for prediction..."
            )

            # ---------------------------------------------------------
            # Transform test data
            # ---------------------------------------------------------

            x = self._map_gender_column(x)

            x = self._drop_id_column(x)

            x = self._create_dummy_columns(x)

            x = self._rename_columns(x)

            # ---------------------------------------------------------
            # Load newly trained model
            # ---------------------------------------------------------

            trained_model = load_object(
                file_path=
                self.model_trainer_artifact.trained_model_file_path
            )

            logging.info(
                "Trained model loaded/exists."
            )

            # ---------------------------------------------------------
            # Get F1 score of newly trained model
            # ---------------------------------------------------------

            trained_model_f1_score = (
                self.model_trainer_artifact
                .metric_artifact
                .f1_score
            )

            logging.info(
                f"F1_Score for this model: "
                f"{trained_model_f1_score}"
            )

            # ---------------------------------------------------------
            # MLflow - Log new model F1 score
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.log_metric(
                    "evaluation_trained_model_f1_score",
                    trained_model_f1_score
                )

            # ---------------------------------------------------------
            # Get production model
            # ---------------------------------------------------------

            best_model_f1_score = None

            best_model = self.get_best_model()

            # ---------------------------------------------------------
            # Compare with production model
            # ---------------------------------------------------------

            if best_model is not None:

                logging.info(
                    "Computing F1_Score for production model.."
                )

                y_hat_best_model = best_model.predict(x)

                best_model_f1_score = f1_score(
                    y,
                    y_hat_best_model
                )

                logging.info(
                    f"F1_Score-Production Model: "
                    f"{best_model_f1_score}, "
                    f"F1_Score-New Trained Model: "
                    f"{trained_model_f1_score}"
                )

                # -----------------------------------------------------
                # MLflow - Log production model score
                # -----------------------------------------------------

                if mlflow.active_run():

                    mlflow.log_metric(
                        "production_model_f1_score",
                        best_model_f1_score
                    )

            else:

                logging.info(
                    "No production model found. "
                    "New model will be compared against 0."
                )

            # ---------------------------------------------------------
            # Calculate comparison score
            # ---------------------------------------------------------

            tmp_best_model_score = (
                0
                if best_model_f1_score is None
                else best_model_f1_score
            )

            difference = (
                trained_model_f1_score
                - tmp_best_model_score
            )

            is_model_accepted = (
                trained_model_f1_score
                > tmp_best_model_score
            )

            # ---------------------------------------------------------
            # Create evaluation response
            # ---------------------------------------------------------

            result = EvaluateModelResponse(

                trained_model_f1_score=
                    trained_model_f1_score,

                best_model_f1_score=
                    best_model_f1_score,

                is_model_accepted=
                    is_model_accepted,

                difference=
                    difference
            )

            logging.info(
                f"Result: {result}"
            )

            # ---------------------------------------------------------
            # MLflow - Log evaluation results
            # ---------------------------------------------------------

            if mlflow.active_run():

                mlflow.log_metric(
                    "model_f1_score_difference",
                    difference
                )

                mlflow.set_tag(
                    "model_accepted",
                    str(is_model_accepted)
                )

                if best_model is not None:

                    mlflow.set_tag(
                        "production_model_available",
                        "True"
                    )

                else:

                    mlflow.set_tag(
                        "production_model_available",
                        "False"
                    )

                if is_model_accepted:

                    mlflow.set_tag(
                        "evaluation_status",
                        "ACCEPTED"
                    )

                else:

                    mlflow.set_tag(
                        "evaluation_status",
                        "REJECTED"
                    )

            return result

        except Exception as e:

            raise MyException(e, sys) from e

    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifact:

        """
        Method Name : initiate_model_evaluation

        Description :
            This function initiates all steps
            of model evaluation.

        Output :
            Returns model evaluation artifact.
        """

        try:

            print(
                "------------------------------------------------------------------------------------------------"
            )

            logging.info(
                "Initialized Model Evaluation Component."
            )

            # ---------------------------------------------------------
            # Evaluate model
            # ---------------------------------------------------------

            evaluate_model_response = (
                self.evaluate_model()
            )

            # ---------------------------------------------------------
            # Get S3 model path
            # ---------------------------------------------------------

            s3_model_path = (
                self.model_eval_config.s3_model_key_path
            )

            # ---------------------------------------------------------
            # Create ModelEvaluationArtifact
            # ---------------------------------------------------------

            model_evaluation_artifact = ModelEvaluationArtifact(

                is_model_accepted=
                    evaluate_model_response.is_model_accepted,

                s3_model_path=
                    s3_model_path,

                trained_model_path=
                    self.model_trainer_artifact
                    .trained_model_file_path,

                changed_accuracy=
                    evaluate_model_response.difference
            )

            logging.info(
                f"Model evaluation artifact: "
                f"{model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:

            raise MyException(e, sys) from e