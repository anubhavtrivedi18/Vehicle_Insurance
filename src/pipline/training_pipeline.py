import sys
import mlflow

from src.exception import MyException
from src.logger import logging

from src.utils.mlflow_utils import setup_mlflow

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class TrainPipeline:

    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()

        self.data_validation_config = DataValidationConfig()

        self.data_transformation_config = DataTransformationConfig()

        self.model_trainer_config = ModelTrainerConfig()

        self.model_evaluation_config = ModelEvaluationConfig()

        self.model_pusher_config = ModelPusherConfig()


    # ==============================================================
    # DATA INGESTION
    # ==============================================================

    def start_data_ingestion(
        self
    ) -> DataIngestionArtifact:

        """
        This method starts the Data Ingestion component.
        """

        try:

            logging.info(
                "Entered start_data_ingestion method."
            )

            logging.info(
                "Getting data from MongoDB."
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=
                self.data_ingestion_config
            )

            data_ingestion_artifact = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                "Train and test data obtained from MongoDB."
            )

            logging.info(
                "Exited start_data_ingestion method."
            )

            return data_ingestion_artifact

        except Exception as e:

            raise MyException(e, sys) from e


    # ==============================================================
    # DATA VALIDATION
    # ==============================================================

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        """
        This method starts the Data Validation component.
        """

        try:

            logging.info(
                "Entered start_data_validation method."
            )

            data_validation = DataValidation(
                data_ingestion_artifact=
                data_ingestion_artifact,

                data_validation_config=
                self.data_validation_config
            )

            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )

            logging.info(
                "Data validation completed."
            )

            logging.info(
                "Exited start_data_validation method."
            )

            return data_validation_artifact

        except Exception as e:

            raise MyException(e, sys) from e


    # ==============================================================
    # DATA TRANSFORMATION
    # ==============================================================

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        """
        This method starts the Data Transformation component.
        """

        try:

            logging.info(
                "Entered start_data_transformation method."
            )

            data_transformation = DataTransformation(

                data_ingestion_artifact=
                data_ingestion_artifact,

                data_transformation_config=
                self.data_transformation_config,

                data_validation_artifact=
                data_validation_artifact
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                "Data transformation completed."
            )

            logging.info(
                "Exited start_data_transformation method."
            )

            return data_transformation_artifact

        except Exception as e:

            raise MyException(e, sys) from e


    # ==============================================================
    # MODEL TRAINING
    # ==============================================================

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        """
        This method starts the Model Trainer component.
        """

        try:

            logging.info(
                "Entered start_model_trainer method."
            )

            model_trainer = ModelTrainer(

                data_transformation_artifact=
                data_transformation_artifact,

                model_trainer_config=
                self.model_trainer_config
            )

            model_trainer_artifact = (
                model_trainer.initiate_model_trainer()
            )

            logging.info(
                "Model training completed."
            )

            logging.info(
                "Exited start_model_trainer method."
            )

            return model_trainer_artifact

        except Exception as e:

            raise MyException(e, sys) from e


    # ==============================================================
    # MODEL EVALUATION
    # ==============================================================

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:

        """
        This method starts the Model Evaluation component.
        """

        try:

            logging.info(
                "Entered start_model_evaluation method."
            )

            model_evaluation = ModelEvaluation(

                model_eval_config=
                self.model_evaluation_config,

                data_ingestion_artifact=
                data_ingestion_artifact,

                model_trainer_artifact=
                model_trainer_artifact
            )

            model_evaluation_artifact = (
                model_evaluation.initiate_model_evaluation()
            )

            logging.info(
                "Model evaluation completed."
            )

            logging.info(
                "Exited start_model_evaluation method."
            )

            return model_evaluation_artifact

        except Exception as e:

            raise MyException(e, sys) from e


    # ==============================================================
    # MODEL PUSHER
    # ==============================================================

    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:

        """
        This method starts the Model Pusher component.
        """

        try:

            logging.info(
                "Entered start_model_pusher method."
            )

            model_pusher = ModelPusher(

                model_evaluation_artifact=
                model_evaluation_artifact,

                model_pusher_config=
                self.model_pusher_config
            )

            model_pusher_artifact = (
                model_pusher.initiate_model_pusher()
            )

            logging.info(
                "Model successfully pushed to S3."
            )

            logging.info(
                "Exited start_model_pusher method."
            )

            return model_pusher_artifact

        except Exception as e:

            raise MyException(e, sys) from e


    # ==============================================================
    # COMPLETE TRAINING PIPELINE + MLFLOW
    # ==============================================================

    def run_pipeline(self) -> None:

        try:

            # ------------------------------------------------------
            # Setup MLflow
            # ------------------------------------------------------

            logging.info(
                "Setting up MLflow."
            )

            setup_mlflow()

            logging.info(
                "MLflow setup completed."
            )

            # ------------------------------------------------------
            # Start MLflow Run
            # ------------------------------------------------------

            with mlflow.start_run(
                run_name="Vehicle-Insurance-Training"
            ):

                try:

                    # ==================================================
                    # MLFLOW TAGS
                    # ==================================================

                    mlflow.set_tag(
                        "project",
                        "Vehicle-Insurance"
                    )

                    mlflow.set_tag(
                        "pipeline",
                        "Training-Pipeline"
                    )

                    mlflow.set_tag(
                        "framework",
                        "scikit-learn"
                    )

                    mlflow.set_tag(
                        "model_type",
                        "RandomForestClassifier"
                    )


                    # ==================================================
                    # 1. DATA INGESTION
                    # ==================================================

                    logging.info(
                        "========== DATA INGESTION =========="
                    )

                    data_ingestion_artifact = (
                        self.start_data_ingestion()
                    )


                    # ==================================================
                    # 2. DATA VALIDATION
                    # ==================================================

                    logging.info(
                        "========== DATA VALIDATION =========="
                    )

                    data_validation_artifact = (
                        self.start_data_validation(

                            data_ingestion_artifact=
                            data_ingestion_artifact
                        )
                    )


                    # ==================================================
                    # 3. DATA TRANSFORMATION
                    # ==================================================

                    logging.info(
                        "========== DATA TRANSFORMATION =========="
                    )

                    data_transformation_artifact = (
                        self.start_data_transformation(

                            data_ingestion_artifact=
                            data_ingestion_artifact,

                            data_validation_artifact=
                            data_validation_artifact
                        )
                    )


                    # ==================================================
                    # 4. MODEL TRAINING
                    # ==================================================

                    logging.info(
                        "========== MODEL TRAINING =========="
                    )

                    model_trainer_artifact = (
                        self.start_model_trainer(

                            data_transformation_artifact=
                            data_transformation_artifact
                        )
                    )


                    # ==================================================
                    # 5. MODEL EVALUATION
                    # ==================================================

                    logging.info(
                        "========== MODEL EVALUATION =========="
                    )

                    model_evaluation_artifact = (
                        self.start_model_evaluation(

                            data_ingestion_artifact=
                            data_ingestion_artifact,

                            model_trainer_artifact=
                            model_trainer_artifact
                        )
                    )


                    # ==================================================
                    # CHECK MODEL ACCEPTANCE
                    # ==================================================

                    if not model_evaluation_artifact.is_model_accepted:

                        logging.info(
                            "New model was NOT accepted."
                        )

                        mlflow.set_tag(
                            "final_status",
                            "MODEL_REJECTED"
                        )

                        logging.info(
                            "Training pipeline stopped because "
                            "model was rejected."
                        )

                        return None


                    # ==================================================
                    # 6. MODEL PUSHER
                    # ==================================================

                    logging.info(
                        "========== MODEL PUSHER =========="
                    )

                    model_pusher_artifact = (
                        self.start_model_pusher(

                            model_evaluation_artifact=
                            model_evaluation_artifact
                        )
                    )


                    # ==================================================
                    # FINAL STATUS
                    # ==================================================

                    mlflow.set_tag(
                        "final_status",
                        "MODEL_DEPLOYED"
                    )

                    logging.info(
                        "=============================================="
                    )

                    logging.info(
                        "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
                    )

                    logging.info(
                        "Model successfully pushed to S3."
                    )

                    logging.info(
                        f"Model Pusher Artifact: "
                        f"{model_pusher_artifact}"
                    )

                    logging.info(
                        "=============================================="
                    )


                # ======================================================
                # ERROR INSIDE MLFLOW RUN
                # ======================================================

                except Exception as e:

                    logging.error(
                        "Training pipeline failed."
                    )

                    # MLflow run is still active here,
                    # so we can successfully log the failure.

                    mlflow.set_tag(
                        "final_status",
                        "PIPELINE_FAILED"
                    )

                    mlflow.set_tag(
                        "error",
                        str(e)
                    )

                    raise


        except Exception as e:

            raise MyException(e, sys) from e