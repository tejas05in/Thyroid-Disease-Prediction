from ThyroidProject.config.configuration import ConfigurationManager
from ThyroidProject.components.model_evaluation import ModelEvaluation
from ThyroidProject import logger

STAGE_NAME = 'Model Evaluation Stage'


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main function of the ModelTrainerTrainingPipeline class.

        This function initializes the ConfigurationManager and loads the model evaluation configuration.
        It then creates an instance of the ModelEvaluation class and logs it into MLflow.

        Returns:
            None
        """

        # Get the model evaluation configuration from the configuration file
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Create an instance of the ModelEvaluation class using the configuration
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # Log the model evaluation into MLflow
        model_evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
    except Exception as e:
        logger.exception(e)
        raise e
