from ThyroidProject.config.configuration import ConfigurationManager
from ThyroidProject.components.model_trainer import ModelTrainer
from ThyroidProject import logger

STAGE_NAME = 'Model Trainer Stage'


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main function of the ModelTrainerTrainingPipeline class.

        This function is responsible for:

        1. Getting the model training configuration from the configuration file.
        2. Creating an instance of the ModelTrainer class using the configuration.
        3. Calling the train method of the ModelTrainer class.

        Returns:
            None

        Raises:
            Exception: If an error occurs during training.
        """

        # Get the model training configuration from the configuration file
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        # Create an instance of the ModelTrainer class using the configuration
        model_trainer_config = ModelTrainer(config=model_trainer_config)

        # Call the train method of the ModelTrainer class
        model_trainer_config.train()

        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
    except Exception as e:
        logger.exception(e)
        raise e
