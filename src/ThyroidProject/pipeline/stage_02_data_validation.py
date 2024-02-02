from ThyroidProject.config.configuration import ConfigurationManager
from ThyroidProject.components.data_validation import DataValidation
from ThyroidProject import logger

STAGE_NAME = 'Data Validation Stage'


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main function of the Data Validation stage.

        This function is responsible for validating all the columns in the data.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If any error occurs during the execution of the function.
        """
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_columns()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
    except Exception as e:
        logger.exception(e)
        raise e
