from ThyroidProject.config.configuration import ConfigurationManager
from ThyroidProject.components.data_transformation import DataTransformation
from ThyroidProject import logger

STAGE_NAME = 'Data Transformation Stage'


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main function of the Data Transformation training pipeline.

        This function initializes the configuration manager, loads the data transformation configuration,
        initializes the data transformation component, and calls the data_clenser_splitter method of the data transformation component.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config)
        data_transformation.data_clenser_splitter()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
    except Exception as e:
        logger.exception(e)
        raise e
