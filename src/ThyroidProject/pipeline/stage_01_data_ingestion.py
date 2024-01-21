from ThyroidProject.config.configuration import ConfigurationManager
from ThyroidProject.components.data_ingestion import DataIngestion
from ThyroidProject import logger

STAGE_NAME = 'Data Ingestion Stage'


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        This function is the main entry point of the Data Ingestion Training Pipeline.

        It initializes the configuration manager, retrieves the data ingestion configuration,
        instantiates the data ingestion component, and calls the download_file and extract_zip_file methods.

        :return: None
        :raises: Exception
        """
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
    except Exception as e:
        logger.exception(e)
        raise e
