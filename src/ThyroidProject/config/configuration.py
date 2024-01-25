from ThyroidProject.constants import *
from ThyroidProject.utils.common import read_yaml, create_directories
from ThyroidProject.entity.config_entity import (DataIngestionConfig,
                                                 DataValidationConfig)


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        """
        Initialize the ConfigurationManager class.

        Args:
            config_filepath (str): The path to the configuration file.
            params_filepath (str): The path to the parameters file.
            schema_filepath (str): The path to the schema file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the configuration, parameters, or schema files cannot be found.
            ValueError: If the configuration or parameters files are not valid YAML files.
        """
        
        # Read the configuration, parameters, and schema files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Create the directories if they do not exist
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the data ingestion configuration.

        Returns:
            DataIngestionConfig: The data ingestion configuration.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get the data validation configuration.

        Returns:
            DataValidationConfig: The data validation configuration.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
