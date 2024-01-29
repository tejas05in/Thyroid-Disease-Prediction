from ThyroidProject.constants import *
from ThyroidProject.utils.common import read_yaml, create_directories
from ThyroidProject.entity.config_entity import (DataIngestionConfig,
                                                 DataValidationConfig,
                                                 DataTransformationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig,
                                                 DriftMonitoringConfig)


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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
            This function returns the data transformation configuration.

            Args:
                self (DataTransformation): An instance of the DataTransformation class.

            Returns:
                DataTransformationConfig: The data transformation configuration.

            """
        config = self.config.data_transformation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            columns=schema

        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        This function returns the ModelTrainerConfig object that contains the configuration for the model training process.

        Args:
            None

        Returns:
            ModelTrainerConfig: The ModelTrainerConfig object containing the configuration for the model training process

        """
        config = self.config.model_trainer
        params = self.params.GradientBoostedTreesLearner
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            model_name=config.model_name,
            parameters=params,
            target_column=schema.name,
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        This function returns the ModelEvaluationConfig object that contains all the configuration parameters required for model evaluation.

        Args:
            None

        Returns:
            ModelEvaluationConfig: The ModelEvaluationConfig object containing all the configuration parameters

        """
        config = self.config.model_evaluation
        params = self.params.GradientBoostedTreesLearner
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/tejas05in/Thyroid-Disease-Prediction.mlflow"
        )

        return model_evaluation_config

    def get_drift_monitoring_config(self) -> DriftMonitoringConfig:
        """
        Get the Drift Monitoring Config.

        Returns:
            DriftMonitoringConfig: The Drift Monitoring Config.

        """
        # Load the config file
        config = self.config.drift_monitoring
        # Load the params file
        params = self.params.GradientBoostedTreesLearner
        # Load the schema file
        schema = self.schema.TARGET_COLUMN

        # Create the directories
        create_directories([config.root_dir])

        # Create the DriftMonitoringConfig object
        drift_monitoring_config = DriftMonitoringConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            report_path_name=config.report_path_name,
            test_path_name=config.test_path_name,
            target_column=schema.name
        )

        return drift_monitoring_config
