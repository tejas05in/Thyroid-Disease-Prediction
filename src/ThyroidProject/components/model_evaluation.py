import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from ThyroidProject.utils.common import save_json
from mlflow.data.pandas_dataset import PandasDataset
from ThyroidProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def log_into_mlflow(self):
        """
        This function logs the metrics and model into mlflow.

        Args:
            None

        Returns:
            None

        """

        # Load the test data
        test_data = pd.read_csv(self.config.test_data_path)

        # Convert the pandas dataframe to a tf.data.Dataset
        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
            test_data, label=self.config.target_column)

        # Load the model
        model = tf.keras.models.load_model(self.config.model_path)

        # Compile the model
        model.compile(metrics=["accuracy"])

        # Evaluate the model on the test data
        evaluation = model.evaluate(test_ds, return_dict=True)

        # Save the metrics as a local file
        # scores = {"accuray": evaluation["accuracy"], "loss": evaluation["loss"]}
        save_json(path=Path(self.config.metric_file_name), data=evaluation)

        # Set the mlflow tracking uri
        mlflow.set_registry_uri(self.config.mlflow_uri)

        # Get the type of the tracking uri
        tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        dataset: PandasDataset = mlflow.data.from_pandas(
            test_data)

        # Start a mlflow run
        with mlflow.start_run():

            # log data
            mlflow.log_input(dataset, context="testing")

            # Log the parameters
            mlflow.log_params(self.config.all_params)

            # Log the metrics
            mlflow.log_metric("loss", evaluation["loss"])
            mlflow.log_metric("accuray", evaluation["accuracy"])

            # Register the model
            if tracking_uri_type_store != 'file':
                mlflow.tensorflow.log_model(
                    model, 'model', registered_model_name="GradientBoostedTreesModel", conda_env="conda.yaml")
            else:
                mlflow.tensorflow.log_model(model, 'model')
