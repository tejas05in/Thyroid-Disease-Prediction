import pandas as pd
from urllib.parse import urlparse
import dagshub
import mlflow
import ydf
from mlflow.data.pandas_dataset import PandasDataset
from ThyroidProject.entity.config_entity import ModelEvaluationConfig


class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = ydf.load_model(
            context.artifacts["model"])
        return self.model

    def predict(self, model_input):
        return self.model.predict(model_input)


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

        # Load the model
        model = ydf.load_model(self.config.model_path)

        # Evaluate the model on the test data
        evaluation = model.evaluate(test_data)

        # storing the training description in a html file
        with open(self.config.metric_file_name, "w") as f:
            f.write(evaluation.html())

        # initialize the dagshub repo
        dagshub.init("Thyroid-Disease-Prediction", "tejas05in", mlflow=True)

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
            mlflow.log_metric("loss", evaluation.loss)
            mlflow.log_metric("accuray", evaluation.accuracy)

            # Register the model
            if tracking_uri_type_store != 'file':
                mlflow.pyfunc.log_model(
                    "model", python_model=CustomModelWrapper(), artifacts={"model": self.config.model_path}, registered_model_name="GradientBoostedTreesModel", pip_requirements='requirements.txt')
            else:
                mlflow.pyfunc.log_model("model", python_model=CustomModelWrapper(), artifacts={
                                        "model": self.config.model_path}, registered_model_name="GradientBoostedTreesModel")
