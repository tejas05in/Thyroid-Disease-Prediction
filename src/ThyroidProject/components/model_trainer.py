import pandas as pd
from ThyroidProject import logger
from ThyroidProject.entity.config_entity import ModelTrainerConfig
import tensorflow_decision_forests as tfdf
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        trains the model using the training data and saves it to the artifacts directory
        """
        train_data = pd.read_csv(self.config.train_data_path)

        tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
            train_data, label=self.config.target_column, max_num_classes=3)
        print(tf_dataset)

        model = tfdf.keras.GradientBoostedTreesModel(
            **self.config.parameters
        )
        model.fit(tf_dataset)
        inspector = model.make_inspector()
        logger.info(
            f"The results of the model building are: {inspector.training_logs()}")
        inspector.export_to_tensorboard(os.path.join(
            self.config.root_dir, "tensorboard_logs"))
        # saving the trained model for serving
        model.save(os.path.join(self.config.root_dir, self.config.model_name))
        model.save('model')
