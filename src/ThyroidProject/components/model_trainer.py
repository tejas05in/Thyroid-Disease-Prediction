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
        # test_data = pd.read_csv(self.config.test_data_path)

        tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
            train_data, label=self.config.target_column, max_num_classes=3)
        print(tf_dataset)

        model = tfdf.keras.GradientBoostedTreesModel(
            categorical_algorithm=self.config.categorical_algorithm,
            l1_regularization=self.config.l1_regularization,
            l2_categorical_regularization=self.config.l2_categorical_regularization,
            l2_regularization=self.config.l2_regularization,
            max_depth=self.config.max_depth,
            num_trees=self.config.num_trees
        )
        model.fit(tf_dataset)
        inspector = model.make_inspector()
        logger.info(
            f"The results of the model building are: {inspector.training_logs()}")
        inspector.export_to_tensorboard(os.path.join(
            self.config.root_dir, "tensorboard_logs"))
        model.save(os.path.join(self.config.root_dir, self.config.model_name))
