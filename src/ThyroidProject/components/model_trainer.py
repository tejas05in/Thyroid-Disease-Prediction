import pandas as pd
from ThyroidProject import logger
from ThyroidProject.entity.config_entity import ModelTrainerConfig
import ydf
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        trains the model using the training data and saves it to the artifacts directory
        """
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        learner = ydf.GradientBoostedTreesLearner(
            categorical_algorithm=self.config.categorical_algorithm,
            l1_regularization=self.config.l1_regularization,
            l2_categorical_regularization=self.config.l2_categorical_regularization,
            l2_regularization=self.config.l2_regularization,
            max_depth=self.config.max_depth,
            num_trees=self.config.num_trees,
            label=self.config.target_column
        )
        model = learner.train(train_data)
        # assert model.task() == ydf.Task.CLASSIFICATION
        results = model.evaluate(test_data)
        logger.info(
            f"The accuracy of the model evaluation is: {results.accuracy:.4f}")
        model.save(os.path.join(self.config.root_dir, self.config.model_name))
