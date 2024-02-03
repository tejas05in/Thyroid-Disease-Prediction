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

        # Hyperparameter templates
        templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
        model = ydf.GradientBoostedTreesLearner(label=self.config.target_column,
                                                task=ydf.Task.CLASSIFICATION, **templates["better_defaultv1"], **self.config.parameters).train(train_data)
        model.save(os.path.join(self.config.root_dir, self.config.model_name))
        # Training Description of the model
        with open(os.path.join(self.config.root_dir, "model_description.html"), "w") as f:
            f.write(model.describe()._html)
