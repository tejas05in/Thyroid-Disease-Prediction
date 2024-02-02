import os
from ThyroidProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ThyroidProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config=DataTransformationConfig):
        self.config = config

    def data_clenser_splitter(self):
        """
        This function takes in the data transformation configuration and splits the data into train and test sets.

        Args:
            self (DataTransformation): An instance of the DataTransformation class.

        Returns:
            None

        """
        try:
            df = pd.read_csv(self.config.data_path,
                             names=list(self.config.columns))

            # tidy the target column
            df['patient_id'] = df["target"].apply(
                lambda x: x.split("[")[1].strip(']'))
            df['target'] = df["target"].apply(lambda x: x.split("[")[0])

            # replacing ? with np.nan
            df.replace({"?": np.nan}, inplace=True)

            # converting object to float
            num_cols = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
            for i in num_cols:
                df[i] = df[i].astype(float)

            # age cannot be 65526
            # capping age to 100 years
            df = df[df["age"] <= 100]
            
            #drop duplicates
            df.drop_duplicates(inplace=True)

            # Remove reduntant columns
            df.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured',
                    'TBG_measured', 'referral_source', 'patient_id', "TBG"], axis=1, inplace=True)

            # Selecting a subset of target which can be classified as Hyper , hypo or Euthyroid (Negative) state
            df = df[df['target'].isin(['A', 'B', 'C', 'D', 'E', 'F', 'G',
                                       'H', 'AK', 'C|I', 'H|K', 'GK', 'FK', 'GI', 'GKJ', 'D|R', '-'])]
            # mapping the target column
            mapping = {'-': "Negative",
                       'A': 'Hyperthyroid', 'AK': "Hyperthyroid", 'B': "Hyperthyroid", 'C': "Hyperthyroid", 'C|I': 'Hyperthyroid', 'D': "Hyperthyroid", 'D|R': "Hyperthyroid",
                       'E': "Hypothyroid", 'F': "Hypothyroid", 'FK': "Hypothyroid", "G": "Hypothyroid", "GK": "Hypothyroid", "GI": "Hypothyroid", 'GKJ': 'Hypothyroid', 'H|K': 'Hypothyroid',
                       }
            df['target'] = df['target'].map(mapping)

            # impute some missing values of sex (total = 254 missing) using pregnancy
            df["sex"] = np.where((df["sex"].isnull()) & (
                df["pregnant"] == "t"), 'F', df["sex"])

            # replacing t with 1 and f with 0
            # df = df.replace({"t": 1, "f": 0}) # this will be onehot encoded

            # Mapping sex to 0 for female and 1 for male
            # df["sex"] = df["sex"].map({"F": 0, "M": 1}) # this will be onehot encoded

            # Mapping target
            df["target"] = df.target.map(
                {'Negative': 0, 'Hypothyroid': 1, 'Hyperthyroid': 2})

            # Split the data into train and test sets. (0.75 , 0.25) split
            train, test = train_test_split(df, random_state=0)

            train.to_csv(os.path.join(
                self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(
                self.config.root_dir, "test.csv"), index=False)

            logger.info(f"Spliting data into training and testing sets finished")
            logger.info(f"Train data shape is: {train.shape}")
            logger.info(f"Train data shape is: {test.shape}")
        except Exception as e:
            logger.exception(e)
            raise e
