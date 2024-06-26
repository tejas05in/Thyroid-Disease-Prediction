import pandas as pd
import numpy as np
import os
from ThyroidProject.utils.common import create_directories
from ThyroidProject.entity.config_entity import DriftMonitoringConfig

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, MulticlassClassificationTestPreset
from evidently.tests import *

import ydf


class DriftMonitoring:
    def __init__(self, config: DriftMonitoringConfig):
        self.config = config

    def generate_drift_reports(self):
        """
        Generate the Drift Reports.

        Returns:
            None
        """
        # creating directories
        create_directories(["drift_reports"])

        # Load the data
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        # Create the test and train features
        target = self.config.target_column
        X_train = train_df.drop(target, axis=1)
        X_test = test_df.drop(target, axis=1)

        # load model
        model = ydf.load_model(self.config.model_path)

        # Make predictions
        train_df["prediction"] = [np.argmax(i) for i in model.predict(X_train)]
        test_df["prediction"] = [np.argmax(i) for i in model.predict(X_test)]

        # Identifying categorical and numerical columns
        cat_col = train_df.select_dtypes(include="object").columns
        num_col = train_df.select_dtypes(exclude="object").columns
        num_col = num_col.drop([self.config.target_column, 'prediction'])

        # Create the column mapping for evidently
        column_mapping = ColumnMapping()
        column_mapping.target = self.config.target_column
        column_mapping.prediction = "prediction"
        column_mapping.numerical_features = cat_col
        column_mapping.categorical_features = num_col

        # Create the report
        report = Report(metrics=[ColumnSummaryMetric(column_name='T3'),
                                 ColumnSummaryMetric(column_name='TSH'),
                                 ColumnSummaryMetric(column_name='T4U'),
                                 ColumnSummaryMetric(column_name='TT4'),
                                 generate_column_metrics(ColumnQuantileMetric, parameters={
                                     'quantile': 0.25}, columns='num'),
                                 DataDriftPreset(),
                                 TargetDriftPreset(),
                                 DataQualityPreset(),
                                 ClassificationPreset()
                                 ])

        report.run(reference_data=train_df, current_data=test_df)

        # save the report
        report.save_html(self.config.report_path_name)
        report.save_html(os.path.join("drift_reports", "report.html"))

        # Perform tests
        test_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
            DataStabilityTestPreset(),
            NoTargetPerformanceTestPreset(),
            MulticlassClassificationTestPreset()
        ])

        test_suite.run(reference_data=train_df, current_data=test_df)

        # save the test suite
        test_suite.save_html(self.config.test_path_name)
        test_suite.save_html(os.path.join("drift_reports", "test.html"))
