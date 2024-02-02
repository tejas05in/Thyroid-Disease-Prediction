from ThyroidProject.config.configuration import ConfigurationManager
from ThyroidProject.components.drift_monitoring import DriftMonitoring
from ThyroidProject import logger

STAGE_NAME = 'Drift Monitoring Stage'


class DriftMonitoringTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            drift_monitoring_config = config.get_drift_monitoring_config()
            drift_monitoring = DriftMonitoring(config=drift_monitoring_config)
            drift_monitoring.generate_drift_reports()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = DriftMonitoringTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
    except Exception as e:
        logger.exception(e)
        raise e
