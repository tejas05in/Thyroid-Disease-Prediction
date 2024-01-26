from ThyroidProject import logger
from ThyroidProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ThyroidProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from ThyroidProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME = 'Data Ingestion Stage'

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Data Validation Stage'

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Data Transformation Stage'

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================x")
except Exception as e:
    logger.exception(e)
    raise e