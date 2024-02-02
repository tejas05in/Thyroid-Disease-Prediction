import pandas as pd
from ThyroidProject import logger
from ThyroidProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate all the columns in the data.

        This function reads the data from the unzip data directory and checks if all the columns are present in the schema and if their data types match with the schema. If any column is missing or has an incorrect data type, the function sets the validation status to False and writes an error message to the status file. If all the columns are valid, the function sets the validation status to True and writes an success message to the status file.

        Returns:
            bool: True if all the columns are valid, False otherwise.

        Raises:
            Exception: If there is an error while validating the columns.
        """
        try:
            validation_status = None

            # Column names of the raw data
            names = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antihyroid_meds", "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
                     "tumor", "hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG", "referral_source", "target"]

            # Read the data from the unzip data directory
            data = pd.read_csv(self.config.unzip_data_dir, names=names)

            # Get a list of all the columns
            all_cols = list(data.columns)

            # Get the schema for all the columns
            all_schema = self.config.all_schema.keys()

            # Loop through all the columns
            for col in all_cols:
                # Check if the column is present in the schema and if its data type matches with the schema
                if col not in all_schema or data[col].dtype != self.config.all_schema[col]:
                    # Set the validation status to False if any column is missing or has an incorrect data type
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                # Set the validation status to True if all the columns are valid
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")

            # Log the validation status
            logger.info(f"Validation status: {validation_status}")

            # Return the validation status
            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e
