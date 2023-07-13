import os
import sys
import pandas as pd
from src.Logger import logging
from sklearn.model_selection import train_test_split
from src.Exception import CustomException
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.storage_paths = DataIngestionConfig()

    def initialize_data_ingestion(self, path):
        try:
            logging.info("Initializing data ingestion")

            # First we will load the csv data file from source directory
            df = pd.read_csv(path)
            logging.info("Loaded the csv file successfully")

            # Let's create a directory to store the train and test csv files
            os.makedirs(
                os.path.dirname(self.storage_paths.train_data_path), exist_ok=True
            )

            # Now we will do the train test split
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)
            logging.info("Completed train and test split")

            # Let's now store the files
            train_data.to_csv(
                self.storage_paths.train_data_path, index=False, header=True
            )
            test_data.to_csv(
                self.storage_paths.test_data_path, index=False, header=True
            )
            logging.info("Stored the files successfully")

            return (
                self.storage_paths.train_data_path,
                self.storage_paths.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingest_obj = DataIngestion()
    train_data_path, test_data_path = data_ingest_obj.initialize_data_ingestion(
        "notebook/dataset.csv"
    )
