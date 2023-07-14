import sys
import numpy as np
import os
import pandas as pd
from src.Logger import logging
from src.Exception import CustomException
from src.utils import save_object
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from dataclasses import dataclass


@dataclass
class Data_Processing_Config:
    processing_pipeline_path = os.path.join("artifacts", "process_data.pkl")


class Data_Ingestion:
    def __init__(self):
        self.pipeline_path = Data_Processing_Config()

    def initialize_data_processing(self, train_data_path, test_data_path):
        try:
            logging.info("Initializing data processing")

            # First let's load train and test data from csv files
            train_data_df = pd.read_csv(train_data_path)
            test_data_df = pd.read_csv(test_data_path)
            logging.info("Successfully loaded the csv files")

            # Dropping some unnecessary columns
            train_data_df.drop(
                [
                    "State FIPS",
                    "Occupation Code",
                    "Projected Year",
                    "Base Year",
                    "Change",
                ],
                axis=1,
                inplace=True,
            )
            test_data_df.drop(
                [
                    "State FIPS",
                    "Occupation Code",
                    "Projected Year",
                    "Base Year",
                    "Change",
                ],
                axis=1,
                inplace=True,
            )
            logging.info("Removed unnecessary columns")

            # Let's now create X_train,y_train, X_test and y_test
            X_train = train_data_df[
                [
                    "Area Name",
                    "Occupation Name",
                    "Base",
                    "Projection",
                    "Average Annual Openings",
                ]
            ]
            y_train = train_data_df[["Percent Change"]]
            X_test = test_data_df[
                [
                    "Area Name",
                    "Occupation Name",
                    "Base",
                    "Projection",
                    "Average Annual Openings",
                ]
            ]
            y_test = test_data_df[["Percent Change"]]
            logging.info("X_train,y_train,X_test,y_test created")

            y_train = np.array(y_train.values)
            y_test = np.array(y_test.values)

            # Let's now build a pipeline
            Yeo_john_trans = ColumnTransformer(
                transformers=[
                    ("Yeo_Johnson_Transformation", PowerTransformer(), [2, 3, 4])
                ],
                remainder="passthrough",
            )

            target_encode_transformer = ColumnTransformer(
                transformers=[
                    (
                        "Target_Encoding",
                        ce.TargetEncoder(
                            smoothing=0.2, handle_missing="return_nan", return_df=False
                        ),
                        [3, 4],
                    )
                ],
                remainder="passthrough",
            )

            scaling_transformer = ColumnTransformer(
                transformers=[("Robust_scaler", RobustScaler(), [2, 3, 4])],
                remainder="passthrough",
            )

            pipe = Pipeline(
                steps=[
                    ("Yeo-John", Yeo_john_trans),
                    ("Target_Encoder", target_encode_transformer),
                    ("Robust_Scaling", scaling_transformer),
                ]
            )

            # Process the training data and testing data
            X_train = pipe.fit_transform(X_train, y_train)
            X_test = pipe.transform(X_test)
            logging.info("Training and testing processing through pipeline completed")

            # Let's now save the pipeline
            save_object(file_path=self.pipeline_path.processing_pipeline_path, obj=pipe)
            logging.info("Saved pipeilne object")

            logging.info("Data processsing completed")
            return (X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)
