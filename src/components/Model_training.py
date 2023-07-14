from src.Logger import logging
from src.Exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import os
import sys
from sklearn.metrics import r2_score


@dataclass
class Model_Training_Config:
    model_file_path = os.path.join("artifacts", "model.pkl")


class Model_Training:
    def __init__(self):
        self.model_path = Model_Training_Config()

    def initialize_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Initializing the model training")

            # Instantiate model
            # Instantiate linear regression model
            model = LinearRegression(
                copy_X=False, fit_intercept=True, n_jobs=9, positive=False
            )

            # Training the model
            model.fit(X_train, y_train)
            logging.info("Completed model training")

            # Let's now save the pipeline
            save_object(file_path=self.model_path.model_file_path, obj=model)
            logging.info("Saved model object")

            # Let's get some prediction from the model
            y_pred = model.predict(X_test)
            logging.info("Got the predictions from model successfully")

            # Let's compute the r2 score
            r2 = r2_score(y_test, y_pred)
            logging.info("Model training completed")

            return r2

        except Exception as e:
            raise CustomException(e, sys)
