import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn import metrics

class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class R2(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_score(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2")
            r2_score = metrics.r2_score(y_test, y_pred)
            logging.info("The R2 score value is: " + str(r2_score))
            return r2_score
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the R2 score. Exception message:  "
                + str(e)
            )
            raise e


class Mae(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_score(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2")
            mae_metric = metrics.mean_absolute_error(y_test, y_pred)
            logging.info("The Mean Absolute Error value is: " + str(mae_metric))
            return mae_metric
        except Exception as e:
            logging.error(
                "Exception occurred in Mean Absolute Error method . Exception message:  "
                + str(e)
            )
            raise e

