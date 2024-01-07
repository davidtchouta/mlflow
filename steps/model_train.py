
import logging
import pandas as pd
from zenml import step

@step
def train_model(df:pd.DataFrame)->None:
    pass










# import logging

# import mlflow
# import pandas as pd
# from model.model_dev import (
#     GradientBoostingModel,
#     LinearRegressionModel,
#     RandomForestModel,
#     SVRModel,
# )
# from sklearn.ensemble import RandomForestRegressor
# from zenml import step
# from zenml.client import Client

# from .config import ModelNameConfig

# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker="epx1")
# def train_model(
#     x_train: pd.DataFrame,
#     x_test: pd.DataFrame,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     config: ModelNameConfig,
# ) -> RandomForestRegressor:
#     """
#     Args:
#         x_train: pd.DataFrame
#         x_test: pd.DataFrame
#         y_train: pd.Series
#         y_test: pd.Series
#     Returns:
#         model: RegressorMixin
#     """
#     try:
#         model = None
#         tuner = None

#         if config.model_name == "svr":
#             mlflow.svr.autolog()
#             model = SVRModel()
#         elif config.model_name == "randomforest":
#             mlflow.sklearn.autolog()
#             model = RandomForestModel()
#         elif config.model_name == "gboost":
#             mlflow.xgboost.autolog()
#             model = GradientBoostingModel()
#         elif config.model_name == "linear_regression":
#             mlflow.sklearn.autolog()
#             model = LinearRegressionModel()
#         else:
#             raise ValueError("Model name not supported")



#         trained_model = model.train(x_train, y_train)

#     except Exception as e:
#         logging.error(e)
#         raise e