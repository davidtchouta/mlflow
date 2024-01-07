import logging
from zenml import step
import pandas as pd

@step
def evaluate_model(df: pd.DataFrame) ->None:


    pass






# import logging

# import mlflow
# import numpy as np
# import pandas as pd
# from model.evaluation import Mae, R2
# from sklearn.ensemble import RandomForestRegressor
# from typing_extensions import Annotated
# from zenml import step
# from zenml.client import Client

# experiment_tracker = Client().active_stack.experiment_tracker
# from typing import Tuple

# #@step(experiment_tracker=experiment_tracker.name)
# @step()
# def evaluation(
#     model:RandomForestRegressor, x_test: pd.DataFrame, y_test: pd.Series
# ) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "mae_metric"]]:
#     """
#     Args:
#         model: RegressorMixin
#         x_test: pd.DataFrame
#         y_test: pd.Series
#     Returns:
#         r2_score: float
#         rmse: float
#     """
#     try:
#         # prediction = model.predict(x_test)
#         # evaluation = Evaluation()
#         # r2_score = evaluation.r2_score(y_test, prediction)
#         # mlflow.log_metric("r2_score", r2_score)
#         # mse = evaluation.mean_squared_error(y_test, prediction)
#         # mlflow.log_metric("mse", mse)
#         # rmse = np.sqrt(mse)
#         # mlflow.log_metric("rmse", rmse)

#         prediction = model.predict(x_test)

#         # Using the MSE class for mean squared error calculation
#         mae_class = Mae()
#         mae = mae_class.calculate_score(y_test, prediction)
#         mlflow.log_metric("Mae", mae)

#         # Using the R2Score class for R2 score calculation
#         r2_class = R2()
#         r2_score = r2_class.calculate_score(y_test, prediction)
#         mlflow.log_metric("r2_score", r2_score)
        
#         return r2_score, mae
#     except Exception as e:
#         logging.error(e)
#         raise e
