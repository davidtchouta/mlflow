
from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df=ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mae, r2_score = evaluate_model(model, x_test, y_test)








# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW
# from zenml.pipelines import pipeline

# docker_settings = DockerSettings(required_integrations=[MLFLOW])


# @pipeline(enable_cache=False, settings={"docker": docker_settings})
# def train_pipeline(ingest_data, clean_data, model_train, evaluation):
#     """
#     Args:
#         ingest_data: DataClass
#         clean_data: DataClass
#         model_train: DataClass
#         evaluation: DataClass
#     Returns:
#         mse: float
#         rmse: float
#     """
    
#     df = ingest_data()
#     x_train, x_test, y_train, y_test = clean_data(df)
#     model = model_train(x_train, x_test, y_train, y_test)
#     mae, r2_score = evaluation(model, x_test, y_test)