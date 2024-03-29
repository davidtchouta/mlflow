from pipelines.training_pipeline import train_pipeline
from zenml.client import Client



if __name__ =="__main__":
    #run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="C:\\Users\\dvid\\Documents\\Python_ML\\Health_Insurance\\data\\1651277648862_healthinsurance.csv")
    

# from pipelines.training_pipeline import train_pipeline
# from steps.clean_data import clean_data
# from steps.evaluation import evaluation
# from steps.ingest_data import ingest_data
# from steps.model_train import train_model
# from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# if __name__ == "__main__":
#     training = train_pipeline(
#         ingest_data(),
#         clean_data(ingest_data()),
#         train_model(),
#         evaluation(),
#     )

#     training.run()

#     print(
#         "Now run \n "
#         f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
#         "To inspect your experiment runs within the mlflow UI.\n"
#         "You can find your runs tracked within the `mlflow_example_pipeline`"
#         "experiment. Here you'll also be able to compare the two runs.)"
#     )
