import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing import Tuple

@step
def clean_df(df: pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        process_strategy=DataPreprocessStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data= data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e



# import logging
# from typing import Tuple

# import pandas as pd
# from model.data_cleaning import (
#     DataCleaning,
#     DataDivideStrategy,
#     DataPreprocessStrategy,
# )
# from typing_extensions import Annotated

# # from zenml.steps import Output, step
# from zenml import step

# @step
# def clean_data(
#     data: pd.DataFrame,
# ) -> Tuple[
#     Annotated[pd.DataFrame, "x_train"],
#     Annotated[pd.DataFrame, "x_test"],
#     Annotated[pd.Series, "y_train"],
#     Annotated[pd.Series, "y_test"],
# ]:
#     """Data cleaning class which preprocesses the data and divides it into train and test data.

#     Args:
#         data: pd.DataFrame
#     """
#     try:
#         preprocess_strategy = DataPreprocessStrategy()
#         data_cleaning = DataCleaning(data, preprocess_strategy)
#         preprocessed_data = data_cleaning.handle_data()

#         divide_strategy = DataDivideStrategy()
#         data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
#         x_train, x_test, y_train, y_test = data_cleaning.handle_data()
#         return x_train, x_test, y_train, y_test
#     except Exception as e:
#         logging.error(e)
#         raise e