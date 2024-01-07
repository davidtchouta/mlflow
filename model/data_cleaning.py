import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            # Initialize the SimpleImputer with the strategy 'mean'
            imputer = SimpleImputer(strategy='mean')

            # Specify the columns with missing values that we want to impute
            columns_with_missing_values = ['age', 'bmi']

            # Apply the imputer to fill missing values with the mean
            data[columns_with_missing_values] = imputer.fit_transform(data[columns_with_missing_values])

            #data.isnull().sum()
            data['sex']=data['sex'].map({'female':0,'male':1})
            # Create a LabelEncoder instance
            label_encoder = LabelEncoder()
            # Encode the "city" column
            data['city'] = label_encoder.fit_transform(data['city'])
            data['hereditary_diseases'] = label_encoder.fit_transform(data['hereditary_diseases'])
            data['job_title'] = label_encoder.fit_transform(data['job_title'])
            #data['city'].unique()
            #data['hereditary_diseases'].unique()
            #x=data.drop(['claim'],axis=1)
            #y=data['claim']
            #y.head()
            return data
        except Exception as e:
            logging.error(e)
            raise e



class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            x=data.drop(['claim'],axis=1)
            y=data['claim']
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e



class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e