import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
# Import machine learning algorithms from scikit-learn (Sklearn)
from sklearn.linear_model import LinearRegression  # Import Linear Regression for regression tasks
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestClassifier for regression tasks
from sklearn.svm import SVR  # Import Support Vector Regression for regression tasks
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor for regression tasks

print("hello word")


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        lr = LinearRegression(**kwargs)
        lr.fit(X_train, y_train)
        return lr

class SVRModel(Model):
    """
    SVR that implements the Model interface.
    """
    

    def train(self, X_train, y_train, **kwargs):
        svm = SVR(**kwargs)
        svm.fit(X_train, y_train)
        return svm

class RandomForestModel(Model):
    """
    Random Forest that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        rf = RandomForestRegressor( **kwargs)
        rf.fit(X_train, y_train)    
        return rf

class GradientBoostingModel(Model):
    """
    Gradient Boosting Regressor that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        gr = GradientBoostingRegressor(**kwargs)
        gr.fit(X_train, y_train)  
        return gr