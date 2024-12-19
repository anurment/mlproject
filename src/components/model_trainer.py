import os
import sys
from dataclasses import dataclass

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, train_and_evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join(".artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Random Forest": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 10, 20],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "XGBRegressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
            }

            report, best_model_name, best_model_score, best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test, models, params)

            report_df = pd.DataFrame(report).T

            print("Model Performance Report:\n")
            print(report_df.round(2))
            #report_df.to_csv("model_performance_report.csv", index=True)
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score
            



            
        except Exception as e:
            raise CustomException(e,sys)
