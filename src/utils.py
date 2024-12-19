import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def train_and_evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
    
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            param_grid = params.get(model_name, {})
            
            if param_grid:
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', verbose=1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            trained_models[model_name] = best_model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = {"Train R2 Score": train_score, "Test R2 Score": test_score}

        # Find the best model
        best_model_name = max(report, key=lambda model: report[model]["Test R2 Score"])
        best_model_score = report[best_model_name]["Test R2 Score"]
        best_model_object = trained_models[best_model_name]

        print(f"Best Model: {best_model_name}")
        print(f"Best Model Test R2 Score: {best_model_score}")

        return report, best_model_name, best_model_score, best_model_object

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
