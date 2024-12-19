import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('.artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_preprocessor_from_df(self, df):
        try:
            num_features = df.select_dtypes(exclude="object").columns
            cat_features = df.select_dtypes(include="object").columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def transform_data(self, train_path, test_path, target_column):

        try:

            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "math_score"

            logging.info("Preprocessing data")
            X_train_df = train_df.drop(columns=[target_column], axis=1)
            y_train_df = train_df[target_column]

            X_test_df = test_df.drop(columns=[target_column], axis=1)
            y_test_df = test_df[target_column]

            logging.info("Get preprocessor")
            preprocessor = self.get_preprocessor_from_df(X_train_df)

            X_train_arr = preprocessor.fit_transform(X_train_df)
            X_test_arr = preprocessor.transform(X_test_df)

            train_arr = np.c_[ X_train_arr, np.array(y_train_df) ]
            test_arr = np.c_[X_test_arr, np.array(y_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
