import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.impute import SimpleImputer

class DataTransformation:
    def __init__(self, num_features, cat_features, data):
        self.num_features = num_features
        self.cat_features = cat_features
        self.data = data

#This creates transformation object that performs data normalization & onehotencoding    
    def data_transformation_object(self):
        num_pipeline = Pipeline(steps = [
                                    ('scaler', StandardScaler(with_mean=False))
                                    ]
                               )
        cat_pipeline = Pipeline(steps = [
                                    ('Onehotencoder', OneHotEncoder()),
                                    ('scaler', StandardScaler(with_mean=False))
                                    ]
                               )
        preprocessor = ColumnTransformer(
                                    [('num_pipeline', num_pipeline, self.num_features),
                                     ('cat_pipeline', cat_pipeline, self.cat_features)
                                    ]
                                        )
        return preprocessor

    def initiate_data_transformation(self):
        # Saving a preprocessor object file
        preprocessor_obj = self.data_transformation_object()
        preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
        save_object(file_path=preprocessor_obj_file_path, obj=preprocessor_obj)

        #Initiate the data tranformation
        x = self.data.drop(columns = ['price'], axis=1)
        input_features = preprocessor_obj.fit_transform(x).toarray()

        return preprocessor_obj_file_path, input_features

    
