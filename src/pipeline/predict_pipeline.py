import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy

from src.utils import load_object


class PredictPipeline:
    def __init__(self, user_feature_data):
        self.user_feature_data = user_feature_data
    
    def predict_pipeline(self):
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        preprocessor = load_object(preprocessor_path)

        model_path = os.path.join('artifacts', 'model.pkl')
        model = load_object(model_path)

        scaled_data = preprocessor.transform(self.user_feature_data)
        yhat = model.predict(scaled_data)

        return yhat
    
class CustomData:
    def __init__(self, curb_weight: float, enginesize: float,
                 horsepower: float, citympg: float, length: float, make: str):
        self.curb_weight = curb_weight
        self.enginesize = enginesize
        self.horsepower = horsepower
        self.citympg = citympg
        self.length = length
        self.make = make

    def convert_data_into_df(self):
        df = {'curb-weight': [self.curb_weight], 'engine-size': [self.enginesize], 
              'horsepower': [self.horsepower], 'city-mpg': [self.citympg], 
              'length': [self.length], 'make': [self.make]}
        
        return pd.DataFrame(df)