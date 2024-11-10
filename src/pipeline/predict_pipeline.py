import os
import sys
import numpy as np
import pandas as pd

from src.utils import load_object


class PredictPipeline:
    def __init__(self, user_features):
        self.user_features = user_features
    
    def predict_pipeline(self):
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        preprocessor = load_object(preprocessor_path)

        model_path = os.path.join('artifacts', 'model.pkl')
        model = load_object(model_path)

        scaled_data = preprocessor.transform(self.user_features).toarray()
        yhat = model.predict(scaled_data)

        return yhat
    
class CustomData:
    def __init__(self, make: str, curb_weight: float, enginesize: float,
                 horsepower: float, citympg: float, length: float):
        self.make = make
        self.curb_weight = curb_weight
        self.enginesize = enginesize
        self.horsepower = horsepower
        self.citympg = citympg
        self.length = length
    
    def convert_data_into_df(self):
        df = {'make': self.make, 'curb-weight': self.curb_weight, 'engine-size': self.enginesize,
              'horsepower': self.horsepower, 'city-mpg': self.citympg, 'length': self.length}
        
        return pd.DataFrame(df)