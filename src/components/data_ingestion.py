from dataclasses import dataclass
import sys
import os
import logging
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_cleaning import Clean_target
from src.components.data_cleaning import Clean_features
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestion:
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self):
        raw_data_path: str=os.path.join('artifacts', 'data.csv')
        logging.info('Import the data from source')
        
        try:
            df = pd.read_csv('notebook/car.csv')
            logging.info('Reading the dataset completed')

            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

            df.to_csv(raw_data_path, index=False, header=True)

            return raw_data_path
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__== "__main__":
    obj = DataIngestion()
    raw_data = obj.initiate_data_ingestion()

    cln_target = Clean_target('price', raw_data)
    cln_target_data = cln_target.clean_target()
    
    cln_feature = Clean_features(cln_target_data)
    cln_feature_data, num_features, cat_features = cln_feature.clean_data()
    cln_feature_data.to_csv('artifacts/clean_data.csv', index=False)

    data_transformation = DataTransformation(num_features, cat_features, cln_feature_data)
    preprocessor_obj, input_features = data_transformation.initiate_data_transformation()
    print(input_features.shape)
    