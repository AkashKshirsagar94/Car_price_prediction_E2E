from dataclasses import dataclass
import sys
import os
import logging
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

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