import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass

from src.utils import evaluate_models
from src.utils import save_object

class ModelTraining:
    def __init__(self, input_feature_arr, target_arr):
        self.input_feature_arr = input_feature_arr
        self.target_arr = target_arr

    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.input_feature_arr, 
                                                            self.target_arr, 
                                                            test_size=0.2, random_state=1)
        models = {  'Decision Tree': DecisionTreeRegressor(),
                    'Linear Regresssion': LinearRegression(),
                    'K-Neighbors Regressor': KNeighborsRegressor(),
                 }
        
        model_report: dict=evaluate_models(x_train=x_train, y_train=y_train, 
                                              x_test=x_test, y_test=y_test, models=models)
        
        #To get the best model score from dictionary
        best_model_score = max(sorted(model_report.values()))

        #To get the best model name fom dictionary
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)]
        
        best_model = models[best_model_name]

        if best_model_score<0.6:
            return print("No best model found")
        
        trained_model_file_path = os.path.join('artifacts', 'model.pkl')
        save_object(file_path= trained_model_file_path, obj=best_model)

        return best_model_score, best_model_name