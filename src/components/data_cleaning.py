from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



class Clean_target:
    def __init__(self, target, data):
        self.target = target
        self.data = data
    
#Create a function that will remove the rows where target data is missing
    def clean_target(self):
        df = pd.read_csv(self.data)
        cln_target_data = df.dropna(subset=self.target, axis=0).reset_index(drop=True)
        features = ['make', 'curb-weight', 'engine-size', 'horsepower', 'city-mpg', 'length']
        cln_target_data = pd.concat([cln_target_data[features], cln_target_data[['price']]], axis=1)
        return cln_target_data

class Clean_features:
    def __init__(self, data):
        self.data = data

#Create a function that will work on the missing data & use relevant strategy to replace null values
    def clean_data(self):
        #Strategy for numerical features
        features = ['make', 'curb-weight', 'engine-size', 'horsepower', 'city-mpg', 'length']
        num_features = [feature for feature in features if self.data[feature].dtype != 'O']
        cat_features = [feature for feature in features if self.data[feature].dtype == 'O']

        num_data = self.data[num_features]
        num_imputer = SimpleImputer(missing_values = np.NaN, strategy ='mean')
        num_imputer = num_imputer.fit(num_data)
        # As the tranform keyword create a data in array w/o columns, changing it to dataframe & adding columns
        num_data = pd.DataFrame(num_imputer.transform(num_data))
        num_data.columns = num_features

        #Strategy for cat features
        cat_data = self.data[cat_features]
        cat_imputer = SimpleImputer(missing_values = np.NaN, strategy ='most_frequent')
        cat_imputer = cat_imputer.fit(cat_data)
        # As the tranform keyword create a data in array w/o columns, changing it to dataframe & adding columns
        cat_data = pd.DataFrame(cat_imputer.transform(cat_data))
        cat_data.columns = cat_features

        #Concatenation
        cleaned_data = pd.concat([num_data, cat_data, self.data[['price']]], axis=1)

        return cleaned_data, num_features, cat_features
