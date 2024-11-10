from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application
#Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            float(request.form.get('curb-weight')),
            float(request.form.get('engine-size')),
            float(request.form.get('horsepower')),
            float(request.form.get('city-mpg')),
            float(request.form.get('length')),
            request.form.get('make')
        )

        user_feature_data = data.convert_data_into_df()
        print(user_feature_data)
        print("Before Prediction")

        predict_pipeline = PredictPipeline(user_feature_data)
        print("Mid Prediction")
        results = predict_pipeline.predict_pipeline()
        print("after Prediction")
        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host='0.0.0.0')