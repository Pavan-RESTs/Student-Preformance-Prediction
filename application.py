import pickle
import sys
import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = {
            'gender': request.form.get('gender'),
            'race_ethnicity': request.form.get('ethnicity'),
            'parental_level_of_education': request.form.get('parental_level_of_education'),
            'lunch': request.form.get('lunch'),
            'test_preparation_course': request.form.get('test_preparation_course'),
            'reading_score': float(request.form.get('reading_score')),
            'writing_score': float(request.form.get('writing_score'))
        }

        pred_data_obj = CustomData(list(data.values()))

        pred_df = pred_data_obj.get_data_as_frame()

        pred_obj = PredictPipeline()

        prediction = pred_obj.predict(pred_df)

        return render_template('home.html',results=prediction)
        

if __name__ == "__main__":
    app.run(host='0.0.0.0')

