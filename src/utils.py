import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import pickle

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging


def save_object(file_path, obj):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try: 
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for key, value in models.items():

            model_name = key

            model = value

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred)

            report[model_name] = score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
