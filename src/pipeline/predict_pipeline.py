import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from utils import load_object

class PredictPipeline():
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = '.artifacts/model.pkl'
            preprocessor_path = '.artifacts/preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            preprocessed_data = preprocessor.transform(features)
            prediction = model.predict(preprocessed_data)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)





class CustomData():
    def __init__(self, 
                df_list):
        
        self.gender = df_list[0]

        self.raise_ethnicity = df_list[1]

        self.parental_level_of_education = df_list[2]

        self.lunch = df_list[3]

        self.test_preparation_courses = df_list[4]

        self.reading_score = df_list[5]

        self.writing_score = df_list[6]


    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                    'gender': [self.gender],
                    'race_ethnicity': [self.raise_ethnicity],  # Note: fixed typo from raise to race
                    'parental_level_of_education': [self.parental_level_of_education],
                    'lunch': [self.lunch],
                    'test_preparation_course': [self.test_preparation_courses],  # Note: made singular to match form
                    'reading_score': [self.reading_score],
                    'writing_score': [self.writing_score]
                    }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)