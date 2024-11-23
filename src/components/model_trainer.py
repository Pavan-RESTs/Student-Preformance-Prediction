import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join('.artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], 
                train_arr[:,-1],
                test_arr[:,:-1], 
                test_arr[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Support Vector Machine': SVR(),
                'Gradient-Boost': GradientBoostingRegressor(),
                'Ada-Boost': AdaBoostRegressor(),
                'XG-Boost': XGBRegressor(),
                'Cat-Boost': CatBoostRegressor(verbose=False),
                'KNN': KNeighborsRegressor()
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, 
                                               models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)
            
            logging.info("Best model is found")

            save_object(self.model_trainer_config.trained_model_file_path,best_model)

            logging.info(f"Best model saved at location {self.model_trainer_config.trained_model_file_path}")

            y_pred = best_model.predict(X_test)

            score = r2_score(y_test, y_pred)

            return score

        except Exception as e:
            raise CustomException(e,sys)
