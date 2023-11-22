import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


## importing logging and Exception handling
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evalute_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

## create class for model Trainer

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainng(self,train_array,test_array):
        try:
            logging.info('Splitting the Dependent and Independent features from the train and test data sets')
            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                'LogosticRegression()':LogisticRegression(),
                'Decesion Tree Classification':DecisionTreeClassifier(),
                'Support Vector Classification':SVC(),
                'Random_Forest':RandomForestClassifier()
            }
            model_report:dict = evalute_model(X_train,y_train, X_test, y_test,models)
            print('Model Report:\n',model_report)
            print('\n'+'*'*100)
            logging.info(f'Model Report:{model_report}')

            ## To get the best model score from  mpdel _report dict
            best_model_score = max(model_report.values())
            ## ?Best model Name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            ## save the pickle file in wb mode

            ## best_model
            best_model = models[best_model_name]
            print(f'Best Model Found, Model Name: {best_model_name} , Accuracy Score: {best_model_score}')
            print('\n'+'='*100 )
            logging.info(f'Best Model Found, Model Name: {best_model_name} , Accuracy Score: {best_model_score}')
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

        except Exception as e:
            logging.error('Error occured in Model Training')
            raise CustomException(e,sys)