import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
## logging and Exception handling
from src.logger import logging
from src.exception import CustomException
## to create a pickle files
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.error('Exception occured in creating pickle file (i.e utils/save_object)')
        raise CustomException(e,sys)

def evalute_model(X_train, y_train, X_test,y_test,models):
    try:
        logging.info('Model Evalalution in model_trainer/evalute_model')
        report = {}
        for i in range(len(models.values())):
            model = list(models.values())[i]
            ### Train the Model
            model.fit(X_train, y_train)
            ## Model  Prediction on the train data
            y_test_pred = model.predict(X_test)
            ## Get the accuracy of the Model 
            test_accuracy = accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_accuracy
        return report
    except Exception as e:
        logging.error('Exception occured Model Training in Model_trainer/evaluate_model')
        raise CustomException(e,sys)
