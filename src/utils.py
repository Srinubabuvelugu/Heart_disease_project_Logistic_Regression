import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
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
        
