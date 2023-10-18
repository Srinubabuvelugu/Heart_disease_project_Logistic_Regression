import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainear import ModelTrainer







if __name__ == "__main__":
    logging.info("data ingestion has started")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    obj = DataTransformation()
    train_arr, test_arr, _ = obj.initiate_data_transformation(train_data, test_data)
    model_trainer =ModelTrainer()
    model_trainer.initiate_model_trainng(train_arr,test_arr)