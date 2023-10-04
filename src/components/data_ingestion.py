import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## initilize the Dataingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')


## create class for data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df = pd.read_csv("notebook/data/heart_disease.csv")
            logging.info('Data Set read as Pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Divide the Data set into Train Data and Test Data')
            train_set, test_set = train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info('Data Ingestion is completed')

            

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Exception Occured in Data Ingestion Stage(i.e DataIngestion)')
            raise CustomException(e,sys)
if __name__ == "__main__":
    logging.info("data ingestion has started")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()