
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  ## Handling Missing Values
from sklearn.preprocessing import StandardScaler   ## Handling Feature Scalling 
from sklearn.compose import ColumnTransformer

## pipe lines
from sklearn.pipeline import Pipeline

## logging and Exception handling
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    train_path = os.path.join('artifacts','train.csv')
    test_path = os.path.join('artifacts','test.csv')

## Create a class for Data Transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

## Create object for data Transformation
    def get_data_transformation_object(self):
        logging.info('Data Transformation initiated')

        try:
            ## Numerical Columns
            num_columns = ['male', 'age', 'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI',
                'heartRate', 'glucose']
            
            ## Pipeline
            ## Numerical Pipe line
            logging.info('Numerical Pipeline started')
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns)
            ])
            logging.info('Pipe line had been completed')

            return preprocessor

        except Exception as e:
            logging.error('Error occured in Data Transformation')
            raise CustomException(e,sys)

## Initiate the Data Transformation 
    def initiate_data_transformation(self,train_path,test_path):
        try:
            ## REading the Train and Test Data sets
            logging.info('Reading the Train and Test data sets')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and Test data sets are reading completed')
            logging.info(f'Train DataFrame Head :\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head :\n{test_df.head().to_string()}')
            logging.info('Obtaing the Preprocessor object')

            preprocessing_obj = self.get_data_transformation_object()
            ## drop the unused columns and dependent feature
            target_column = 'TenYearCHD'
            drop_columns = [target_column]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info('Applying prepeocessing object on training and testing datasets.')
            ## Transforming using prepeocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            ## creating numpy arrays to concatenate
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            ## calling the save_object in utils folder
            save_object(    
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("preprocessor Pickel file saving completed.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.error('Exception occured in the initiate_data_transformation')
            raise CustomException(e,sys)


