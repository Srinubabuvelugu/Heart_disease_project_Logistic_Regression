import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            logging.info('This is the prediction Pipeline')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path =os.path.join('artifacts','model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            logging.error('Error occured in prediction pipeline')
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
                male,
                age,
                education,
                currentSmoker,
                cigsPerDay,
                BPMeds ,
                prevalentStroke,
                prevalentHyp,
                diabetes,
                totChol,
                sysBP,
                diaBP,
                BMI,
                heartRate,
                glucose
            ):
        self.male = male
        self.age = age
        self.education = education
        self.currentSmoker = currentSmoker
        self.cigsPerDay = cigsPerDay
        self.BPMeds =BPMeds
        self.prevalentStroke = prevalentStroke
        self.prevalentHyp = prevalentHyp
        self.diabetes = diabetes
        self.totChol = totChol
        self.sysBP = sysBP
        self.diaBP = diaBP
        self.BMI = BMI
        self.heartRate = heartRate
        self.glucose =glucose

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'male':[self.male],
                'age':[self.age],
                'education':[self.education],
                'currentSmoker':[self.currentSmoker],
                'cigsPerDay':[self.cigsPerDay],
                'BPMeds':[self.BPMeds],
                'prevalentStroke':[self.prevalentStroke],
                'prevalentHyp':[self.prevalentHyp],
                'diabetes':[self.diabetes],
                'totChol':[self.totChol],
                'sysBP':[self.sysBP],
                'diaBP':[self.diaBP],
                'BMI':[self.BMI],
                'heartRate':[self.heartRate],
                'glucose':[self.glucose]
        
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Data Frame Gathered')
            return df
        except Exception as e:
            logging.error('Error occured in prediction Pipeline custom data imput')
            raise CustomException(e,sys)
