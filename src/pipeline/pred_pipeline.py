import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from src.utlit import save_obj,lode_obj
 


class PredictPipline:
    def __init__(self,):
        pass
    def predict(self,feature):
        model_path = 'artifact/model.pkl'
        preprossure_path = 'artifact/preprocesser.pkl'
        model = lode_obj(file_path=model_path)
        preprossure = lode_obj(file_path=preprossure_path)
        data_scaled = preprossure.transform(feature)
        preds = model.predict(data_scaled)
        return preds
        
    
class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender = gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)