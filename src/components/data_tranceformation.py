import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utlit import save_obj

@dataclass
class datatransformationconfig:
    preprocessing_obj_file_path = os.path.join('artifact', 'preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation = datatransformationconfig()
    
    def get_data_transformation_obj(self):
        try:
            num_feature = ['reading_score', 'writing_score']
            cat_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            logging.info('Categorical columns encoding is complete')
            logging.info('Numerical columns standard scaling is complete')
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipline, num_feature),
                ('cat_pipeline', cat_pipline, cat_feature)
            ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transfrom(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data successfully')
            
            preprocessor = self.get_data_transformation_obj()
            target_col = 'math_score'
            
            input_feature_train = train_df.drop(columns=[target_col], axis=1)
            target_feature_train = train_df[target_col]
            
            input_feature_test = test_df.drop(columns=[target_col], axis=1)
            target_feature_test = test_df[target_col]
            
            logging.info('Starting preprocessing')
            
            input_feature_train = preprocessor.fit_transform(input_feature_train)
            input_feature_test = preprocessor.transform(input_feature_test)
            
            train_arr = np.c_[input_feature_train, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test, np.array(target_feature_test)]
            
            save_obj(
                file_path=self.data_transformation.preprocessing_obj_file_path,
                obj=preprocessor
            )
            
            return (train_arr, test_arr, self.data_transformation.preprocessing_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)
