import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_obj_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):

        try:
            cat_columns = ['cut','color','clarity']
            num_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_maper = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_maper = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_maper = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder',OrdinalEncoder(categories=[cut_maper,color_maper,clarity_maper])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipeline',cat_pipeline,cat_columns)
            ])

            logging.info('Preprocessor object created')

            return preprocessor


        except Exception as e:
            logging.info('Exception occured in data transformation phase')
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_obj()

            target_column = 'price'
            drop_columns = ['id',target_column]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on train and test data')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_obj_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_obj_path
            )



        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)
