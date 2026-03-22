import sys,os
from src.logging.logger import logging
from src.exception.exception import CustomException

import pandas as pd
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

from src.constants.training_pipeline import TARGET_COLUMN
from src.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig

from src.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    # Initializing Datatransformation object with DataValidationArtifact and DataTransformationConfig
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    # Helper function that reads file and returns dataframes of test and train
    def read_data(filepath:str)->pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def get_data_transformation_pipeline(cls)->Pipeline:
        # KNN Imputer usage -> finds the k nearest neighbors for a datapoint (f1,missing val) with other datapoints in fspace (f1,f2) and finds average of these k neighbors to find the missing value of datapoint
        logging.info('Entered pipeline builder method!')
        
        try:
            # Initializing KNN with double star ** to consider the params as a key value pair
            imputer:KNNImputer =KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            # Creating Pipeline for transformer object
            processor:Pipeline = Pipeline([
                ('imputer',imputer)
            ])
            return processor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info('Initiated Data Transformation Process!')
            # Reading validated test and train file path
            logging.info('Reading train and test Data')
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            print("VALIDATION STATUS:", self.data_validation_artifact.validation_status)
            print("TRAIN PATH:", self.data_validation_artifact.valid_train_file_path)
            print("TEST PATH:", self.data_validation_artifact.valid_test_file_path)
            
            
            logging.info('Seperating input and output features for train and test data!')
            # Seperating Target Feature from train dataframe 
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            # Output feature contains just -1 and 1 so converting all -1 to 0 for readability
            target_feature_train_df = target_feature_train_df.replace(-1,0)
            
            # Seperating Target Feature from train dataframe 
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            # Output feature contains just -1 and 1 so converting all -1 to 0 for readability
            target_feature_test_df = target_feature_test_df.replace(-1,0)
            
            logging.info('Building Transformation Pipeline object')
            # retrieving pipeline from the pipeline builder method
            pipeline = self.get_data_transformation_pipeline()
            
            logging.info('Fitting Transformer object with training data and applying on test data')
            # Fitting training Data in pipeline to get preprocessor object
            preprocessor_object = pipeline.fit(input_feature_train_df)
            # transforming input features train and test data
            input_feature_train_df = preprocessor_object.transform(input_feature_train_df)
            input_feature_test_df = preprocessor_object.transform(input_feature_test_df)
            
            logging.info('Converting and storing numpy array in file location')
            # Converting Dataframe into Numpy array by combining input and output features for both train and test data
            train_array = np.c_[input_feature_train_df,np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_df,np.array(target_feature_test_df)]

            logging.info('Saving transformed Data')
            # Saving Train and Test array inside filepath
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_array)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_array)
            logging.info('Saving preprocessor object')
            # Saving Preprocessor object inside the filepath
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object) 
            # Saving final preprocessor for training pipeline
            os.makedirs('final_model',exist_ok=True)
            save_object('final_model/preprocessor.pkl',obj=preprocessor_object)
            
            
            logging.info('Data Validation successful')
            # Preparing Data Transformation Artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)