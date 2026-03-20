import sys
from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants.training_pipeline import SCHEMA_FILE_PATH

from scipy.stats import ks_2samp
import os
import pandas as pd
import numpy as np

from src.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
    
    # Helper method used to read data from a path and return dataframe 
    @staticmethod
    def read_data(filepath:str)->pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CustomException(e,sys)
    
    # Helper function used to validate number of columns in schema
    def validate_number_of_columns(self,dataframe:pd.DataFrame):
        try:
            # Retrieving number of columns from schema
            number_of_columns = len(self._schema_config)
            logging.info(f'Required Number of Columns -> {number_of_columns}')
            logging.info(f'Number of columns in dataframe -> {len(dataframe.columns)}')
            # Returning true if no. of columns match otherwise return false
            if len(dataframe.columns)==number_of_columns:
                return True
            else:
                return False
            
        except Exception as e:
            raise CustomException(e,sys)    
    # Helper function used to validate the numerical columns in the dataframe to match with schema 
    def validate_numerical_columns(self,dataframe:pd.DataFrame):
        try:
            # Retrieving Numerical Columns from schema
            schema_num_columns = self._schema_config['numerical_columns']
            
            logging.info('Matching Numerical columns present in dataframe with schema!')
            # Retrieving columns from dataframe 
            dataframe_columns = dataframe.columns
            # Finding Number of columns not present in dataframe
            num_cols_not_present = [column for column in schema_num_columns if column not in dataframe_columns]
            
            logging.info(f'Numerical columns in schema :{len(schema_num_columns)} ,  numerical columns found in dataframe :{len(schema_num_columns)-len(num_cols_not_present)}')
            # If numerical columns not found in dataframe return false else return true
            if len(num_cols_not_present)>0:
                return False
            else:
                return True
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def detect_data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,threshold:float=0.30)->bool:
        try:
            # Initiating status value as false
            status=False

            # Initiating drift report as empty
            report = {}
            # Checking and comparing distribution for every column
            logging.info('Initiating data drift validation')
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1,d2)

                if threshold<=is_same_dist.pvalue:
                    is_found= False
                else:
                    is_found=True
                    status=True
                report.update({
                    column: {
                        'pvalue': float(is_same_dist.pvalue),
                        'drift_status': is_found
                    }
                })
                
            # Retrieving drift report file path
            logging.info('Preparing data drift validation report')
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            # Creating directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
                
            logging.info('Storing data drift validation report')
            write_yaml_file(file_path=drift_report_file_path,content=report)
                
            logging.info('data drift validation completed!')
            return status
        except Exception as e:
            raise CustomException(e,sys)
    
    # Main Data validation method
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info('Data Validation Started')
            # Retrieving train and test file path           
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path 
            
            logging.info('Retrieved train and test data from Artifacts')
            # Read Data from train and test file path 
            train_data =DataValidation.read_data(train_file_path)
            test_data = DataValidation.read_data(test_file_path)
            
            error_message1=''
            # Validating Number of columns
            col_status = self.validate_number_of_columns(dataframe=train_data)
            if not col_status:
                error_message1 = f'Train Data does not contain all columns required for model training!\n'
            col_status = self.validate_number_of_columns(dataframe=test_data)
            if not col_status:
                error_message1 = f'Test Data does not contain all columns required for model training!\n'
            
            logging.info('Started numerical columns Validation process')
            # Validating Numerical columns
            error_message2=''
            num_status = self.validate_numerical_columns(dataframe=train_data)
            if not num_status:
                error_message2 = f'Train Data does not contain all numerical columns required for model training!\n'
            num_status = self.validate_numerical_columns(dataframe=test_data)
            if not num_status:
                error_message2 = f'Test Data does not contain all numerical columns required for model training!\n'
                
            logging.info('Started data drift Validation process')
            # Checking Data drift
            status = self.detect_data_drift(base_df=train_data,current_df=test_data)
            dir_path = os.path.dirname(self.data_validation_config.valid_data_dir)
            os.makedirs(dir_path,exist_ok=True)
            
            
            logging.info('Processing data based on report')
            
            os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_file_path),exist_ok=True)
            if status==False:
            
                # Storing Valid Data in valid folder 
                train_data.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
                test_data.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            
                data_validation_artifact = DataValidationArtifact(
                    validation_status=status,
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )
                if error_message1:
                    logging.info(error_message1)
                if error_message2:
                    logging.info(error_message2)
                    
                return data_validation_artifact
            else:
                # Storing Valid Data in valid folder 
                train_data.to_csv(self.data_validation_config.invalid_train_file_path,index=False,header=True)
                test_data.to_csv(self.data_validation_config.invalid_test_file_path,index=False,header=True)
            
                data_validation_artifact = DataValidationArtifact(
                    validation_status=status,
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )
                
                if error_message1:
                    logging.info(error_message1)
                if error_message2:
                    logging.info(error_message2)
                return data_validation_artifact
                           
        except Exception as e:
            raise CustomException(e,sys)
        



