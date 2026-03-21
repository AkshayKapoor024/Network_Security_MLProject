import os 
import sys
import numpy as np 
import pandas as pd


"""
DEFINING COMMON CONSTANT VARIABLES FOR TRAINING PIPELINE
"""
TARGET_COLUMN = 'Result'
PIPELINE_NAME :str='src',
ARTIFACT_DIR : str = 'Artifacts'
FILE_NAME : str = 'phisingData.csv'
TRAIN_FILE_NAME:str='train.csv'
TEST_FILE_NAME:str='test.csv'
SCHEMA_FILE_PATH:str=os.path.join('data_schema','schema.yaml')
"""
Here in this file all the Data Ingestion Related constants will be stored for efficient retrieval upon requirements
"""

DATA_INGESTION_COLLECTION_NAME:str='Network_Phising_Project'
DATA_INGESTION_DATABASE_NAME:str='Ml_projects'
DATA_INGESTION_DIR_NAME:str='data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR:str='feature_store'
DATA_INGESTION_INGESTED_DIR:str='ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2

"""
Data Validation related constants will be defined here for efficient retrieval and usage in DataValidationConfig
"""

DATA_VALIDATION_DIR_NAME:str='data_validation'
DATA_VALIDATION_VALID_DIR_NAME:str='validated'
DATA_VALIDATION_INVALID_DIR_NAME:str='invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR:str='drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str='report.yaml'

"""
Data Transformation related constants will be defined here for efficient retrieval and usage in DataValidationConfig
"""
DATA_TRANSFORMATION_DIR_NAME :str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR :str = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str='transformed_object'
PREPROCESSING_OBJECT_FILE_NAME='preprocessor.pkl'
DATA_TRANSFORMATION_IMPUTER_PARAMS:dict={
    'missing_values':np.nan,
    'n_neighbors':3,
    'weights':'uniform'
}