import sys
from src.logging.logger import logging
from src.exception.exception import CustomException
import os 
import pymongo
from sklearn.model_selection import train_test_split
import numpy as np
from src.entity.config_entity import DataIngestionConfig

from dotenv import load_dotenv

import pandas as pd 

from src.entity.artifact_entity import DataIngestionArtifact

load_dotenv()

uri = os.getenv('MONGODB_URL')
# Modular class performing Data Ingestion
class DataIngestion:
    # Initiating Data Ingestion Config inside main data ingestion component
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:
            raise CustomException(e,sys)
    
    # Helper function to read from mongodb and create and return a dataframe
    def export_collection_as_dataframe(self):
        try:
            # Retrieving constants from config file 
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            # Creating Mongodb Client
            logging.info('Establishing Connection with MongoDB')
            self.mongo_client = pymongo.MongoClient(uri)
            collection = self.mongo_client[database_name][collection_name]
            # Retrieving data from Mongodb
            df = pd.DataFrame(list(collection.find()))
            
            logging.info('Successfully Read data from mongodb into dataframe')
            # Removing Id column from retrieved data
            if "_id" in df.columns.to_list():
                df = df.drop(columns=['_id'],axis=1)
            # Replacing null values with numpy nan for better readability    
            df.replace({"na":np.nan},inplace=True)  
            
            logging.info('Returned final Data frame')
            return df
        except Exception as e:
            raise CustomException(e,sys)
            
    # Helper Function that exports data (raw) into feature store in artifacts library
    def export_data_into_feature_store(self,df:pd.DataFrame):
        try:
            logging.info('Initiated storing raw data in feature store directory')
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            df.to_csv(feature_store_file_path,index=False,header=True)
            logging.info('Successfully stored raw data in feature store directory')
            return df
        except Exception as e :
            raise CustomException(e,sys)
        
    def export_data_train_test(self,df:pd.DataFrame):
        try:
            
            train_set , test_set = train_test_split(df,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            
            logging.info('Performed train test split on the dataframe')
            
            logging.info('Exited export_data_as_train_test method of Data Ingestion class')
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f'Exporting train test files into artifacts/')
            
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            
            logging.info('Successfully exported train test data into artifacts/ingested/ folder')
            
        except Exception as e:
            raise CustomException(e,sys)
    # Creates Ingestion files and artifacts after performing Reading Mongodb
    def initiate_data_ingestion(self):
        try:
            # Collecting Data from mongodb as dataframe using helper
            logging.info('Collecting Data from mongodb as dataframe using helper in data ingestion')
            dataframe = self.export_collection_as_dataframe()
            # Storing Raw data in artifacts folders/
            logging.info('Storing Raw data in local repository using helper function')
            dataframe = self.export_data_into_feature_store(dataframe)
            # Storing Train Test Split data in ingested folder 
            logging.info('Performing Train Test Split in Data ingestion using helper')
            self.export_data_train_test(df=dataframe)
            
            # Storing Artifacts related config in DataIngestionArtifact (Storing the output of DataIngestion step)
            logging.info('Storing Output of Data Ingestion (Train path + Test Path in DataIngestionArtifacts object)')
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info('Data Ingestion Component Executed Successfully')
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        