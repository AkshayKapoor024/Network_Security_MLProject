from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig , Training_Pipeline_Config
import sys
from src.logging.logger import logging
from src.exception.exception import CustomException
import os 

# Performing Data Ingestion from scratch through main file 
if __name__=='__main__':
    try:
        training_pipeline_config = Training_Pipeline_Config()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info('Initiate Data Ingestion!')
        
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise CustomException(e,sys)