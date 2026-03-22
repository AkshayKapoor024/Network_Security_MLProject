from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig , Training_Pipeline_Config , DataValidationConfig , DataTransformationConfig,ModelTrainerConfig
import sys
from src.logging.logger import logging
from src.exception.exception import CustomException
import os 
from dotenv import load_dotenv
load_dotenv()

import dagshub
dagshub.init(repo_name=os.getenv('DAGSHUB_REPO_NAME'),repo_owner=os.getenv('DAGSHUB_REPO_OWNER'),mlflow=True)
# Performing Data Ingestion from scratch through main file 
if __name__=='__main__':
    try:
        # Data Ingestion Process
        training_pipeline_config = Training_Pipeline_Config()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info('Initiate Data Ingestion!')
        
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info('Completed Data Ingestion!')
        print(data_ingestion_artifact)
        
        # Data Validation Process
        logging.info('Initiate Data Validation')
        dataValidationConfig = DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=dataValidationConfig)
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info('Completed Data Validation')
        print(data_validation_artifact)
        
        # Data Transformation Process
        logging.info('Initiate Data Transformation')
        dataTransformationConfig = DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=dataTransformationConfig)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info('Completed Data Transformation')
        print(data_transformation_artifact)
        
        
        # Data Transformation Process
        logging.info('Initiate Model Training')
        modelTrainerConfig = ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=modelTrainerConfig)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        print(model_trainer_artifact)
        
    except Exception as e:
        raise CustomException(e,sys)

