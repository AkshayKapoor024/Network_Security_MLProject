import os,sys

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import Training_Pipeline_Config,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig

from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact


class TrainingPipeline:
    # Initializing Training Pipeline config 
    def __init__(self):
        try:
            self.training_pipeline_config = Training_Pipeline_Config()
            
        except Exception as e:
            raise CustomException(e,sys)
    # Data Ingestion Function
    def start_data_ingestion(self):
        try:
            logging.info('Initiate Data Ingestion!')
            self.data_ingestion_config:DataIngestionConfig = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            
            data_ingestion:DataIngestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            
            data_ingestion_artifact:DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
            logging.info('Completed Data Ingestion!')
            return data_ingestion_artifact
            
        except Exception as e:
            raise CustomException(e,sys)
    
    # Data Validation Function
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info('Initiate Data Validation')
            self.data_validation_config:DataValidationConfig = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            
            data_validation:DataValidation = DataValidation(data_validation_config=self.data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            
            data_validation_artifact:DataValidationArtifact = data_validation.initiate_data_validation()
            logging.info('Completed Data Ingestion!')
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e,sys)
            
    # Data Transformation Process
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            logging.info('Initiate Data Transformation')
            self.dataTransformationConfig :DataTransformationConfig = DataTransformationConfig(self.training_pipeline_config)
            data_transformation :DataTransformation=DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=self.dataTransformationConfig)
            data_transformation_artifact :DataTransformationArtifact=data_transformation.initiate_data_transformation()
            logging.info('Completed Data Transformation')
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    # Model Training Process
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info('Initiate Model Trainer')
            self.modelTrainerConfig :ModelTrainerConfig = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer :ModelTrainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=self.modelTrainerConfig)
            model_trainer_artifact :ModelTrainerArtifact=model_trainer.initiate_model_trainer()
            logging.info('Completed Model Trainer')
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    # Running Full Pipeline
    def run_pipeline(self):
        try:
            logging.info('Started Training Pipeline')
            # Starting Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            # Starting Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # Starting Data Transformation
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            # Starting Data Ingestion
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            logging.info('Completed Training Pipeline')
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
        