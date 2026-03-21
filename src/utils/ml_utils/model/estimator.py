from src.constants.training_pipeline import SAVED_MODEL_DIR , MODEL_FILE_NAME

import os,sys

from src.logging.logger import logging
from src.exception.exception import CustomException
# Reusable Model Class used to predict x inputs with different models and preprocessors
class Model:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self,x):
        try:
            x_transformed = self.preprocessor.transform(x)
            y_preds = self.model.predict(x_transformed)
            
            logging.info('Model Prediction Successful')
            return y_preds
        except Exception as e:
            raise CustomException(e,sys)