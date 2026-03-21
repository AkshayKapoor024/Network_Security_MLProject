from src.logging.logger import logging
from src.exception.exception import CustomException
from src.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score,recall_score,precision_score
import sys,os
# Helper function to calculate classification scores 
def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        logging.info('Started Classification scoring process')
        model_f1_score = f1_score(y_true=y_true,y_pred=y_pred)
        model_recall_score = recall_score(y_true=y_true,y_pred=y_pred)
        model_precision_score = precision_score(y_true=y_true,y_pred=y_pred)
    
        logging.info('Ended Classification scoring process')
        return ClassificationMetricArtifact(
            precision_score=model_precision_score,
            f1_score=model_f1_score,
            recall_score=model_recall_score
        )
    except Exception as e:
        raise CustomException(e,sys)
    