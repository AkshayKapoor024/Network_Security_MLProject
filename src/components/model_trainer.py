import os,sys

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact

from src.utils.ml_utils.model.estimator import Model
from src.utils.main_utils.utils import save_object,load_object
from src.utils.main_utils.utils import load_numpy_array_data,evaluate_model
from src.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBRFClassifier


# Main Model training Class 
class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    # Helper function that trains multiple models 
    def train_model(self,x_train,y_train,x_test,y_test):
        # Defining Models to train
        models = {
            'Logistic Regression':LogisticRegression(verbose=1),
            'Support Vector Classifier':SVC(verbose=1),
            'Decision Tree':DecisionTreeClassifier(),
            'Random Forest':RandomForestClassifier(verbose=1),
            'AdaBoost':AdaBoostClassifier(),
            'Gradient Boosting':GradientBoostingClassifier(verbose=1),
            'XG Boost':XGBRFClassifier()
        }
    # Parameters for grid search CV Cross validation
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                'criterion':['gini', 'entropy', 'log_loss'],
                
                'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Support Vector Classifier":{},
            'XG Boost':{
                "loss": ["log_loss", "exponential"],   # classification losses
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 5, 8 , 10, 20 , 15 ],
                "colsample_bytree":[0.5,0.8,1,0.3,0.4]
            }
            
            }
        logging.info('Started building model report')
        model_report: dict= evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
        logging.info('Completed building model report')


        # Retrieved Best Model score
        best_model_score = max(sorted(model_report.values()))
        
        # Retrieved Best Model name
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        # Retrieved Best Model
        best_model = models[best_model_name]
        logging.info(f'Found {best_model_name} as the best model with r2_score :{best_model_score} for the given data')
        
        logging.info('Started Best Model training')
        # Now Predicting Y train from actual best model
        y_train_preds = best_model.predict(x_train)
        # Y train Classification Scores
        logging.info('Building CLassification report for best model')
        classification_metrics_train = get_classification_score(y_train,y_train_preds)
        
        # Track the ML Flow
        
        y_test_preds = best_model.predict(x_test)
        classification_metrics_test = get_classification_score(y_test,y_test_preds)
        
        logging.info('Retrieved Preprocessor object file')
        # Retrieving Preprocessor .pkl file
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        model_file_path = self.model_trainer_config.trained_model_file_path
        os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
        
        # Saving Final Model CLass containing both preprocessor and best_model so that anynew data can be preprocessed and predict data based on that
        logging.info('Building final model containing both preprocessor and best model')
        final_model = Model(preprocessor=preprocessor , model= best_model)
        
        logging.info('Saving final model')
        save_object(model_file_path,obj=final_model)
        
        # Creating Model trainer artifact
        logging.info('Created and Returned Model Trainer Artifact')
        logging.info('Model Training Completed Successfuly')
        return ModelTrainerArtifact(trained_model_file_path=model_file_path,train_metric_artifact=classification_metrics_train,test_metric_artifact=classification_metrics_test)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info('Started Model Training Process')
            # Retrieving train test data 
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            
            # Retrieving train test data 
            logging.info('Retrieved train and test data')
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train , y_train , x_test , y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            logging.info('Train test splitting data into input and output features')
            # Returning Output of model trainer
            return self.train_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
            )
            
        except Exception as e:
            raise CustomException(e,sys)
