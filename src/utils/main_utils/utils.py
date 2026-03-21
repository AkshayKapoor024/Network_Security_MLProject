import yaml
from src.exception.exception import CustomException
from src.logging.logger import logging

import os,sys
import numpy as np
import dill
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Helper function used to read a yaml file
def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'rb') as yaml_file:
            logging.info('Successfully read yaml schema file !')
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logging.info('Error while reading yaml schema file !')
        raise CustomException(e,sys)

# Helper function used to write a yaml file
def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            logging.info('Successfully wrote yaml schema file !')
            yaml.dump(content,file)
            
    except Exception as e:
        logging.info('Error while writing yaml schema file !')
        raise CustomException(e,sys)
# Helper function used to save numpy array into filepath    
def save_numpy_array_data(filepath:str,array:np.array)->None:
    try:
        dir_path = os.path.dirname(filepath)
        
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as file:
            np.save(file,array)
        logging.info('Successfully saved numpy array at its location!')
            
    except Exception as e:
        raise CustomException(e,sys)

# Helper function used to load numpy array into filepath    
def load_numpy_array_data(filepath:str)->np.array:
    try:
        if not os.path.exists(filepath):
            raise Exception(f'The file:{filepath} does not exist')
        
        logging.info('Entered the load numpy array method of Main_utils class')
        
        with open(filepath,'rb') as file:
            logging.info('Exited the load numpy array method successfully')
            return np.load(file)
    except Exception as e:
        raise CustomException(e,sys)

# Helper function to save objects generic on file location
def save_object(filepath:str,obj:object)->None:
    try:
        logging.info('Entered the save object method of Main_utils class')
        dir_name = os.path.dirname(filepath)
        os.makedirs(dir_name,exist_ok=True)
        
        with open(filepath,'wb') as file:
            pickle.dump(obj,file)
        logging.info('Exited the save object method successfully')
    except Exception as e:
        raise CustomException(e,sys)

# Helper function to load objects generic on file location
def load_object(filepath:str)->object:
    try:
        if not os.path.exists(filepath):
            raise Exception(f'The file:{filepath} does not exist')
        
        logging.info('Entered the load object method of Main_utils class')
        
        with open(filepath,'rb') as file:
            logging.info('Exited the load object method successfully')
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)
    
# Reusable evaluate model for evaluation of multiple models and params with given train test data  
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        
        report = {}
        # For every model finding best params calculating r2 score and then creating report
        for i in range(len(models.values())):
            
            model = list(models.values())[i]
            
            param = params[list(models.keys())[i]]
            
            rcv = RandomizedSearchCV(estimator=model,param_distributions=param,cv=3,n_jobs=-1,refit=True,scoring='accuracy')
            
            rcv.fit(x_train,y_train)
            
            model.set_params(**rcv.best_params_)
            model.fit(x_train,y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_score = accuracy_score(y_train,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            # Appending model report in report
            report[list(models.keys())[i]]=test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)