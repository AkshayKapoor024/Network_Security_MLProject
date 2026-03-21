import yaml
from src.exception.exception import CustomException
from src.logging.logger import logging

import os,sys
import numpy as np
import dill
import pickle

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

# Helper function to save objects generic on file location
def save_object(filepath:str,obj:object)->None:
    try:
        logging.info('Entered the save object method of Main_utils class')
        
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        
        with open(filepath,'wb') as file:
            pickle.dump(obj,file)
        logging.info('Exited the save object method successfully')
    except Exception as e:
        raise CustomException(e,sys)