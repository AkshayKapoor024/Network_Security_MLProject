import json
import os
import sys

import pandas as pd 
import numpy as np 
import pymongo

from src.logging.logger import logging
from src.exception.exception import CustomException

from dotenv import load_dotenv
# Set of root certificates used by python to make secure http connections and make requests
import certifi

# Load ENV variables
load_dotenv()

uri = os.getenv("MONGODB_URL")

# ca stores trusted certificate authorities
ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    # Function to convert Csv data to json
    def csv_to_json(self,filepath):
        try:
            logging.info('Entered CSV TO JSON converter function')
            data = pd.read_csv(filepath)
            data.reset_index(drop=True,inplace=True)

            records = list(json.loads((data.T.to_json())).values())
            logging.info('CSV to json converted successfully!!')
            return records
        
        except Exception as e:
            raise CustomException(e,sys)
    # Function to store data on mongodb cloud   
    def insert_into_mongodb(self,records,database,collection):
        try:
            logging.info('Starting Mongodb export !')
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client=pymongo.MongoClient(uri)

            self.database=self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            
            self.collection.insert_many(self.records)
            logging.info('MongoDb export successful!!')
            
            return len(self.records)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    FILE_PATH = "Data\\phisingData.csv"
    DATABASE = "Ml_projects"
    COLLECTION = "Network_Phising_Project"
    
    network_obj = NetworkDataExtract()
    records = network_obj.csv_to_json(filepath=FILE_PATH)
    no_of_records = network_obj.insert_into_mongodb(records=records,collection=COLLECTION,database=DATABASE)
    print(no_of_records)