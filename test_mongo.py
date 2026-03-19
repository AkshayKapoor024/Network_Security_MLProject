
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from src.logging.logger import logging
import sys
from dotenv import load_dotenv
import os
from src.exception.exception import CustomException

# load env variables
load_dotenv()

# get value
uri = os.getenv("MONGODB_URL")


# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logging.info('Error while connecting to MongoDb')
    raise CustomException(e,sys)