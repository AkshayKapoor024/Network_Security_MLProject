import os
import sys

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv('MONGODB_URL')

import pymongo
from src.logging.logger import logging
from src.exception.exception import CustomException
from src.pipelines.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory='./templates')
import pandas as pd

from src.utils.ml_utils.model.estimator import Model

from src.utils.main_utils.utils import load_object

from src.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME

client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods= ['*'],
    allow_headers = ['*']
)

@app.get('/',tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def train_route():
    try:
        # Initiate Trainingg pipeline
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        return Response('Training is Successful')
    except Exception as e:
        raise CustomException(e,sys)
    
@app.post('/predict')
async def predict(request:Request,file:UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)
        print(df.iloc[0])
        model = load_object('final_model/model.pkl')
        
        y_preds = model.predict(df)
        print(y_preds)
        df['Predicted_column']=y_preds
        print(df['Predicted_column'])
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table_striped')
        return templates.TemplateResponse('table.html',{'request':request,'table':table_html})
    except Exception as e:
        raise CustomException(e,sys)
if __name__=='__main__':
    app_run(app,host='0.0.0.0',port=8000)
    
# uvicorn command to run this app 
# uvicorn app:app --reload