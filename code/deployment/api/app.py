from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import List
import numpy as np
import os

# get path to the model from environment
MODEL_PATH = os.environ["MODEL"]
# columns in data
columns = ["temp", "feels_like", "temp_min", "temp_max", "pressure", "humidity", "clouds", "wind_speed", "wind_deg", "weather_now"]

# class to parse the json to df
class Item(BaseModel):
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: float
    humidity: float
    clouds: float
    wind_speed: float
    wind_deg: float
    weather_now: str

app = FastAPI()
# best threshold based on learning
best_treshold = 0.25252525252525254
# load the model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(X: Item):
    global model
    df = pd.DataFrame([[
        X.temp,
        X.feels_like,
        X.temp_min,
        X.temp_max,
        X.pressure,
        X.humidity,
        X.clouds,
        X.wind_speed, 
        X.wind_deg,
        X.weather_now
    ]], columns=columns)
    prediction = model.predict_proba(df)
    prediction = prediction[:, 1] > best_treshold
    if prediction:
        return 'It could be rainny in an hour'
    return 'No rain in an hour'

@app.get("/")
def read_root():
    return {"message": "Welcome to the model API!"}