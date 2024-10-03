import streamlit as st
import pandas as pd
import numpy as np
from datetime import timezone
import datetime
import requests
import os

# get env variable to access the weather api
API = os.environ["WEATHER_API"]
# coordinates of Innopolis to search for the weather now
LAT, LON = 55.7513305, 48.732095

# columns in dataframe and labels to provide more user friendly naming
columns = ["temp", "feels_like", "temp_min", "temp_max", "pressure", "humidity", "clouds", "wind_speed", "wind_deg", "weather_now"]
labels = ["Temp", "Feels", "Temp min", "Temp max", "Pressure", "Humidity", "Clouds", "Wind Speed", "Wind Degree"]

# function to collect weather data for now
def collect_data(start_timestamp, max_values):
    collected_data = []
    response = requests.get(f'https://history.openweathermap.org/data/2.5/history/city?lat={LAT}&lon={LON}&type=hour&start={start_timestamp}&cnt={max_values}&appid={API}')
    if response.status_code == 200:
        response = response.json()
        try:
            for data in response["list"]:
                
                collected_data.append({"temp": data["main"]["temp"], "feels_like": data["main"]["feels_like"], 
                                        "temp_min": data["main"]["temp_min"], "temp_max": data["main"]["temp_max"],
                                        "pressure": data["main"]["pressure"], 
                                        "humidity": data["main"]["humidity"], 
                                        "clouds": data["clouds"]["all"], "wind_speed": data["wind"]["speed"],
                                        "wind_deg": data["wind"]["deg"], "weather": data["weather"][0]["main"]}) 
        except:
            print(response)
    return collected_data

# streamlit user interface to interact with the model and api
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=columns, data=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Clouds']])

def get_weather_now():
    dt = datetime.datetime.now(timezone.utc) 
    utc_time = dt.replace(tzinfo=timezone.utc) 
    now = int(utc_time.timestamp())
    now -= 3600
    collected = collect_data(now, 1)
    st.session_state.data.iloc[0, :] = list(collected[0].values())
    return st.session_state.data
    
    


column_configs = {key: st.column_config.NumberColumn(label=labels[i], required=True) for i, key in enumerate(columns[:-1])}
column_configs['weather_now'] = st.column_config.SelectboxColumn(
            label="Weather Now",
            options=[
                'Clouds', 
                'Clear', 
                'Rain'
            ],
            required=True,
        )

edited = st.data_editor(data=st.session_state.data, use_container_width=False, hide_index=True, column_order=columns, column_config=column_configs)
if st.button('Autofill'):
    data = get_weather_now()
    st.session_state.data = data
    edited = data
    st.rerun()
if st.button("Make prediction"):
    payload=edited.iloc[0].to_json()
    st.write(requests.post("http://fastapi:8000/predict", data=payload).text)
    
    
