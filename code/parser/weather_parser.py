import os
import requests
import pandas as pd
from datetime import timezone 
import datetime 
import sys

API = os.environ["WEATHER_API"]
LAT, LON = 55.7513305, 48.732095
PROJECT_DIR = os.environ["PROJECT"]

dt = datetime.datetime.now(timezone.utc) 
utc_time = dt.replace(tzinfo=timezone.utc) 
TODAY = int(utc_time.timestamp())

def collect_data(start_timestamp, max_values):
    collected_data = []
    response = requests.get(f'https://history.openweathermap.org/data/2.5/history/city?lat={LAT}&lon={LON}&type=hour&start={start_timestamp}&cnt={max_values}&appid={API}')
    if response.status_code == 200:
        response = response.json()
        print(response["cnt"])
        for data in response["list"]:
            
            collected_data.append({"temp": data["main"]["temp"], "feels_like": data["main"]["feels_like"], 
                                    "temp_min": data["main"]["temp_min"], "temp_max": data["main"]["temp_max"],
                                    "pressure": data["main"]["pressure"], 
                                    "humidity": data["main"]["humidity"], 
                                    "clouds": data["clouds"]["all"], "wind_speed": data["wind"]["speed"],
                                    "wind_deg": data["wind"]["deg"], "weather": data["weather"][0]["main"]}) 
    return collected_data
        
        

def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print('run the command with start timestamp in UTC')
        raise KeyError()
    start_timestamp = int(args[0])
    df = pd.DataFrame(columns=["temp", "feels_like", "temp_min", "temp_max", "pressure", "humidity", "clouds", "wind_speed", "wind_deg", "weather_now", "weather_next_hour"])
    for stamp in range(start_timestamp, TODAY, 3600*24*7):
        data = collect_data(stamp, 200)
        for i in range(len(data)-1):
            row = list(data[i].values())
            row.append(data[i+1]['weather'])
            # print(row)
            df.loc[len(df)] = row
    df.to_csv(f'{PROJECT_DIR}/code/datasets/weather_innopolis_from_{start_timestamp}.csv', index=False)


if __name__ == '__main__':
    main()