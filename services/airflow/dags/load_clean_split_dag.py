from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.decorators import task
import os


PROJECT = os.environ['PROJECT']


with DAG(dag_id="data_cleaning_and_splitting",
         schedule_interval=None,
         catchup=False) as dag:

   
    @task()
    def read_data():
        # Read the raw data into a DataFrame
        df = pd.read_csv(f'{PROJECT}/data/raw/data.csv')
        return df

    @task()
    def clean_data(df):
        df.fillna(method='ffill', inplace=True)  # Impute missing values
        return df

    @task()
    def split_data(df):
        # Split the data into training and testing datasets
        from sklearn.model_selection import train_test_split
        preprocessed = f'{PROJECT}/data/preprocessed'
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == 'Rain' else 0)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        train_df = pd.concat([train_X, train_y], axis=1)
        test_df = pd.concat([test_X, test_y], axis=1)
        train_df.to_csv(f'{preprocessed}/train_data.csv', index=False)
        test_df.to_csv(f'{preprocessed}/test_data.csv', index=False)

    # Set up the DAG tasks
    raw_data = read_data()
    cleaned_data = clean_data(raw_data)
    split_data(cleaned_data)


