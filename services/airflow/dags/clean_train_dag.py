from datetime import datetime
from airflow import DAG
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

PROJECT = os.environ['PROJECT']


    


with DAG(dag_id="main_dag",
         start_date=datetime(2024, 9, 30),
         schedule_interval="*/20 * * * *",  # Manual trigger
         catchup=False) as dag:

    # Task to trigger another DAG
    trigger_another_dag = TriggerDagRunOperator(
        task_id="trigger_another_dag",
        trigger_dag_id="data_cleaning_and_splitting",  # Replace with your actual DAG ID
        wait_for_completion=True,
        execution_date='{{ ts }}',
        dag=dag
    )

    # Task to train model and log metrics
    train_model = BashOperator(
        task_id="train_and_log_model",
        bash_command=f"source {PROJECT}/.venv/bin/activate && python {PROJECT}/code/models/model_training.py"
    )

    # Task to run Docker Compose build
    run_docker_compose = BashOperator(
        task_id="run_docker_compose",
        bash_command=f"sudo docker-compose --project-directory {PROJECT} -f {PROJECT}/code/deployment/compose.yaml build",  # Update with your path
    )

    # Setting up dependencies
    trigger_another_dag  >> train_model >> run_docker_compose
