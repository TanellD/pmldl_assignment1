FROM python:3.11

ENV MODEL_PATH=/weather_classifier.pkl
ENV PYTHONUNBUFFERED=1
ENV WEATHER_API=326bfa7120e2e53920aa619285082b1c
COPY /code/deployment/app/requirements.txt /requirements.txt
COPY /code/deployment/app/app.py /app.py
RUN pip install --upgrade -r /requirements.txt

EXPOSE 8501
# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]