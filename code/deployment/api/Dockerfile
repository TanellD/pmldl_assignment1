FROM python:3.11

ENV MODEL=/weather_classifier.pkl
ENV PYTHONUNBUFFERED=1

COPY /code/deployment/api/requirements.txt /requirements.txt
COPY /code/deployment/api/app.py /app.py
COPY /models/weather_classifier.pkl /weather_classifier.pkl
RUN pip install --upgrade -r /requirements.txt

EXPOSE 80
# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]