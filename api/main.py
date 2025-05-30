from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import joblib
import numpy as np
import mlflow
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

app = FastAPI()

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("sentiment_analysis_tf")
latest_run = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)[0]
model_uri = f"runs:/{latest_run.info.run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
vectorizer = joblib.load("tfidf.joblib")

# Logger vers Azure
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string="InstrumentationKey=USE_ENV_VAR_OR_SECRET"))

class TextInput(BaseModel):
    text: str

class Feedback(BaseModel):
    text: str
    prediction: int
    validated: bool

@app.post("/predict")
def predict_sentiment(input: TextInput):
    X = vectorizer.transform([input.text])
    pred = model.predict(X)
    return {"prediction": int(pred[0])}

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    logger.warning("Prediction Feedback", extra={
        'custom_dimensions': {
            'text': feedback.text,
            'prediction': feedback.prediction,
            'validated': feedback.validated
        }
    })
    return {"status": "feedback logged"}
