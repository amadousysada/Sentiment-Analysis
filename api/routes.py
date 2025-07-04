import logging
from functools import lru_cache
from typing import Dict, List

import mlflow
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from settings import get_settings
from telemetry import get_metrics
from utils import (bert_tokenize, clean_tokenize, ftext_tokenizer,
                   w2vec_tokenizer)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}

prediction_counter, total_feedback_counter, positive_feedback_counter, negative_feedback_counter = get_metrics()

MODEL_MAP = {
    #'model simple (logististic reg)': 'word2vec-logistic-regression-with-optimized-hyperparameters',
    #'model avancé': 'fasttext-lstm-advanced-model',
    'model avancé': 'bert-model',
    #'model bert': 'bert-model'
}

LABEL_MAP = {
    "0": "negative",
    "1": "positive",
}

TOKENIZER_MAP = {
    #"model simple (logististic reg)": w2vec_tokenizer,
    #"model avancé": ftext_tokenizer,
    "model avancé": bert_tokenize,
    #"model bert": bert_tokenize
}

class PredictRequest(BaseModel):
    text: str
    model_name: str

class PredictResponse(BaseModel):
    probabilities: List[float]
    labels: List[str]
    model: str

class FeedbackRequest(BaseModel):
    model: str
    text: str
    probability: float
    validated: bool
    predicted_sentiment: str

@lru_cache
def get_models() -> dict[str, mlflow.pyfunc.PyFuncModel]:
    return models

@router.get("/health-check/", summary="Healthcheck")
def root() -> Dict[str, str]:
    return {"status": "ok", "models": "|".join(models.keys())}


@router.get("/models/", summary="All models")
def fetch_models():
    return {"models": [key for key in models.keys()]}

@router.post("/predict/", response_model=PredictResponse)
async def predict(
    *,
    payload: PredictRequest,
) -> PredictResponse:
    """Return sentiment probabilities & labels using the chosen model."""
    model_name = payload.model_name
    if model_name not in models.keys():
        logger.error(models.keys())
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    model = models[model_name]
    threshold = get_settings().DEFAULT_THRESHOLD

    texts = payload.text

    tokens = clean_tokenize(texts)
    tokenizer = TOKENIZER_MAP.get(model_name)
    if not tokenizer:
        raise HTTPException(status_code=500, detail="Tokenizer not configured for model")
    embeddings = tokenizer(tokens)
    proba = model.predict(embeddings)
    logger.info(f"Predicted: {proba}")
    logger.info("Embeddings shape=%s dtype=%s snippet=%s",
                 embeddings.shape,
                 embeddings.dtype,
                 embeddings.flatten()[:5])

    # If model returns 2‑dim (e.g., [[p_neg, p_pos]]), take positive class
    if proba.ndim == 2 and proba.shape[1] > 1:
        proba = proba[:, 1]
    proba = proba.flatten().tolist()

    labels = [LABEL_MAP[str(int(p >= threshold))] for p in proba]

    proba = [
        (1 - p) if int(p >= threshold) == 0 else p
        for p in proba
    ]
    prediction_counter.add(1, {"model": model_name, "text": texts, "labels_predicted": ",".join(labels)})

    return PredictResponse(probabilities=proba, labels=labels, model=model_name)


@router.post("/feedback/")
async def feedback(req: FeedbackRequest):
    # increment feedback metric
    total_feedback_counter.add(
        1,
        {
            "model": req.model,
            "validated": str(req.validated),
            "text": req.text,
            "probability": req.probability,
            "predicted_sentiment": req.predicted_sentiment
        }
    )
    if not req.validated:
        negative_feedback_counter.add(
            1,
            {
                "model": req.model,
                "validated": str(req.validated),
                "text": req.text,
                "probability": req.probability,
                "predicted_sentiment": req.predicted_sentiment
            }
        )
        logger.warning(
            "Prediction invalid: model=%s text=%s prob=%.3f sentiment=%s",
            req.model, req.text, req.probability,req.predicted_sentiment
        )
    else:
        positive_feedback_counter.add(
            1,
            {
                "model": req.model,
                "validated": str(req.validated),
                "text": req.text,
                "probability": req.probability,
                "predicted_sentiment": req.predicted_sentiment
            }
        )
        logger.info(
            "Prediction validated: model=%s text=%s prob=%.3f",
            req.model, req.text, req.probability,
        )
    return {"status": "ok"}