from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from routes import MODEL_MAP, models, router
from settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = get_settings()


DEFAULT_THRESHOLD = conf.DEFAULT_THRESHOLD
MLFLOW_TRACKING_URI = conf.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

async def load_models():
    for name, slug in MODEL_MAP.items():
        try:
            logger.info("Loading model '%s' ......", slug)
            run = mlflow.search_runs(experiment_names=[slug], order_by=["metrics.loss ASC"], max_results=1)
            uri = f"runs:/{run.iloc[0]['run_id']}/model-artifact"
            models[name] = mlflow.pyfunc.load_model(uri)
            logger.info("Loaded model '%s' from %s", name, uri)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading model %s, %s: %s", uri, name, exc)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    stop_event = asyncio.Event()

    async def _background_loader():
        try:
            await load_models()
        finally:
            await stop_event.wait()  # attends le signal de shutdown

    task = asyncio.create_task(_background_loader())

    try:
        yield
    finally:
        stop_event.set()
        await task
        logger.info("🧹 Models cleared from memory")
        models.clear()

app = FastAPI(title="Sentiment API", version="1.0.0", lifespan=lifespan)

FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Include all routes
app.include_router(router=router)