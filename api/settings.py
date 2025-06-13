from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APPLICATIONINSIGHTS_CONNECTION_STRING: str = ""
    MLFLOW_TRACKING_URI: str = ""
    DEFAULT_THRESHOLD: float = 0.5
    SERVICE_NAME: str = ""

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()