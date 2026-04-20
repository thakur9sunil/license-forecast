from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "license_forecasting"
    model_registry_uri: str = "sqlite:///./mlflow/mlflow.db"
    artifact_root: str = "./mlflow/artifacts"
    model_cache_dir: str = "./models"
    data_dir: str = "./data"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
