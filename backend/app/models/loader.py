"""
Loads trained Prophet models from MLflow registry with in-memory cache.
Falls back to local pickle files if the registry is unavailable.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.prophet

logger = logging.getLogger(__name__)

PRODUCTS = ["Jira", "Slack", "Zoom"]


class ModelLoader:
    def __init__(self, tracking_uri: str, cache_dir: str):
        self._cache: dict[str, Any] = {}
        self._versions: dict[str, str] = {}
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        try:
            from mlflow import MlflowClient
            self.client = MlflowClient()
        except Exception:
            self.client = None

    def load_production_model(self, product_name: str) -> Any:
        if product_name in self._cache:
            return self._cache[product_name]

        model = self._load_from_registry(product_name)
        if model is None:
            model = self._load_from_pickle(product_name)
        if model is None:
            raise RuntimeError(f"No model available for {product_name}")

        self._cache[product_name] = model
        return model

    def _load_from_registry(self, product_name: str) -> Any | None:
        if self.client is None:
            return None
        try:
            model_name = f"license_forecast_{product_name.lower()}"
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                # Fall back to any latest version
                versions = self.client.get_latest_versions(model_name)
            if not versions:
                return None
            v = versions[0]
            uri = f"models:/{model_name}/{v.version}"
            model = mlflow.prophet.load_model(uri)
            self._versions[product_name] = str(v.version)
            logger.info(f"Loaded {model_name} v{v.version} from MLflow registry")
            return model
        except Exception as e:
            logger.warning(f"MLflow registry load failed for {product_name}: {e}")
            return None

    def _load_from_pickle(self, product_name: str) -> Any | None:
        pkl_path = self._cache_dir / f"{product_name.lower()}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            self._versions[product_name] = "local-v1"
            logger.info(f"Loaded {product_name} model from local pickle {pkl_path}")
            return model
        return None

    def save_to_pickle(self, product_name: str, model: Any) -> None:
        pkl_path = self._cache_dir / f"{product_name.lower()}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

    def get_model_version(self, product_name: str) -> str:
        return self._versions.get(product_name, "unknown")

    def reload_all(self) -> None:
        self._cache.clear()
        self._versions.clear()
        for product in PRODUCTS:
            try:
                self.load_production_model(product)
            except Exception as e:
                logger.error(f"Failed to load model for {product}: {e}")

    def health_check(self) -> dict[str, bool]:
        return {p: p in self._cache for p in PRODUCTS}


_loader_instance: ModelLoader | None = None


def get_loader() -> ModelLoader:
    if _loader_instance is None:
        raise RuntimeError("ModelLoader not initialized — call init_loader() first")
    return _loader_instance


def init_loader(tracking_uri: str, cache_dir: str) -> ModelLoader:
    global _loader_instance
    _loader_instance = ModelLoader(tracking_uri=tracking_uri, cache_dir=cache_dir)
    return _loader_instance
