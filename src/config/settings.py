"""
Configuration settings for the F1 Strategy Simulation project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATION_DATA_DIR = DATA_DIR / "validation"
CACHE_DIR = DATA_DIR / "cache"

# Model paths
LAP_TIME_MODEL_PATH = MODELS_DIR / "lap_time_model.pkl"
TIRE_WEAR_MODEL_PATH = MODELS_DIR / "tire_wear_model.pkl"
WEATHER_MODEL_PATH = MODELS_DIR / "weather_model.pkl"

# FastF1 settings
FASTF1_CACHE_DIR = CACHE_DIR / "fastf1"
FASTF1_SEASONS = [2023, 2024]

# MLflow settings
MLFLOW_TRACKING_URI = "file:" + str(MLRUNS_DIR)
MLFLOW_EXPERIMENT_NAME = "f1-strategy-sim"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Simulation settings
DEFAULT_SIMULATION_STEPS = 100
DEFAULT_WEATHER_UPDATE_INTERVAL = 5  # laps
