"""
Training script for the Lap Time prediction model.

This script processes F1 race data, trains an XGBoost model for lap time prediction,
and logs the model and its metrics to MLflow.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = MODELS_DIR / "laptime_xgboost_pipeline_tuned_v1.joblib"


def load_and_prepare_data():
    """
    Load F1 race data and prepare it for training the lap time prediction model.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Loading data from processed directory...")
    data_path = PROCESSED_DATA_DIR / "seasons_2023_2024_data_enhanced.csv"

    df = pd.read_csv(data_path)

    # Filter data for clean laps only
    if "IsCleanLap" in df.columns:
        df = df[df["IsCleanLap"] == True]

    df = df[df["Deleted"] == False]

    # Drop rows with NaN in important columns
    required_columns = [
        "LapTime_s",
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
    ]
    if "Rainfall" in df.columns:
        required_columns.append("Rainfall")

    df = df.dropna(subset=required_columns)

    # Add rainfall if it doesn't exist
    if "Rainfall" not in df.columns:
        df["Rainfall"] = 0

    # Features used for prediction (matching those in the LapTimeAgent)
    feature_columns = [
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
        "Event",  # Using Event as the circuit identifier
    ]

    # Target variable
    target_column = "LapTime_s"

    # Select features and target
    X = df[feature_columns]
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(
        f"Data loaded and split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples"
    )

    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the lap time model.

    Returns:
        Scikit-learn preprocessing pipeline
    """
    # Identify numeric and categorical columns
    numeric_features = [
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
    ]

    categorical_features = ["Event"]  # Using Event as the circuit identifier

    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_model(X_train, X_test, y_train, y_test):
    """
    Train the XGBoost lap time prediction model with hyperparameter tuning.

    Args:
        X_train, X_test, y_train, y_test: Training and test data

    Returns:
        Trained pipeline and evaluation metrics
    """
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()

    # Create the model pipeline
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror", random_state=42, n_jobs=-1
                ),
            ),
        ]
    )

    # Hyperparameter search space
    param_distributions = {
        "regressor__n_estimators": randint(100, 500),
        "regressor__max_depth": randint(3, 10),
        "regressor__learning_rate": uniform(0.01, 0.3),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bytree": uniform(0.6, 0.4),
        "regressor__min_child_weight": randint(1, 10),
        "regressor__gamma": uniform(0, 1),
    }

    # Randomized search cross-validation
    logger.info("Starting hyperparameter tuning...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    # Fit the model
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")

    # Get the best model
    best_model = search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(
        f"Model performance on test set: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
    )

    # Return the trained model and metrics
    return best_model, {"RMSE": rmse, "MAE": mae, "R2": r2}


def save_model(model, metrics):
    """
    Save the trained model to disk and log to MLflow.

    Args:
        model: Trained model pipeline
        metrics: Evaluation metrics dictionary
    """
    # Save the model
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Log the model and metrics
    with mlflow.start_run(run_name="lap_time_xgboost_v1"):
        # Log parameters
        params = model.get_params()
        for key, value in params.items():
            if key.startswith("regressor__"):
                param_name = key.replace("regressor__", "")
                mlflow.log_param(param_name, value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log the model
        mlflow.sklearn.log_model(model, "lap_time_model")

        # Log the model as an artifact
        mlflow.log_artifact(str(model_path))

    logger.info("Model saved and logged to MLflow")


def main():
    """Main function to train the lap time prediction model."""
    logger.info("Starting lap time model training process")

    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data()

        # Train the model
        model, metrics = train_model(X_train, X_test, y_train, y_test)

        # Save the model and log to MLflow
        save_model(model, metrics)

        logger.info("Lap time model training completed successfully")

    except Exception as e:
        logger.error(f"Error training lap time model: {str(e)}")
        raise


if __name__ == "__main__":
    main()
