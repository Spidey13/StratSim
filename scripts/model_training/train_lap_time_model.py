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
from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV,
)  # Changed from train_test_split
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
model_path = MODELS_DIR / "laptime_xgboost_pipeline_tuned_v2_driver_team.joblib"


def load_and_prepare_data():
    """
    Load F1 race data and prepare it for training the lap time prediction model.

    Returns:
        df_train, df_test: DataFrames for training and testing, respectively.
        feature_columns, target_column: Lists of feature and target column names.
    """
    logger.info("Loading data from processed directory...")
    data_path = PROCESSED_DATA_DIR / "seasons_2023_2024_data_enhanced.csv"

    df = pd.read_csv(data_path)

    # Convert timedelta strings to total seconds for all relevant columns
    time_cols_to_convert = [
        "LapTime",
        "PitOutTime",
        "PitInTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Sector1SessionTime",
        "Sector2SessionTime",
        "Sector3SessionTime",
        "LapStartTime",
    ]

    for col in time_cols_to_convert:
        if col in df.columns:
            # Check if column is already timedelta (e.g., if loaded from a parquet or previous processing)
            if pd.api.types.is_timedelta64_dtype(df[col]):
                df[f"{col}_s"] = df[col].dt.total_seconds()
            elif isinstance(
                df[col].iloc[0], str
            ):  # Handle string format like '0 days 00:01:23.456000000'
                df[f"{col}_s"] = df[col].apply(
                    lambda x: pd.to_timedelta(x).total_seconds()
                    if pd.notna(x)
                    else np.nan
                )
            else:
                logger.warning(
                    f"Column '{col}' is not timedelta or string, skipping conversion to seconds."
                )

    # Convert LapStartDate to datetime and sort (important for TimeSeriesSplit)
    if "LapStartDate" in df.columns:
        df["LapStartDate"] = pd.to_datetime(df["LapStartDate"])
        df = df.sort_values(by="LapStartDate").reset_index(drop=True)
    else:
        logger.warning(
            "Warning: 'LapStartDate' not found. TimeSeriesSplit might not behave as expected without sorting by time."
        )
        # If LapStartDate is not available, try to sort by 'Time_s' which is created from 'Time'
        if "Time_s" in df.columns:
            df = df.sort_values(by="Time_s").reset_index(drop=True)
        else:
            logger.error(
                "Neither 'LapStartDate' nor 'Time_s' available for temporal sorting. This may affect TimeSeriesSplit validity."
            )

    # Filter data for clean laps only
    initial_rows_before_clean_filter = df.shape[0]
    if "IsCleanLap" in df.columns:
        df = df[df["IsCleanLap"] == True].copy()
    else:  # Fallback if IsCleanLap is not present, use previous logic
        df = df[df["Deleted"] == False].copy()
        if "IsOutlap" in df.columns:
            df = df[df["IsOutlap"] == False]
        if "IsInlap" in df.columns:
            df = df[df["IsInlap"] == False]
    logger.info(
        f"Filtered down to {df.shape[0]} clean laps from {initial_rows_before_clean_filter} total laps."
    )

    # Add rainfall if it doesn't exist or fill NaN (assuming 0 for no rain)
    if "Rainfall" not in df.columns:
        logger.warning("'Rainfall' column not found. Adding it with default value 0.")
        df["Rainfall"] = 0
    else:
        # Fill NaN values in Rainfall with 0, assuming NaN means no rain
        df["Rainfall"] = df["Rainfall"].fillna(0)

    # Features used for prediction
    # Removed 'PitStopDuration'
    feature_columns = [
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
        "Event",  # Using Event as the circuit identifier
        "Driver",
        "Team",
        # Adding more numerical features based on initial data exploration
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "SpeedST",
        "SpeedI1_Diff",
        "SpeedI2_Diff",
        "SpeedFL_Diff",
        "SpeedST_Diff",
        "TempDelta",
        "WetTrack",
        "GripLevel",
        "WeatherStability",
        "TyreWearPercentage",  # Added TyreWearPercentage as an input feature
        "WindSpeed_Avg",
        "TrackCondition",
    ]

    # Target variable
    target_column = "LapTime_s"

    # Drop rows with NaN in important columns *after* rainfall handling
    # Ensure all features in feature_columns list are present in the dataframe
    # Filter out columns that do not exist before dropping NaNs
    existing_feature_columns = [col for col in feature_columns if col in df.columns]

    required_columns_for_dropna = existing_feature_columns + [target_column]

    initial_rows_before_dropna = df.shape[0]
    df = df.dropna(
        subset=required_columns_for_dropna
    ).copy()  # Added .copy() to prevent SettingWithCopyWarning
    if df.shape[0] < initial_rows_before_dropna:
        logger.warning(
            f"Dropped {initial_rows_before_dropna - df.shape[0]} rows due to NaN values in required columns."
        )

    # Ensure all drivers and teams are strings to prevent issues with OneHotEncoder
    df["Driver"] = df["Driver"].astype(str)
    df["Team"] = df["Team"].astype(str)
    df["Event"] = df["Event"].astype(str)
    df["Compound"] = df["Compound"].astype(str)
    df["Session"] = df["Session"].astype(str)

    # Use the full filtered and cleaned dataframe for TimeSeriesSplit
    # We pass the full dataframe to simulate a time-series split on the entire dataset
    # The actual train/test split will happen implicitly within TimeSeriesSplit's folds
    logger.info(f"Data ready for TimeSeriesSplit: {df.shape[0]} samples.")

    # We will return the full preprocessed DataFrame,
    # and the split will be handled by TimeSeriesSplit in RandomizedSearchCV
    return df, existing_feature_columns, target_column


def create_preprocessing_pipeline(feature_columns, df_for_dtypes):
    """
    Create a preprocessing pipeline for the lap time model.
    Dynamically identifies numeric and categorical columns based on the DataFrame's dtypes.

    Args:
        feature_columns (list): List of all feature names.
        df_for_dtypes (pd.DataFrame): A sample DataFrame to infer column dtypes.

    Returns:
        Scikit-learn preprocessing pipeline (ColumnTransformer)
    """
    numeric_features = []
    categorical_features = []

    for col in feature_columns:
        if col in df_for_dtypes.columns:  # Ensure column exists before checking dtype
            # Check for numeric dtypes (int, float)
            if pd.api.types.is_numeric_dtype(df_for_dtypes[col]):
                numeric_features.append(col)
            # Treat 'Year' and 'TrackCondition' as categorical if they were int before
            # and we want them one-hot encoded
            elif col in [
                "Year",
                "TrackCondition",
                "WetTrack",
            ] and pd.api.types.is_integer_dtype(df_for_dtypes[col]):
                categorical_features.append(col)
            # Check for object/string dtypes (for Driver, Team, Event, Compound, Session)
            elif pd.api.types.is_object_dtype(df_for_dtypes[col]):
                categorical_features.append(col)
        else:
            logger.warning(
                f"Feature column '{col}' not found in the DataFrame. It will be ignored."
            )

    # Remove duplicates just in case
    numeric_features = list(set(numeric_features))
    categorical_features = list(set(categorical_features))

    logger.info(f"Identified numeric features for preprocessing: {numeric_features}")
    logger.info(
        f"Identified categorical features for preprocessing: {categorical_features}"
    )

    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any columns not specified in transformers
    )

    return preprocessor


def train_model(
    df, feature_columns, target_column
):  # Removed df_train, df_test as separate args
    """
    Train the XGBoost lap time prediction model with hyperparameter tuning.

    Args:
        df: Full DataFrame containing features and target, sorted by time for TimeSeriesSplit.
        feature_columns, target_column: Lists of feature and target column names.

    Returns:
        Trained pipeline and evaluation metrics
    """
    X = df[feature_columns]
    y = df[target_column]

    # Create preprocessing pipeline dynamically based on X's dtypes
    preprocessor = create_preprocessing_pipeline(feature_columns, X)

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

    # Hyperparameter search space (ADJUSTED)
    param_distributions = {
        "regressor__n_estimators": randint(200, 1500),  # Increased max range
        "regressor__max_depth": randint(5, 25),  # Wider range for depth
        "regressor__learning_rate": uniform(0.005, 0.15),  # Adjusted range
        "regressor__subsample": uniform(0.7, 0.3),  # Minimized to 0.7
        "regressor__colsample_bytree": uniform(0.7, 0.3),  # Minimized to 0.7
        "regressor__gamma": uniform(0, 0.2),  # Reduced max gamma
        "regressor__min_child_weight": randint(
            1, 5
        ),  # Narrowed range for min_child_weight
        "regressor__reg_alpha": uniform(0, 1.0),
        "regressor__reg_lambda": uniform(0, 1.0),
    }

    # TimeSeriesSplit for cross-validation (CRUCIAL CHANGE)
    tscv = TimeSeriesSplit(n_splits=5)  # 5 splits for evaluation

    # Randomized search cross-validation
    logger.info("Starting hyperparameter tuning with TimeSeriesSplit...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=100,  # Increased iterations for more thorough search
        cv=tscv,  # Use TimeSeriesSplit
        scoring="neg_mean_squared_error",
        verbose=2,  # More verbose output during search
        n_jobs=-1,
        random_state=42,
        error_score="raise",
    )

    # Fit the model
    search.fit(
        X, y
    )  # Fit on the full dataset, TimeSeriesSplit handles train/validation splits

    logger.info(f"Best parameters: {search.best_params_}")

    # Get the best model
    best_model = search.best_estimator_

    # For evaluation, we need to manually perform a final train/test split on the sorted data
    # This simulates predicting on future data after the training period
    # We use the 'Year' column if available, or a simple time-based split

    # Identify unique years and determine split point
    unique_years = sorted(df["Year"].unique()) if "Year" in df.columns else []

    if len(unique_years) >= 2:
        split_year_idx = int(
            len(unique_years) * 0.8
        )  # 80% for training, 20% for testing by year
        if split_year_idx >= len(
            unique_years
        ):  # Ensure index is within bounds if only 1 year of data
            split_year_idx = len(unique_years) - 1
        split_year = unique_years[split_year_idx]

        train_mask = df["Year"] <= split_year
        test_mask = df["Year"] > split_year

        if df[test_mask].empty and len(unique_years) > 1:
            logger.warning(
                f"Time-aware test set is empty with split_year={split_year}. Attempting split by date index."
            )
            # Fallback to simple time-based index split if year split fails
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

        logger.info(
            f"Final evaluation split: Training on years <= {split_year}, Testing on years > {split_year}"
        )

    else:
        logger.warning(
            "Less than 2 unique years for time-aware split. Performing 80/20 time-based index split."
        )
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if X_test.empty:
        logger.error(
            "The test set for final evaluation is empty. Cannot compute metrics."
        )
        return best_model, {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}

    # Evaluate on the final test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(
        f"Model performance on final test set: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
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
    with mlflow.start_run(run_name="lap_time_xgboost_v2_driver_team_tuned"):
        # Log parameters - extract only regressor parameters
        # Access best parameters from the RandomizedSearchCV object if available
        # Note: 'model' here is the best_estimator_ which is a Pipeline, not the RandomizedSearchCV object itself
        # So we access its parameters directly.
        params_to_log = {
            k.replace("regressor__", ""): v
            for k, v in model.get_params().items()
            if k.startswith("regressor__")
        }

        for param_name, param_value in params_to_log.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log the model with specific artifact path for clarity
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lap_time_model_v2_tuned",
            registered_model_name="LapTimeXGBoostModelV2",
        )

        # Also log the joblib file as an artifact
        mlflow.log_artifact(str(model_path), "model_joblib")

    logger.info("Model saved and logged to MLflow")


def main():
    """Main function to train the lap time prediction model."""
    logger.info("Starting lap time model training process")

    try:
        # Load and prepare data (returns full df for TimeSeriesSplit)
        df, feature_columns, target_column = load_and_prepare_data()

        # Train the model (TimeSeriesSplit happens inside train_model)
        model, metrics = train_model(df, feature_columns, target_column)

        # Save the model and log to MLflow
        save_model(model, metrics)

        logger.info("Lap time model training completed successfully")

    except Exception as e:
        logger.error(f"Error training lap time model: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
