# src/models/train_tire_wear_model.py

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
from catboost import CatBoostRegressor
import warnings

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress specific FutureWarning from pandas
warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.",
    category=FutureWarning,
)
# Suppress MLflow UserWarning about inferred integer columns if you're confident in your type handling
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
# Suppress CatBoost warnings (e.g., about no GPU)
warnings.filterwarnings("ignore", category=UserWarning, module="catboost")


# Assuming src.config.settings exists and contains paths
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
# Updated model path name to reflect CatBoost and new version
model_path = MODELS_DIR / "tire_wear_catboost_pipeline_tuned_v1.joblib"

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME + "_TireWear_CatBoost")

# Define features for Tire Wear model based on data snippet - these are global for consistency
GLOBAL_NUMERICAL_FEATURES = [
    "LapNumberInStint",
    "TyreLifeSquared",
    "TyreLifeCubed",
    "TrackTemp_Avg",
    "AirTemp_Avg",
    "Humidity_Avg",
    "Rainfall",
    "SpeedST",
    "SpeedI1",
    "SpeedI2",
    "PrevLapTimeDegradation_s",
    "LapNumberInStint_TrackTemp_Interaction",
    "LapNumberInStint_IsSoft_Interaction",
    "LapNumberInStint_IsMedium_Interaction",
    "LapNumberInStint_IsHard_Interaction",
    "TrackTemp_Avg_IsSoft_Interaction",
    "TrackTemp_Avg_IsMedium_Interaction",
    "TrackTemp_Avg_IsHard_Interaction",
    "IsSoft",
    "IsMedium",
    "IsHard",
]

GLOBAL_CATEGORICAL_FEATURES = [
    "Compound",
    "Event",
    "Driver",
    "Team",
    "Year",
]

# TARGET VARIABLE DEFINITION
TARGET_COL_TIRE_WEAR = "LapTimeDegradationPerLap_s"


def load_and_prepare_tire_wear_data():
    """
    Load F1 race data and prepare it for training the tire wear prediction model.
    This will involve calculating the LapTimeDegradationPerLap_s target.
    """
    logger.info("Loading data from processed directory for tire wear model...")
    data_path = PROCESSED_DATA_DIR / "seasons_2023_2024_data_enhanced.csv"
    df = pd.read_csv(data_path)

    # --- Data Cleaning & Filtering ---
    # Convert timedelta strings to seconds, or ensure LapTime_s exists
    if "LapTime" in df.columns and pd.api.types.is_timedelta64_dtype(df["LapTime"]):
        df.loc[:, "LapTime_s"] = df["LapTime"].dt.total_seconds()
    elif "LapTime" in df.columns and isinstance(df["LapTime"].iloc[0], str):
        try:
            df.loc[:, "LapTime_s"] = df["LapTime"].apply(
                lambda x: pd.to_timedelta(x).total_seconds() if pd.notna(x) else np.nan
            )
        except Exception as e:
            logger.warning(
                f"Could not convert 'LapTime' string to timedelta, assuming 'LapTime_s' exists: {e}"
            )
            if "LapTime_s" not in df.columns:
                raise ValueError(
                    "Neither 'LapTime' could be converted nor 'LapTime_s' found."
                )

    # NEW FIX: Explicitly convert numerical columns including LapNumberInStint
    # and ensure 'Year' is an integer for the train/test split.
    numerical_cols_to_convert = [
        "LapNumberInStint",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
        "SpeedST",
        "SpeedI1",
        "SpeedI2",
    ]
    for col in numerical_cols_to_convert:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Converting '{col}' to numeric type.")
                df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[:, col] = df[col].fillna(
                df[col].mean()
            )  # Fill NaNs for numerical features

    # CRITICAL FIX for 'Year' column: Ensure it's an integer before the train/test split
    if "Year" in df.columns:
        df.loc[:, "Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df.loc[:, "Year"] = (
            df["Year"].fillna(-1).astype(int)
        )  # Use -1 if years are always positive
    else:
        logger.error(
            "'Year' column not found, which is essential for time-aware splitting."
        )
        raise KeyError("'Year' column is missing from the DataFrame.")

    # Filter for clean laps only (crucial for degradation calculation)
    if "IsCleanLap" in df.columns:
        df_filtered = df[df["IsCleanLap"] == True].copy()
    else:
        logger.warning(
            "No 'IsCleanLap' column found. Attempting to infer clean laps from Inlap/Outlap/Deleted."
        )
        df_filtered = df[(df["Deleted"] == False)].copy()
        if "IsOutlap" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["IsOutlap"] == False].copy()
        if "IsInlap" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["IsInlap"] == False].copy()

    # Create local copies of feature lists to modify based on available columns
    numerical_features_used = GLOBAL_NUMERICAL_FEATURES[:]
    categorical_features_used = GLOBAL_CATEGORICAL_FEATURES[:]

    # Ensure categorical columns exist and fill NaNs with a placeholder
    # and convert to string type for CatBoost, EXCEPT for 'Year'
    # which is already int for the split and can be handled as int categorical by CatBoost.
    for col in categorical_features_used:
        if col in df_filtered.columns:
            if col != "Year":  # Do NOT convert 'Year' to string here
                df_filtered.loc[:, col] = (
                    df_filtered[col].fillna("Unknown_Category").astype(str)
                )
            # else: 'Year' is already int from the earlier explicit conversion.
            # CatBoost can handle integer categorical features directly.
        else:
            logger.error(
                f"Missing essential categorical column: '{col}'. Removing from features."
            )
            if col in categorical_features_used:
                categorical_features_used.remove(col)

    # --- Feature Engineering for Tire Wear Target and New Interaction Features ---
    if "Stint" not in df_filtered.columns:
        logger.error(
            "'Stint' column not found, which is essential for stint-based degradation calculation."
        )
        raise KeyError("'Stint' column is missing from the DataFrame.")
    if "LapNumberInStint" not in df_filtered.columns:
        logger.error(
            "'LapNumberInStint' column not found, which is essential for degradation calculation."
        )
        raise KeyError("'LapNumberInStint' column is missing from the DataFrame.")

    # New derived features (ensuring base columns are present)
    if "LapNumberInStint" in df_filtered.columns:
        df_filtered.loc[:, "TyreLifeSquared"] = df_filtered["LapNumberInStint"] ** 2
        df_filtered.loc[:, "TyreLifeCubed"] = df_filtered["LapNumberInStint"] ** 3
    else:
        df_filtered.loc[:, "TyreLifeSquared"] = 0
        df_filtered.loc[:, "TyreLifeCubed"] = 0

    # NEW INTERACTION FEATURES
    if (
        "LapNumberInStint" in df_filtered.columns
        and "TrackTemp_Avg" in df_filtered.columns
    ):
        df_filtered.loc[:, "LapNumberInStint_TrackTemp_Interaction"] = (
            df_filtered["LapNumberInStint"] * df_filtered["TrackTemp_Avg"]
        )
    else:
        df_filtered.loc[:, "LapNumberInStint_TrackTemp_Interaction"] = 0

    # Compound-specific interactions (assuming IsSoft, IsMedium, IsHard are 0/1 booleans)
    if "IsSoft" in df_filtered.columns and "LapNumberInStint" in df_filtered.columns:
        df_filtered.loc[:, "LapNumberInStint_IsSoft_Interaction"] = (
            df_filtered["LapNumberInStint"] * df_filtered["IsSoft"]
        )
    else:
        df_filtered.loc[:, "LapNumberInStint_IsSoft_Interaction"] = 0
    if "IsMedium" in df_filtered.columns and "LapNumberInStint" in df_filtered.columns:
        df_filtered.loc[:, "LapNumberInStint_IsMedium_Interaction"] = (
            df_filtered["LapNumberInStint"] * df_filtered["IsMedium"]
        )
    else:
        df_filtered.loc[:, "LapNumberInStint_IsMedium_Interaction"] = 0
    if "IsHard" in df_filtered.columns and "LapNumberInStint" in df_filtered.columns:
        df_filtered.loc[:, "LapNumberInStint_IsHard_Interaction"] = (
            df_filtered["LapNumberInStint"] * df_filtered["IsHard"]
        )
    else:
        df_filtered.loc[:, "LapNumberInStint_IsHard_Interaction"] = 0

    if "IsSoft" in df_filtered.columns and "TrackTemp_Avg" in df_filtered.columns:
        df_filtered.loc[:, "TrackTemp_Avg_IsSoft_Interaction"] = (
            df_filtered["TrackTemp_Avg"] * df_filtered["IsSoft"]
        )
    else:
        df_filtered.loc[:, "TrackTemp_Avg_IsSoft_Interaction"] = 0
    if "IsMedium" in df_filtered.columns and "TrackTemp_Avg" in df_filtered.columns:
        df_filtered.loc[:, "TrackTemp_Avg_IsMedium_Interaction"] = (
            df_filtered["TrackTemp_Avg"] * df_filtered["IsMedium"]
        )
    else:
        df_filtered.loc[:, "TrackTemp_Avg_IsMedium_Interaction"] = 0
    if "IsHard" in df_filtered.columns and "TrackTemp_Avg" in df_filtered.columns:
        df_filtered.loc[:, "TrackTemp_Avg_IsHard_Interaction"] = (
            df_filtered["TrackTemp_Avg"] * df_filtered["IsHard"]
        )
    else:
        df_filtered.loc[:, "TrackTemp_Avg_IsHard_Interaction"] = 0

    # Group by unique stint and calculate baseline lap time

    stint_id_cols = [
        "Driver",
        "Event",
        "Year",
        "Stint",
    ]  # Year is now in categorical_features_used, so treat as part of ID

    # Sort data for correct shifting later
    df_filtered = df_filtered.sort_values(
        by=stint_id_cols + ["LapNumberInStint"]
    ).copy()

    # Filter out very short stints or those with too few clean laps for baseline
    df_filtered_stints = (
        df_filtered.groupby(stint_id_cols)
        .filter(
            lambda x: (x["LapNumberInStint"].max() >= 5)
            and (x["LapNumberInStint"] <= 3).sum() >= 1
        )
        .copy()
    )

    if df_filtered_stints.empty:
        logger.error(
            "No valid stints found after filtering for length and clean laps. Cannot train model."
        )
        raise ValueError("Filtered DataFrame for stints is empty.")

    # Calculate baseline: Mean of first 3 clean laps in a stint
    baseline_laps = df_filtered_stints[df_filtered_stints["LapNumberInStint"] <= 3]
    stint_baseline_laptimes = (
        baseline_laps.groupby(stint_id_cols)["LapTime_s"].mean().reset_index()
    )
    stint_baseline_laptimes.rename(
        columns={"LapTime_s": "BaselineLapTime_s"}, inplace=True
    )

    df_merged = pd.merge(
        df_filtered_stints, stint_baseline_laptimes, on=stint_id_cols, how="left"
    )

    df_merged.dropna(subset=["BaselineLapTime_s"], inplace=True)

    # Calculate LapTimeDegradation (absolute increase from baseline)
    df_merged.loc[:, "LapTimeDegradation_s"] = (
        df_merged["LapTime_s"] - df_merged["BaselineLapTime_s"]
    )

    # NEW FEATURE: Lagged degradation (important to keep)
    df_merged.loc[:, "PrevLapTimeDegradation_s"] = df_merged.groupby(stint_id_cols)[
        "LapTimeDegradation_s"
    ].shift(1)
    df_merged.loc[:, "PrevLapTimeDegradation_s"].fillna(0, inplace=True)

    # --- TARGET CALCULATION AND TRANSFORMATION ---
    # Calculate LapTimeDegradationPerLap_s (our target)
    df_merged.loc[:, TARGET_COL_TIRE_WEAR] = df_merged.apply(
        lambda row: (row["LapTimeDegradation_s"] / (row["LapNumberInStint"] - 1))
        if row["LapNumberInStint"] > 1
        else 0,
        axis=1,
    )
    # Ensure degradation is non-negative and clamp to a reasonable upper bound (1.0 seconds per lap)
    df_merged.loc[:, TARGET_COL_TIRE_WEAR] = np.clip(
        df_merged[TARGET_COL_TIRE_WEAR], 0, 1.0
    )

    # Apply log1p transformation to the target variable
    df_merged.loc[:, TARGET_COL_TIRE_WEAR] = np.log1p(df_merged[TARGET_COL_TIRE_WEAR])

    # --- Final Data Preparation ---
    # Ensure all numerical_features_used are present after feature engineering
    final_numerical_features = [
        f for f in numerical_features_used if f in df_merged.columns
    ]
    final_categorical_features = [
        f for f in categorical_features_used if f in df_merged.columns
    ]

    all_features = final_numerical_features + final_categorical_features
    required_cols_final = all_features + [TARGET_COL_TIRE_WEAR]

    missing_cols_final = [
        col for col in required_cols_final if col not in df_merged.columns
    ]
    if missing_cols_final:
        logger.error(
            f"Missing required columns for final dataset: {missing_cols_final}. Available columns: {df_merged.columns.tolist()}"
        )
        raise ValueError(f"Missing required columns: {missing_cols_final}")

    df_final = df_merged[required_cols_final].copy()

    initial_rows = df_final.shape[0]
    df_final.dropna(subset=required_cols_final, inplace=True)
    dropped_rows = initial_rows - df_final.shape[0]
    if dropped_rows > 0:
        logger.warning(
            f"Dropped {dropped_rows} rows due to NaN values in final feature/target set."
        )

    # Time-aware split: Train on 2023, Test on 2024
    train_df = df_final[df_final["Year"] == 2023].copy()
    test_df = df_final[df_final["Year"] == 2024].copy()

    if test_df.empty or train_df.empty:
        logger.warning(
            f"Train set (Year 2023) size: {train_df.shape[0]}, Test set (Year 2024) size: {test_df.shape[0]}."
        )
        logger.error(
            "Train or Test set is empty with 2023/2024 split. Check data availability for these years."
        )
        raise ValueError("Train or Test set is empty after time-aware split.")

    logger.info(
        f"Time-aware split: Training on Year 2023 ({train_df.shape[0]} samples), Testing on Year 2024 ({test_df.shape[0]} samples)."
    )

    return (
        train_df,
        test_df,
        final_numerical_features,
        final_categorical_features,
        TARGET_COL_TIRE_WEAR,
    )


def create_preprocessing_pipeline_tire_wear(numerical_features):
    """Creates a preprocessing pipeline for numerical features only."""
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
        ],
        remainder="passthrough",  # Pass through non-numerical features
        verbose_feature_names_out=True,  # CRITICAL: This ensures output names are available
    )
    return preprocessor


def train_tire_wear_model_pipeline(
    train_df, test_df, numerical_features, categorical_features, target_col
):
    logger.info("Starting tire wear model hyperparameter tuning...")

    X_train = train_df[numerical_features + categorical_features]
    y_train = train_df[target_col]
    X_test = test_df[numerical_features + categorical_features]
    y_test = test_df[target_col]

    # Create preprocessor
    preprocessor = create_preprocessing_pipeline_tire_wear(numerical_features)

    # CatBoostRegressor initialization
    cat_regressor = CatBoostRegressor(
        objective="MAE",
        random_state=42,
        iterations=2000,
        early_stopping_rounds=100,
        verbose=0,
        loss_function="MAE",
        eval_metric="MAE",
    )

    # Pipeline now includes preprocessor and CatBoost regressor
    model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", cat_regressor)]
    )

    # MODIFIED HYPERPARAMETER DISTRIBUTION FOR CatBoost
    param_distributions = {
        "regressor__iterations": randint(300, 2000),
        "regressor__learning_rate": uniform(0.005, 0.2),
        "regressor__depth": randint(4, 10),
        "regressor__l2_leaf_reg": uniform(0.01, 10.0),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bylevel": uniform(0.6, 0.4),
        "regressor__min_data_in_leaf": randint(1, 100),
    }

    random_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=300,  # <<<--- CHANGE THIS BACK TO YOUR DESIRED NUMBER (e.g., 300)
        cv=5,
        verbose=2,  # <<<--- YOU CAN CHANGE THIS back to 2 for more output during the long run
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    # CRITICAL FIX: Get the feature names AFTER preprocessor is fitted
    # and use those to determine the indices for CatBoost's cat_features.
    # We need to fit the preprocessor ONCE outside of the main RandomizedSearchCV
    # to get the correct transformed feature names/indices.
    # The `RandomizedSearchCV` will then clone this preprocessor and apply it.

    # Create a temporary preprocessor to get feature names after transformation
    temp_preprocessor = create_preprocessing_pipeline_tire_wear(numerical_features)
    temp_preprocessor.fit(X_train)  # Fit it to learn transformations and feature names

    # Get the feature names out from the temporary preprocessor
    post_transform_feature_names = temp_preprocessor.get_feature_names_out()

    # Determine which of these names correspond to our original categorical features.
    # `remainder__` prefix is added by ColumnTransformer with `verbose_feature_names_out=True`.
    cat_feature_indices_for_catboost = []
    for original_cat_col in categorical_features:
        # Check for both original name and prefixed name
        if f"remainder__{original_cat_col}" in post_transform_feature_names:
            cat_feature_indices_for_catboost.append(
                list(post_transform_feature_names).index(
                    f"remainder__{original_cat_col}"
                )
            )
        elif (
            original_cat_col in post_transform_feature_names
        ):  # Fallback if no prefix for some reason
            cat_feature_indices_for_catboost.append(
                list(post_transform_feature_names).index(original_cat_col)
            )

    fit_params = {
        "regressor__cat_features": cat_feature_indices_for_catboost  # Pass indices for robustness
    }

    random_search.fit(X_train, y_train, **fit_params)

    best_model = random_search.best_estimator_

    logger.info(
        f"Best hyperparameters found for Tire Wear model: {random_search.best_params_}"
    )
    with mlflow.start_run(run_name="tire_wear_tuning_run_catboost", nested=False):
        params_to_log = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in random_search.best_params_.items()
        }
        mlflow.log_params(params_to_log)

        # Inverse transform predictions before calculating metrics
        y_pred_transformed = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_transformed)

        # Inverse transform actual values for metric calculation if y_test was transformed
        y_test_original_scale = np.expm1(y_test)

        mae = mean_absolute_error(y_test_original_scale, y_pred)
        r2 = r2_score(y_test_original_scale, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_original_scale, y_pred))

        logger.info(f"Tire Wear Model Evaluation on Test Set (Original Scale):")
        logger.info(f"    MAE: {mae:.4f}")
        logger.info(f"    R2 Score: {r2:.4f}")
        logger.info(f"    RMSE: {rmse:.4f}")

        mlflow.log_metrics({"mae": mae, "r2_score": r2, "rmse": rmse})

    mlflow.end_run()

    return best_model, {"MAE": mae, "R2": r2, "RMSE": rmse}


def save_tire_wear_model(
    model, metrics, X_train_sample, numerical_features, categorical_features
):
    """
    Save the trained tire wear model to disk and log to MLflow.
    """
    new_model_path = MODELS_DIR / "tire_wear_catboost_pipeline_tuned_v1.joblib"
    logger.info(f"Saving tire wear model to {new_model_path}")
    joblib.dump(model, new_model_path)

    with mlflow.start_run(run_name="tire_wear_catboost_v1", nested=False):
        params_to_log = {
            k.replace("regressor__", ""): (v.item() if isinstance(v, np.generic) else v)
            for k, v in model.named_steps["regressor"].get_params().items()
            if isinstance(k, str)
            and not k.startswith("base_")
            and not k.startswith("validate_")
        }

        for name, transformer, _ in model.named_steps["preprocessor"].transformers:
            if hasattr(transformer, "get_params"):
                for param_name, param_value in transformer.get_params().items():
                    params_to_log[f"preprocessor_{name}_{param_name}"] = param_value

        # The post_transform_feature_names_for_logging is needed for the MLflow signature
        # to ensure correct column names are captured.
        temp_preprocessor_for_logging = create_preprocessing_pipeline_tire_wear(
            numerical_features
        )
        # Fit on the *original* X_train_sample to get correct feature names
        temp_preprocessor_for_logging.fit(
            X_train_sample[numerical_features + categorical_features]
        )
        post_transform_feature_names_for_logging = (
            temp_preprocessor_for_logging.get_feature_names_out()
        )

        cat_features_for_mlflow = []
        for original_cat_col in categorical_features:
            if (
                f"remainder__{original_cat_col}"
                in post_transform_feature_names_for_logging
            ):
                cat_features_for_mlflow.append(f"remainder__{original_cat_col}")
            elif original_cat_col in post_transform_feature_names_for_logging:
                cat_features_for_mlflow.append(original_cat_col)

        params_to_log["cat_features_used"] = (
            cat_features_for_mlflow  # Log the names passed to CatBoost
        )

        mlflow.log_params(params_to_log)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Pass the ORIGINAL X_train_sample to model.predict() for signature.
        # The pipeline handles its own transformations internally.
        sample_y_pred_transformed = model.predict(X_train_sample)
        sample_y_pred_original_scale = np.expm1(sample_y_pred_transformed)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="tire_wear_catboost_model_v1",
            registered_model_name="TireWearCatBoostModelV1",
            signature=mlflow.models.infer_signature(
                X_train_sample,  # Pass original X_train_sample here
                sample_y_pred_original_scale,
            ),
            input_example=X_train_sample.head(5),
        )
        mlflow.log_artifact(str(new_model_path), "model_joblib")

    logger.info("Tire wear model saved and logged to MLflow")


def main():
    logger.info("Starting tire wear model training process")
    try:
        (
            train_df,
            test_df,
            numerical_features_final,
            categorical_features_final,
            target_col,
        ) = load_and_prepare_tire_wear_data()

        # --- Visualization of Target Variable ---
        logger.info("Analyzing and visualizing target variable distribution...")

        # Set up plot style
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 6))

        # Plot 1: Distribution of the target variable (transformed scale)
        plt.subplot(1, 2, 1)
        sns.histplot(train_df[target_col], bins=50, kde=True, color="skyblue")
        plt.title(f"Distribution of {target_col} (Training Data - Transformed)")
        plt.xlabel("Log(1 + Degradation per Lap (s))")
        plt.ylabel("Frequency")
        plt.axvline(
            train_df[target_col].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {train_df[target_col].mean():.4f}",
        )
        plt.axvline(
            train_df[target_col].median(),
            color="green",
            linestyle=":",
            label=f"Median: {train_df[target_col].median():.4f}",
        )
        plt.legend()

        # Plot 2: Target vs. LapNumberInStint (Original Scale for interpretation)
        train_df_original_target = train_df.copy()
        train_df_original_target[target_col] = np.expm1(
            train_df_original_target[target_col]
        )

        plt.subplot(1, 2, 2)
        sns.lineplot(
            data=train_df_original_target.sample(
                n=min(len(train_df_original_target), 10000), random_state=42
            ).sort_values("LapNumberInStint"),
            x="LapNumberInStint",
            y=target_col,
            hue="Compound",
            errorbar=("ci", 95),
            palette="viridis",
            linewidth=2,
        )
        plt.title(
            f"Degradation per Lap (s) vs. LapNumberInStint by Compound (Original Scale)"
        )
        plt.xlabel("Lap Number in Stint")
        plt.ylabel("Degradation per Lap (s)")
        plt.ylim(bottom=0)
        plt.legend(title="Compound")

        plt.tight_layout()
        plot_path_dist = MODELS_DIR / "tire_wear_target_distribution_v2_catboost.png"
        plt.savefig(plot_path_dist)
        logger.info(f"Saved target distribution plot to {plot_path_dist}")
        plt.close()

        # X_train_sample for signature needs to include both numerical and categorical features
        X_train_sample_full = train_df[
            numerical_features_final + categorical_features_final
        ].head(5)  # Take a sample before passing to functions

        best_model, metrics = train_tire_wear_model_pipeline(
            train_df,
            test_df,
            numerical_features_final,
            categorical_features_final,
            target_col,
        )
        save_tire_wear_model(
            best_model,
            metrics,
            X_train_sample_full,  # This is the full sample DataFrame
            numerical_features_final,
            categorical_features_final,
        )

        with mlflow.start_run(run_name="log_plots_for_catboost_v1", nested=True):
            mlflow.log_artifact(
                str(plot_path_dist), "target_distribution_plots_v2_catboost"
            )
        mlflow.end_run()

        logger.info("Tire wear model training completed successfully")
    except Exception as e:
        logger.error(f"Error training tire wear model: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
