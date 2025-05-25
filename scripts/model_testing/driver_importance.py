# scripts/model_analysis/analyze_driver_impact.py

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,  # Assuming REPORTS_DIR is defined in your settings
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure reports directory exists
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define paths
model_path = MODELS_DIR / "laptime_xgboost_pipeline_tuned_v2_driver_team.joblib"
data_path = PROCESSED_DATA_DIR / "seasons_2023_2024_data_enhanced.csv"


def load_data_and_model():
    """Load the enhanced data and the trained lap time model."""
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    return df, model


def preprocess_data_for_prediction(df_raw, model):
    """
    Apply the same preprocessing steps as in training to get a consistent dataset
    for prediction and analysis.
    """
    df = df_raw.copy()

    # Convert timedelta strings to total seconds
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
            if pd.api.types.is_timedelta64_dtype(df[col]):
                df[f"{col}_s"] = df[col].dt.total_seconds()
            elif isinstance(df[col].iloc[0], str):
                df[f"{col}_s"] = df[col].apply(
                    lambda x: pd.to_timedelta(x).total_seconds()
                    if pd.notna(x)
                    else np.nan
                )

    # Convert LapStartDate to datetime and sort
    if "LapStartDate" in df.columns:
        df["LapStartDate"] = pd.to_datetime(df["LapStartDate"])
        df = df.sort_values(by="LapStartDate").reset_index(drop=True)
    elif (
        "Time_s" in df.columns
    ):  # Fallback if LapStartDate is not directly available but Time_s is
        df = df.sort_values(by="Time_s").reset_index(drop=True)
    else:
        logger.warning(
            "Neither 'LapStartDate' nor 'Time_s' available for temporal sorting. Analysis might be affected."
        )

    # Filter for clean laps
    if "IsCleanLap" in df.columns:
        df = df[df["IsCleanLap"] == True].copy()
    else:
        df = df[df["Deleted"] == False].copy()
        if "IsOutlap" in df.columns:
            df = df[df["IsOutlap"] == False]
        if "IsInlap" in df.columns:
            df = df[df["IsInlap"] == False]

    # Handle Rainfall (assuming 0 for NaN)
    if "Rainfall" not in df.columns:
        df["Rainfall"] = 0
    else:
        df["Rainfall"] = df["Rainfall"].fillna(0)

    # Define features based on what the model was trained with
    # Get feature names from the model's preprocessor's fit
    # This is a bit tricky as the pipeline's ColumnTransformer stores fitted transformers.
    # We need to know the *original* feature columns fed into the pipeline
    # The 'feature_columns' list from the training script is essential here.
    # For now, let's replicate it directly.

    feature_columns = [
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
        "Event",
        "Driver",
        "Team",
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
        "TyreWearPercentage",
        "WindSpeed_Avg",
        "TrackCondition",
    ]
    target_column = "LapTime_s"

    # Filter df to only include columns used by the model, then drop NaNs
    existing_feature_columns = [col for col in feature_columns if col in df.columns]
    required_columns_for_dropna = existing_feature_columns + [target_column]

    df = df.dropna(subset=required_columns_for_dropna).copy()

    # Ensure categorical columns are string type
    for col in ["Driver", "Team", "Event", "Compound", "Session", "TrackCondition"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Perform the same time-aware split as in training for the final evaluation set
    # Identify unique years and determine split point
    unique_years = sorted(df["Year"].unique()) if "Year" in df.columns else []

    if len(unique_years) >= 2:
        split_year_idx = int(len(unique_years) * 0.8)
        if split_year_idx >= len(unique_years):
            split_year_idx = len(unique_years) - 1
        split_year = unique_years[split_year_idx]

        test_df = df[df["Year"] > split_year].copy()

        if test_df.empty and len(unique_years) > 1:
            logger.warning(
                f"Time-aware test set is empty with split_year={split_year}. Falling back to time-based index split."
            )
            split_idx = int(len(df) * 0.8)
            test_df = df.iloc[split_idx:].copy()
        logger.info(f"Data for analysis (test set): Years > {split_year}")
    else:
        logger.warning(
            "Less than 2 unique years for time-aware split. Performing 80/20 time-based index split for analysis."
        )
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()

    if test_df.empty:
        logger.error(
            "The test set for analysis is empty after preprocessing and splitting."
        )
        return pd.DataFrame(), pd.Series(), []

    X_test = test_df[existing_feature_columns]
    y_test = test_df[target_column]

    logger.info(f"Prepared {X_test.shape[0]} samples for analysis.")
    return X_test, y_test, existing_feature_columns


def analyze_residuals_by_driver(X_test, y_test, model):
    """
    Analyzes and visualizes residuals grouped by driver.
    """
    logger.info("Analyzing residuals by driver...")
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    analysis_df = X_test.copy()
    analysis_df["ActualLapTime_s"] = y_test
    analysis_df["PredictedLapTime_s"] = y_pred
    analysis_df["Residual"] = residuals

    # Sort drivers by their median residual for better visualization
    median_residuals = analysis_df.groupby("Driver")["Residual"].median().sort_values()
    sorted_drivers = median_residuals.index.tolist()

    plt.figure(figsize=(16, 8))
    sns.boxplot(
        data=analysis_df,
        x="Driver",
        y="Residual",
        order=sorted_drivers,
        palette="viridis",
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Zero Residual")
    plt.title("Lap Time Prediction Residuals by Driver (Actual - Predicted)")
    plt.xlabel("Driver")
    plt.ylabel("Residual (Seconds)")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plot_path = FIGURES_DIR / "laptime_residuals_by_driver.png"
    plt.savefig(plot_path)
    logger.info(f"Saved residuals by driver plot to {plot_path}")
    plt.close()

    # Print median and std dev of residuals per driver
    driver_summary = (
        analysis_df.groupby("Driver")["Residual"]
        .agg(["median", "std", "count"])
        .sort_values("median")
    )
    logger.info("\nMedian and Std Dev of Residuals per Driver:")
    logger.info(driver_summary)


def simulate_driver_impact(df_raw, model, feature_columns):
    """
    Simulates the impact of changing drivers on predicted lap times,
    holding other features constant.
    """
    logger.info("Simulating driver impact on lap times...")

    # Select a representative "baseline" lap from the data
    # Choose a common event, dry conditions, mid-stint, etc.
    # Let's try to find a lap from a representative event like 'Silverstone' or 'Bahrain'
    # And a common compound like 'MEDIUM'

    # Filter for a common track and compound
    df_filtered_for_baseline = df_raw[
        (df_raw["Event"] == "British Grand Prix")
        & (df_raw["Compound"] == "MEDIUM")
        & (df_raw["Rainfall"] == 0)
    ].copy()

    if df_filtered_for_baseline.empty:
        df_filtered_for_baseline = df_raw[
            (df_raw["Compound"] == "MEDIUM") & (df_raw["Rainfall"] == 0)
        ].copy()
        if df_filtered_for_baseline.empty:
            logger.warning(
                "Could not find a specific baseline lap. Using first clean, non-NaN lap."
            )
            baseline_row = (
                df_raw.dropna(subset=["LapTime_s"] + feature_columns).iloc[0].copy()
            )
        else:
            baseline_row = (
                df_filtered_for_baseline.dropna(subset=["LapTime_s"] + feature_columns)
                .iloc[0]
                .copy()
            )
    else:
        baseline_row = (
            df_filtered_for_baseline.dropna(subset=["LapTime_s"] + feature_columns)
            .iloc[0]
            .copy()
        )

    # Get all unique drivers
    all_drivers = df_raw["Driver"].unique()
    all_drivers = [d for d in all_drivers if pd.notna(d)]  # Filter out potential NaNs

    predicted_lap_times = []
    drivers = []

    for driver in all_drivers:
        # Create a copy of the baseline row and modify only the driver
        sim_data = pd.DataFrame(
            [baseline_row[feature_columns].values], columns=feature_columns
        )
        sim_data["Driver"] = driver

        # Ensure other categorical columns are consistent
        for col in ["Team", "Event", "Compound", "Session", "TrackCondition"]:
            if col in feature_columns and col not in sim_data.columns:
                sim_data[col] = baseline_row[col]
            elif col in sim_data.columns:
                sim_data[col] = sim_data[col].astype(str)  # Ensure string type

        try:
            pred_lap_time = model.predict(sim_data)[0]
            predicted_lap_times.append(pred_lap_time)
            drivers.append(driver)
        except Exception as e:
            logger.warning(f"Could not predict for driver {driver}: {e}")
            continue

    if not drivers:
        logger.error("No predictions could be made for driver simulation.")
        return

    sim_results_df = (
        pd.DataFrame({"Driver": drivers, "PredictedLapTime_s": predicted_lap_times})
        .sort_values(by="PredictedLapTime_s")
        .reset_index(drop=True)
    )

    # Calculate difference from the fastest driver
    sim_results_df["DeltaFromFastest_s"] = (
        sim_results_df["PredictedLapTime_s"]
        - sim_results_df["PredictedLapTime_s"].min()
    )

    logger.info("\nSimulated Lap Times by Driver (holding other factors constant):")
    logger.info(sim_results_df)

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="Driver", y="PredictedLapTime_s", data=sim_results_df, palette="viridis"
    )
    plt.title(
        f"Simulated Lap Times by Driver (Baseline: {baseline_row['Event']}, {baseline_row['Compound']}, Dry)"
    )
    plt.xlabel("Driver")
    plt.ylabel("Predicted Lap Time (seconds)")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_path_sim_abs = FIGURES_DIR / "laptime_simulated_by_driver_absolute.png"
    plt.savefig(plot_path_sim_abs)
    logger.info(f"Saved simulated absolute lap times plot to {plot_path_sim_abs}")
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="Driver", y="DeltaFromFastest_s", data=sim_results_df, palette="cividis"
    )
    plt.title(
        f"Simulated Delta from Fastest Driver (Baseline: {baseline_row['Event']}, {baseline_row['Compound']}, Dry)"
    )
    plt.xlabel("Driver")
    plt.ylabel("Delta from Fastest (seconds)")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_path_sim_delta = FIGURES_DIR / "laptime_simulated_by_driver_delta.png"
    plt.savefig(plot_path_sim_delta)
    logger.info(f"Saved simulated delta lap times plot to {plot_path_sim_delta}")
    plt.close()


def main():
    logger.info("Starting driver impact analysis...")
    df_raw, model = load_data_and_model()

    # Prepare the test set for analysis
    X_test, y_test, feature_cols = preprocess_data_for_prediction(df_raw, model)

    if X_test.empty or y_test.empty:
        logger.error("Skipping analysis due to empty test set.")
        return

    # Analyze residuals by driver
    analyze_residuals_by_driver(X_test, y_test, model)

    # Simulate driver impact
    simulate_driver_impact(
        df_raw, model, feature_cols
    )  # Pass df_raw for baseline selection

    logger.info("Driver impact analysis completed.")


if __name__ == "__main__":
    main()
