import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Attempt to import XGBoost and ColumnTransformer, handle if not available in every environment
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("WARNING: xgboost library not found. XGBoost model analysis will be limited.")

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
    print(
        "WARNING: ColumnTransformer not found. Automatic feature name extraction might be affected."
    )

# --- Configuration ---
# Adjust these paths to point to your model files and project root if necessary
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TIRE_WEAR_MODEL_FILENAME = "tire_wear_xgboost_pipeline_tuned_v1.joblib"
# Add other model filenames here if you want to analyze them
LAP_TIME_MODEL_FILENAME = "laptime_xgboost_pipeline_tuned_v2_driver_team.joblib"


def get_feature_names_from_pipeline(pipeline, xgb_step_name):
    """Attempts to extract feature names from a scikit-learn pipeline,
    especially if it contains a ColumnTransformer.

    Args:
        pipeline: The fitted scikit-learn pipeline.
        xgb_step_name: The name of the XGBoost model step in the pipeline.

    Returns:
        A list of feature names, or None if they cannot be extracted.
    """
    if not ColumnTransformer:
        print("DEBUG: ColumnTransformer not available for feature name extraction.")
        return None

    preprocessor = None
    # Try to find a ColumnTransformer step before the XGBoost model step
    for i, (step_name, step_estimator) in enumerate(pipeline.steps):
        if step_name == xgb_step_name:
            if i > 0:  # Check if there's a step before the model
                potential_preprocessor_step = pipeline.steps[i - 1][1]
                if isinstance(potential_preprocessor_step, ColumnTransformer):
                    preprocessor = potential_preprocessor_step
                    print(f"DEBUG: Found ColumnTransformer: {pipeline.steps[i - 1][0]}")
                    break
                # Add checks for other types of preprocessors if necessary
            else:  # Model is the first step, unlikely if there's preprocessing
                print(
                    "DEBUG: Model is the first step. Assuming features are directly input."
                )
                return None  # Or try to get from model.feature_names_in_ if available

    if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return preprocessor.get_feature_names_out()
        except Exception as e:
            print(f"ERROR: Could not get feature names from ColumnTransformer: {e}")
            print(
                "INFO: This might happen if the ColumnTransformer contains transformers"
            )
            print(
                "      that don't support get_feature_names_out (e.g., custom functions without it),"
            )
            print(
                "      or if it wasn't fitted with DataFrame input with column names."
            )
            return None
    elif hasattr(pipeline.named_steps[xgb_step_name], "feature_names_in_"):
        # Fallback if model itself has feature names (e.g. if pipeline starts with model and features were named DFs)
        print("DEBUG: Using 'feature_names_in_' from the model step.")
        return list(pipeline.named_steps[xgb_step_name].feature_names_in_)
    else:
        print(
            "DEBUG: Preprocessor step not found or does not support get_feature_names_out()."
        )
        print(
            "INFO: Feature names will not be available for the plot unless manually provided."
        )
        return None


def analyze_xgboost_model(model_path, model_name="XGBoost Model"):
    """Loads an XGBoost model (or a pipeline containing it) and analyzes its feature importance."""
    if not xgb:
        print(
            f"Skipping XGBoost analysis for {model_name}: xgboost library not installed."
        )
        return

    print(f"--- Analyzing {model_name} from {model_path} ---")
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}: {e}")
        return

    xgboost_model = None
    xgb_step_name = None

    if hasattr(pipeline, "steps"):  # It's a scikit-learn Pipeline
        print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")
        for step_name, step_estimator in pipeline.steps:
            if isinstance(step_estimator, xgb.XGBRegressor) or isinstance(
                step_estimator, xgb.XGBClassifier
            ):
                xgboost_model = step_estimator
                xgb_step_name = step_name
                print(f"Found XGBoost model step: '{xgb_step_name}'")
                break
        if not xgboost_model:
            print("ERROR: XGBoost model not found within the pipeline steps.")
            return
    elif isinstance(pipeline, xgb.XGBRegressor) or isinstance(
        pipeline, xgb.XGBClassifier
    ):  # It's a raw XGBoost model
        xgboost_model = pipeline
        print("Loaded a raw XGBoost model (not a pipeline).")
    else:
        print(
            "ERROR: Loaded object is not a scikit-learn pipeline or an XGBoost model."
        )
        return

    # Get feature names
    feature_names = None
    if (
        hasattr(pipeline, "steps") and xgb_step_name
    ):  # Only try to get from pipeline if it is one
        feature_names = get_feature_names_from_pipeline(pipeline, xgb_step_name)
    elif hasattr(
        xgboost_model, "feature_names_in_"
    ):  # For raw models fitted with DataFrames
        feature_names = list(xgboost_model.feature_names_in_)

    if hasattr(xgboost_model, "feature_importances_"):
        importances = xgboost_model.feature_importances_

        if feature_names is not None and len(feature_names) == len(importances):
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            )
            importance_df = importance_df.sort_values(
                by="importance", ascending=False
            ).reset_index(drop=True)

            print(f"\nTop 40 Features for {model_name}:")
            print(importance_df.head(40))

            # Plotting
            plt.figure(figsize=(12, max(8, len(importance_df) // 2)))
            plt.title(f"{model_name} - Feature Importance")
            bars = plt.barh(
                importance_df["feature"], importance_df["importance"], color="skyblue"
            )
            plt.gca().invert_yaxis()  # Display top features at the top
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()

            # Add values on bars
            for bar in bars:
                plt.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                )

            plot_filename = (
                f"{model_name.replace(' ', '_').lower()}_feature_importance.png"
            )
            # Ensure the reports/figures directory exists before saving
            figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, plot_filename))
            print(
                f"\nSaved feature importance plot to: {os.path.join(figures_dir, plot_filename)}"
            )
            # plt.show() # Uncomment to display plot if running interactively

        else:
            print("\nFeature Importances (indices):")
            print(importances)
            print(
                f"Warning: Could not map all importances to feature names for {model_name}."
            )
            print(
                f"Number of feature names found: {len(feature_names) if feature_names else 0}"
            )
            print(f"Number of importances: {len(importances)}")
            print("Plotting importance by feature index.")
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                xgb.plot_importance(
                    xgboost_model,
                    ax=ax,
                    max_num_features=30,
                    height=0.8,
                    title=f"{model_name} - Feature Importance (by index)",
                )
                plot_filename = f"{model_name.replace(' ', '_').lower()}_feature_importance_indexed.png"
                # Ensure the reports/figures directory exists before saving
                figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
                os.makedirs(figures_dir, exist_ok=True)
                plt.savefig(os.path.join(figures_dir, plot_filename))
                print(
                    f"\nSaved indexed feature importance plot to: {os.path.join(figures_dir, plot_filename)}"
                )
                # plt.show() # Uncomment to display plot
            except Exception as e_plot:
                print(f"Could not generate indexed plot: {e_plot}")
    else:
        print(
            f"The model object for {model_name} does not have 'feature_importances_'."
        )


def main():
    print("Starting Feature Importance Analysis Script...")

    # Ensure reports/figures directory exists (already handled in analyze_xgboost_model, but good practice)
    figures_path = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_path, exist_ok=True)

    # --- Analyze Tire Wear Model ---
    tire_wear_model_full_path = os.path.join(MODELS_DIR, TIRE_WEAR_MODEL_FILENAME)
    if os.path.exists(tire_wear_model_full_path):
        analyze_xgboost_model(
            tire_wear_model_full_path, model_name="Tire Wear XGBoost Model"
        )
    else:
        print(f"Tire wear model not found at: {tire_wear_model_full_path}")

    # --- Analyze Lap Time Model (Example - Uncomment and adapt) ---
    lap_time_model_full_path = os.path.join(MODELS_DIR, LAP_TIME_MODEL_FILENAME)
    if os.path.exists(lap_time_model_full_path):
        analyze_xgboost_model(
            lap_time_model_full_path, model_name="Lap Time XGBoost Model"
        )
    else:
        print(f"Lap time model not found at: {lap_time_model_full_path}")

    print("\n--- Analysis Script Finished ---")
    print("NOTE: If feature names are incorrect or missing, you may need to:")
    print(
        "1. Ensure your pipeline's ColumnTransformer (or other preprocessor) was fitted with named pandas DataFrames."
    )
    print(
        "2. Verify that all transformers in your ColumnTransformer support 'get_feature_names_out'."
    )
    print(
        "3. Or, manually provide the list of feature names in the script if automatic extraction fails."
    )
    print(
        "   (This list should match the order and number of features seen by the XGBoost model itself)."
    )


if __name__ == "__main__":
    main()
