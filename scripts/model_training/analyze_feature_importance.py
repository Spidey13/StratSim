"""
Script to analyze feature importance from the trained lap time prediction model.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR


def load_model_and_data():
    """
    Load the trained model and a sample of data.

    Returns:
        model: Trained model
        feature_names: List of feature names
    """
    # Load model
    model_path = MODELS_DIR / "laptime_xgboost_pipeline_tuned_v1.joblib"
    model = joblib.load(model_path)

    # Load a sample of data to get feature names
    data_path = PROCESSED_DATA_DIR / "seasons_2023_2024_data_enhanced.csv"
    df = pd.read_csv(data_path, nrows=1000)

    # Get feature names
    feature_columns = [
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
        "Event",
    ]

    return model, feature_columns


def analyze_feature_importance(model, feature_columns):
    """
    Analyze and plot feature importance from the model.

    Args:
        model: Trained pipeline model
        feature_columns: List of feature column names
    """
    # Extract feature names after preprocessing
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["regressor"]

    # Get feature names after one-hot encoding
    numeric_features = [
        "TyreLife",
        "CompoundHardness",
        "TrackTemp_Avg",
        "AirTemp_Avg",
        "Humidity_Avg",
        "Rainfall",
    ]
    categorical_features = ["Event"]

    # Get the one-hot encoder from the column transformer
    ohe = preprocessor.named_transformers_["cat"]

    # Get a sample of data to extract one-hot encoded feature names
    sample_data = pd.read_csv(
        PROCESSED_DATA_DIR / "seasons_2023_2024_data_enhanced.csv", nrows=1000
    )
    events = sample_data["Event"].unique()

    # Try to extract categories from one-hot encoder
    try:
        encoded_event_features = [f"Event_{c}" for c in ohe.categories_[0]]
    except:
        # Fallback if we can't get categories from encoder
        encoded_event_features = [f"Event_{c}" for c in events]

    # Combine all feature names
    all_features = numeric_features + encoded_event_features

    # Get feature importance from XGBoost model
    importance_scores = xgb_model.feature_importances_

    # If number of features doesn't match, just use importance scores
    if len(all_features) != len(importance_scores):
        all_features = [f"Feature_{i}" for i in range(len(importance_scores))]

    # Create a dataframe of feature importance
    importance_df = pd.DataFrame(
        {
            "Feature": all_features[: len(importance_scores)],
            "Importance": importance_scores,
        }
    )

    # Sort by importance
    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)

    # Display top 20 features
    print("Top features by importance:")
    print(importance_df.head(20))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
    plt.title("Top 20 Features by Importance")
    plt.tight_layout()

    # Save the plot
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "feature_importance.png")

    print(f"Feature importance plot saved to {output_dir / 'feature_importance.png'}")

    # Also save as CSV
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"Feature importance data saved to {output_dir / 'feature_importance.csv'}")


def main():
    """Main function to analyze feature importance."""
    try:
        model, feature_columns = load_model_and_data()
        analyze_feature_importance(model, feature_columns)
        print("Feature importance analysis completed successfully")
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")
        raise


if __name__ == "__main__":
    main()
