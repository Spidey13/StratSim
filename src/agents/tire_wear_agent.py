"""
Tire wear agent that predicts tire degradation per lap using a trained model.
"""

from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import joblib
import os

from .base_agent import BaseAgent
from src.config.settings import MODELS_DIR

# logging.basicConfig(level=logging.DEBUG) # Changed from INFO to DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO

# Define the default model path relative to the project root
DEFAULT_TIRE_WEAR_MODEL_PATH = os.path.join(
    MODELS_DIR, "tire_wear_catboost_pipeline_tuned_v1.joblib"
)

# Define features - these should match those used in training
# From scripts/model_training/train_tire_wear_model.py
NUMERICAL_FEATURES = [
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
CATEGORICAL_FEATURES = ["Compound", "Event", "Driver", "Team", "Year"]
ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class TireWearAgent(BaseAgent):
    """
    Agent responsible for predicting tire wear using a trained CatBoost model.
    """

    def __init__(
        self,
        name: str = "TireWearAgent",
        model_path: Optional[str] = DEFAULT_TIRE_WEAR_MODEL_PATH,
    ):
        super().__init__(name)
        self.model_path = (
            model_path if model_path is not None else DEFAULT_TIRE_WEAR_MODEL_PATH
        )
        self.model = self._load_model(self.model_path)
        if not self.model:
            logger.error(
                f"CRITICAL: Tire wear model failed to load from {self.model_path}. Agent will not predict accurately."
            )
        # Heuristic fallback parameters (can be used if model fails catastrophically)
        self.base_wear_rate_s = {
            "SOFT": 0.08,
            "MEDIUM": 0.06,
            "HARD": 0.04,
            "INTERMEDIATE": 0.10,
            "WET": 0.12,
        }

    def _load_model(self, model_path: str):
        try:
            logger.info(f"Loading tire wear model from {model_path}")
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load tire wear model from {model_path}: {str(e)}")
            return None

    def _calculate_interaction_features(self, inputs_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates necessary interaction features and type conversions."""
        # Create compound indicator columns
        inputs_df["IsSoft"] = (inputs_df["Compound"] == "SOFT").astype(float)
        inputs_df["IsMedium"] = (inputs_df["Compound"] == "MEDIUM").astype(float)
        inputs_df["IsHard"] = (inputs_df["Compound"] == "HARD").astype(float)

        # Calculate tire life polynomial features
        inputs_df["TyreLifeSquared"] = inputs_df["LapNumberInStint"] ** 2
        inputs_df["TyreLifeCubed"] = inputs_df["LapNumberInStint"] ** 3

        # Calculate interaction terms
        inputs_df["LapNumberInStint_TrackTemp_Interaction"] = (
            inputs_df["LapNumberInStint"] * inputs_df["TrackTemp_Avg"]
        )
        inputs_df["LapNumberInStint_IsSoft_Interaction"] = (
            inputs_df["LapNumberInStint"] * inputs_df["IsSoft"]
        )
        inputs_df["LapNumberInStint_IsMedium_Interaction"] = (
            inputs_df["LapNumberInStint"] * inputs_df["IsMedium"]
        )
        inputs_df["LapNumberInStint_IsHard_Interaction"] = (
            inputs_df["LapNumberInStint"] * inputs_df["IsHard"]
        )
        inputs_df["TrackTemp_Avg_IsSoft_Interaction"] = (
            inputs_df["TrackTemp_Avg"] * inputs_df["IsSoft"]
        )
        inputs_df["TrackTemp_Avg_IsMedium_Interaction"] = (
            inputs_df["TrackTemp_Avg"] * inputs_df["IsMedium"]
        )
        inputs_df["TrackTemp_Avg_IsHard_Interaction"] = (
            inputs_df["TrackTemp_Avg"] * inputs_df["IsHard"]
        )

        # Ensure all numeric columns are float
        for col in NUMERICAL_FEATURES:
            if col in inputs_df.columns:
                inputs_df[col] = pd.to_numeric(inputs_df[col], errors="coerce").astype(
                    float
                )

        # Fill any NaN values with appropriate defaults
        default_values = {
            "TyreLifeSquared": 0.0,
            "TyreLifeCubed": 0.0,
            "LapNumberInStint_TrackTemp_Interaction": 0.0,
            "LapNumberInStint_IsSoft_Interaction": 0.0,
            "LapNumberInStint_IsMedium_Interaction": 0.0,
            "LapNumberInStint_IsHard_Interaction": 0.0,
            "TrackTemp_Avg_IsSoft_Interaction": 0.0,
            "TrackTemp_Avg_IsMedium_Interaction": 0.0,
            "TrackTemp_Avg_IsHard_Interaction": 0.0,
        }
        inputs_df = inputs_df.fillna(default_values)

        return inputs_df

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes inputs to estimate tire degradation for the current lap using the model.

        Args:
            inputs: Dictionary containing:
                - Compound: Current tire compound (e.g., "SOFT", "MEDIUM")
                - LapNumberInStint: Current lap count on this set of tires
                - Event: Circuit ID (e.g., "BAHRAIN")
                - Driver: Driver ID (used for aggressiveness, simplified)
                - weather: Dictionary with weather info (e.g., {"rainfall": 0.0})
                - SpeedST, SpeedI1, SpeedI2: Section speeds (simplified use)
                - PrevLapTimeDegradation_s: Degradation from previous lap (not directly used here for current lap calc, but for context)

        Returns:
            Dictionary containing:
                - estimated_degradation_per_lap_s: Estimated lap time degradation for this lap (in seconds)
                - optimal_window: Boolean indicating if tires are in optimal window
                - wear_factors: Dictionary of factors applied
        """
        if not self.model:
            logger.warning("Tire wear model not loaded. Using heuristic fallback.")
            # Simplified heuristic fallback
            compound = inputs.get("Compound", "MEDIUM").upper()
            base_wear = self.base_wear_rate_s.get(
                compound, self.base_wear_rate_s["MEDIUM"]
            )
            return {
                "estimated_degradation_per_lap_s": base_wear,
                "optimal_window": True,  # Simplified
                "wear_factors": {"reason": "fallback_no_model"},
            }

        # Prepare features for the model
        # The input keys from TireManagerAgent might need some mapping to match training feature names
        feature_dict = {
            "Compound": inputs.get("Compound", "MEDIUM"),
            "LapNumberInStint": inputs.get("LapNumberInStint", 1),
            "Event": inputs.get("Event", "unknown"),  # Circuit ID
            "Driver": inputs.get("Driver", "unknown"),
            "Team": inputs.get("Team", "unknown"),
            "Year": inputs.get("Year", 2024),  # Assuming a default year
            "TrackTemp_Avg": inputs.get("weather", {}).get("track_temp", 30),
            "AirTemp_Avg": inputs.get("weather", {}).get("air_temp", 25),
            "Humidity_Avg": inputs.get("weather", {}).get("humidity", 50),
            "Rainfall": inputs.get("weather", {}).get("rainfall", 0),
            "SpeedST": inputs.get("SpeedST", 280.0),
            "SpeedI1": inputs.get("SpeedI1", 200.0),
            "SpeedI2": inputs.get("SpeedI2", 150.0),
            "PrevLapTimeDegradation_s": inputs.get("PrevLapTimeDegradation_s", 0.0),
        }

        features_df = pd.DataFrame([feature_dict])
        features_df = self._calculate_interaction_features(features_df)

        logger.debug(f"TireWearAgent: Constructed feature_dict: {feature_dict}")
        logger.debug(
            f"TireWearAgent: DataFrame after _calculate_interaction_features (head):\n{features_df.head().to_string()}"
        )

        # Ensure all required columns are present for the model, in the correct order
        missing_cols = [
            col for col in ALL_MODEL_FEATURES if col not in features_df.columns
        ]
        if missing_cols:
            logger.error(
                f"TireWearAgent: Missing columns for model prediction: {missing_cols}. Available: {features_df.columns.tolist()}"
            )
            # Fallback if essential features are missing
            compound = inputs.get("Compound", "MEDIUM").upper()
            base_wear = self.base_wear_rate_s.get(
                compound, self.base_wear_rate_s["MEDIUM"]
            )
            return {
                "estimated_degradation_per_lap_s": base_wear,
                "optimal_window": True,
                "wear_factors": {
                    "reason": "fallback_missing_features",
                    "missing": missing_cols,
                },
            }

        features_df_ordered = features_df[ALL_MODEL_FEATURES]
        logger.debug(
            f"TireWearAgent: Final features_df_ordered for model (head):\n{features_df_ordered.head().to_string()}"
        )

        try:
            prediction_transformed = self.model.predict(features_df_ordered)[0]
            logger.debug(
                f"TWA Raw Model Pred: {prediction_transformed} for compound {feature_dict['Compound']} LapInStint {feature_dict['LapNumberInStint']}"
            )
            estimated_degradation_per_lap_s = np.expm1(prediction_transformed)
            logger.debug(
                f"TWA After Expm1: {estimated_degradation_per_lap_s} for compound {feature_dict['Compound']} LapInStint {feature_dict['LapNumberInStint']}"
            )

            # Clip predictions to a realistic range (e.g., 0 to 1.0 seconds of degradation per lap)
            estimated_degradation_per_lap_s = np.clip(
                estimated_degradation_per_lap_s, 0, 1.0
            )
            logger.debug(
                f"TireWearAgent: After clipping: {estimated_degradation_per_lap_s}"
            )

        except Exception as e:
            logger.error(
                f"Tire wear model prediction failed: {str(e)}. Falling back to heuristic."
            )
            compound = inputs.get("Compound", "MEDIUM").upper()
            base_wear = self.base_wear_rate_s.get(
                compound, self.base_wear_rate_s["MEDIUM"]
            )
            estimated_degradation_per_lap_s = base_wear

        # Determine optimal window (can be a simple heuristic or enhanced)
        lap_number_in_stint = inputs.get("LapNumberInStint", 1)
        optimal_window = 3 <= lap_number_in_stint <= 15

        # logger.info( # Commenting out verbose INFO log
        #     f"TireWearAgent (Model): Compound={compound}, Lap={lap_number_in_stint}, "
        #     f"RawPred={prediction_transformed:.4f}, Degradation={estimated_degradation_per_lap_s:.4f}s, Optimal={optimal_window}"
        # )

        return {
            "estimated_degradation_per_lap_s": estimated_degradation_per_lap_s,
            "optimal_window": optimal_window,  # This can be refined
            "wear_factors": {
                "model_used": True,
                "raw_prediction": prediction_transformed
                if "prediction_transformed" in locals()
                else None,
            },
        }
