# src/agents/lap_time_agent.py

import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

from .base_agent import BaseAgent

from src.config.settings import MODELS_DIR


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the default model path relative to the project root
DEFAULT_MODEL_PATH = os.path.join(
    MODELS_DIR, "laptime_xgboost_pipeline_tuned_v2_driver_team.joblib"
)


class LapTimeAgent(BaseAgent):
    """
    Agent responsible for predicting lap times based on various factors
    including tire wear, weather conditions, and track characteristics.
    """

    def __init__(
        self,
        name: str = "LapTimeAgent",
        model_path: Optional[str] = str(DEFAULT_MODEL_PATH),
        debug_mode: bool = True,
    ):
        """
        Initialize the lap time prediction agent.

        Args:
            name: Agent name
            model_path: Path to the trained lap time prediction model.
                        Defaults to the standard tuned model path.
            debug_mode: Whether to enable debugging capabilities
        """
        super().__init__(name)
        self.model_path = model_path
        self.model = self._load_model(self.model_path) if self.model_path else None
        self.debug_mode = debug_mode
        self.prediction_history = []
        self.feature_importance = {}

        # Create debug log directory
        self.debug_dir = Path(__file__).parent.parent.parent / "logs" / "lap_time_agent"
        if debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Initialize debug metrics
        self.metrics = {
            "total_predictions": 0,
            "model_predictions": 0,
            "fallback_predictions": 0,
            "prediction_range": {"min": float("inf"), "max": float("-inf")},
            "feature_stats": {},
            "errors": [],
        }

        # UPDATED features list based on model error and intended inputs
        self.features = [
            "TyreLife",  # Tire age
            "CompoundHardness",  # Tire compound hardness
            "Event",  # Circuit/Event
            "Driver",  # Driver name
            "Team",  # Team name
            # Speed and performance features
            "SpeedST",
            "SpeedI1",
            "SpeedI2",
            "SpeedFL",
            "SpeedST_Diff",
            "SpeedI1_Diff",
            "SpeedI2_Diff",
            "SpeedFL_Diff",
            # Tire performance metrics
            "GripLevel",
            "DegradationPerLap_s",
            "TireWearPercentage",
            # Track condition (simplified)
            "TrackCondition",
            # Required weather features (model dependency)
            "TrackTemp_Avg",
            "AirTemp_Avg",
            "Humidity_Avg",
            "Rainfall",
            "WetTrack",
            "WeatherStability",
            "WindSpeed_Avg",
            "TempDelta",
        ]

    def _load_model(self, model_path: str):
        """
        Load the trained lap time prediction model.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object
        """
        try:
            logger.info(f"Loading lap time model from {model_path}")
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None

    def _validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data and log any issues.

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        required_fields = ["compound", "tire_age"]

        # Check required fields
        for field in required_fields:
            if field not in inputs:
                warnings.append(f"Missing required field: {field}")

        # Validate numerical ranges
        if "tire_age" in inputs and (
            inputs["tire_age"] < 0 or inputs["tire_age"] > 100
        ):
            warnings.append(f"Suspicious tire age value: {inputs['tire_age']}")

        if "grip_level" in inputs and (
            inputs["grip_level"] < 0 or inputs["grip_level"] > 1
        ):
            warnings.append(f"Invalid grip level: {inputs['grip_level']}")

        # Validate speed inputs
        speed_fields = ["SpeedST", "SpeedI1", "SpeedI2", "SpeedFL"]
        for field in speed_fields:
            if field in inputs and (
                not isinstance(inputs[field], (int, float)) or inputs[field] < 0
            ):
                warnings.append(f"Invalid speed value for {field}: {inputs[field]}")

        # Validate tire performance metrics
        if "tire_wear_percentage" in inputs and (
            inputs["tire_wear_percentage"] < 0 or inputs["tire_wear_percentage"] > 100
        ):
            warnings.append(
                f"Invalid tire wear percentage: {inputs['tire_wear_percentage']}"
            )

        if "degradation_per_lap_s" in inputs and inputs["degradation_per_lap_s"] < 0:
            warnings.append(
                f"Invalid degradation per lap: {inputs['degradation_per_lap_s']}"
            )

        return len(warnings) == 0, warnings

    def _log_prediction(
        self, inputs: Dict[str, Any], prediction: float, used_model: bool
    ):
        """Log prediction details for debugging."""
        if not self.debug_mode:
            return

        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "inputs": {
                k: str(v) if isinstance(v, (dict, list)) else v
                for k, v in inputs.items()
            },
            "prediction": prediction,
            "used_model": used_model,
        }

        # Update metrics
        self.metrics["total_predictions"] += 1
        if used_model:
            self.metrics["model_predictions"] += 1
        else:
            self.metrics["fallback_predictions"] += 1

        self.metrics["prediction_range"]["min"] = min(
            self.metrics["prediction_range"]["min"], prediction
        )
        self.metrics["prediction_range"]["max"] = max(
            self.metrics["prediction_range"]["max"], prediction
        )

        # Log to file
        log_file = (
            self.debug_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Keep prediction history (last 1000 predictions)
        self.prediction_history.append(log_entry)
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)

    def _analyze_prediction(
        self, prediction: float, inputs: Dict[str, Any]
    ) -> List[str]:
        """Analyze prediction for potential issues."""
        warnings = []

        # Check for unrealistic lap times
        if prediction < 70:
            warnings.append(f"Unusually fast lap time: {prediction:.2f}s")
        elif prediction > 120:
            warnings.append(f"Unusually slow lap time: {prediction:.2f}s")

        # Check for extreme conditions with tire wear
        if inputs.get("tire_age", 0) > 30 and prediction < 85:
            warnings.append(
                f"Suspiciously fast lap time ({prediction:.2f}s) with old tires"
            )

        # Check tire performance metrics
        tire_wear = inputs.get("tire_wear_percentage", 0)
        if tire_wear > 70 and prediction < 85:
            warnings.append(
                f"Suspiciously fast lap time ({prediction:.2f}s) with high tire wear ({tire_wear}%)"
            )

        grip_level = inputs.get("grip_level", 1.0)
        if grip_level < 0.8 and prediction < 85:
            warnings.append(
                f"Suspiciously fast lap time ({prediction:.2f}s) with low grip ({grip_level})"
            )

        return warnings

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to predict lap times with enhanced debugging.

        Args:
            inputs: Dictionary containing:
                - circuit_id: ID of the circuit
                - compound: Tire compound
                - tire_age: Age of tires in laps
                - weather: Weather conditions
                - driver: Driver ID (e.g., 'HAM', 'VER')
                - event: Event/Grand Prix name (e.g., 'Monaco Grand Prix')
                - driver_name: Name of the driver (e.g., 'Lewis Hamilton')
                - team_name: Name of the team (e.g., 'Mercedes')
                - degradation_per_lap_s: Estimated lap time degradation due to wear for this lap (in seconds)
                - tire_wear_percentage: Cumulative tire wear (0-100%)
                - grip_level: Current grip level (0.0-1.0)

        Returns:
            Dictionary containing:
                - predicted_laptime: Predicted lap time in seconds
                - confidence: Confidence level of prediction
                - factors: Factors influencing the prediction
                - warnings: List of warnings related to the prediction
                - used_model: Whether the model was used for the prediction
        """
        # Validate inputs
        is_valid, warnings = self._validate_inputs(inputs)
        if not is_valid:
            logger.warning("Input validation warnings: %s", warnings)

        try:
            # Attempt model prediction
            if self.model:
                features = self._prepare_features(inputs)
                predicted_time = self._model_predict(features)
                used_model = True
            else:
                predicted_time = self._fallback_prediction(
                    inputs,
                    inputs.get("driver_name", "Unknown"),
                    inputs.get("team_name", "Unknown"),
                    inputs.get("degradation_per_lap_s", 0.0),
                    inputs.get("tire_wear_percentage", 0.0),
                    inputs.get("grip_level", 1.0),
                )
                used_model = False

            # Analyze prediction
            prediction_warnings = self._analyze_prediction(predicted_time, inputs)
            if prediction_warnings:
                logger.warning("Prediction warnings: %s", prediction_warnings)

            # Log prediction
            self._log_prediction(inputs, predicted_time, used_model)

            # Determine factors
            factors = self._determine_factors(
                inputs,
                predicted_time,
                inputs.get("driver_name", "Unknown"),
                inputs.get("team_name", "Unknown"),
                inputs.get("degradation_per_lap_s", 0.0),
                inputs.get("tire_wear_percentage", 0.0),
                inputs.get("grip_level", 1.0),
            )

            return {
                "predicted_laptime": predicted_time,
                "confidence": 0.95 if used_model else 0.7,
                "factors": factors,
                "warnings": warnings + prediction_warnings,
                "used_model": used_model,
            }

        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            self.metrics["errors"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "inputs": str(inputs),
                }
            )
            raise

    def get_debug_stats(self) -> Dict[str, Any]:
        """Get debugging statistics."""
        if not self.debug_mode:
            return {}

        return {
            "metrics": self.metrics,
            "recent_predictions": self.prediction_history[-10:],  # Last 10 predictions
            "feature_importance": self.feature_importance,
        }

    def _update_feature_stats(self, features: pd.DataFrame):
        """Update running statistics for feature values."""
        if not self.debug_mode:
            return

        for column in features.columns:
            if column not in self.metrics["feature_stats"]:
                self.metrics["feature_stats"][column] = {
                    "min": float("inf"),
                    "max": float("-inf"),
                    "sum": 0,
                    "count": 0,
                }

            stats = self.metrics["feature_stats"][column]
            values = features[column].values

            stats["min"] = min(stats["min"], float(values.min()))
            stats["max"] = max(stats["max"], float(values.max()))
            stats["sum"] += float(values.sum())
            stats["count"] += len(values)

    def _prepare_features(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for model input.

        Args:
            inputs: Input dictionary

        Returns:
            DataFrame with prepared features
        """
        # Create a DataFrame with a single row
        features = pd.DataFrame(
            {
                "TyreLife": [inputs.get("tire_age", 0)],
                "CompoundHardness": [
                    self._get_compound_hardness(inputs.get("compound", "MEDIUM"))
                ],
                "Event": [
                    inputs.get(
                        "event", inputs.get("circuit_id", "unknown") + " Grand Prix"
                    )
                ],
                "Driver": [inputs.get("driver_name", "Unknown Driver")],
                "Team": [inputs.get("team_name", "Unknown Team")],
                # Speed and performance features
                "SpeedST": [inputs.get("SpeedST", 0.0)],
                "SpeedI1": [inputs.get("SpeedI1", 0.0)],
                "SpeedI2": [inputs.get("SpeedI2", 0.0)],
                "SpeedFL": [inputs.get("SpeedFL", 0.0)],
                "SpeedST_Diff": [inputs.get("SpeedST_Diff", 0.0)],
                "SpeedI1_Diff": [inputs.get("SpeedI1_Diff", 0.0)],
                "SpeedI2_Diff": [inputs.get("SpeedI2_Diff", 0.0)],
                "SpeedFL_Diff": [inputs.get("SpeedFL_Diff", 0.0)],
                # Tire performance metrics
                "GripLevel": [inputs.get("grip_level", 1.0)],
                "DegradationPerLap_s": [inputs.get("degradation_per_lap_s", 0.0)],
                "TireWearPercentage": [inputs.get("tire_wear_percentage", 0.0)],
                # Track condition (simplified)
                "TrackCondition": [inputs.get("TrackCondition", 0)],
                # Required weather features with neutral default values
                "TrackTemp_Avg": [30.0],  # Neutral track temperature
                "AirTemp_Avg": [25.0],  # Neutral air temperature
                "Humidity_Avg": [50.0],  # Medium humidity
                "Rainfall": [0.0],  # No rain
                "WetTrack": [0],  # Dry track
                "WeatherStability": [1],  # Stable weather
                "WindSpeed_Avg": [5.0],  # Light wind
                "TempDelta": [5.0],  # Normal temperature variation
            }
        )

        # Ensure all expected columns are present, add if missing with default, then reorder
        for feature_col in self.features:
            if feature_col not in features.columns:
                logger.warning(
                    f"Feature '{feature_col}' was missing from DataFrame, adding with default 0. Check inputs."
                )
                features[feature_col] = (
                    0.0  # Add with a default value (e.g., 0.0 or np.nan)
                )

        # Reorder/select columns according to self.features, ensuring model gets exactly what it expects
        try:
            return features[self.features]
        except KeyError as e:
            missing_in_df = set(self.features) - set(features.columns)
            extra_in_df = set(features.columns) - set(self.features)
            logger.error(f"KeyError during final feature selection for model: {e}")
            logger.error(
                f"Features expected by self.features but MISSING in constructed DataFrame: {missing_in_df}"
            )
            logger.error(
                f"Features in constructed DataFrame but NOT in self.features: {extra_in_df}"
            )
            raise  # Re-raise the error after logging details

    def _get_compound_hardness(self, compound: str) -> int:
        """
        Convert tire compound to hardness value.

        Args:
            compound: Tire compound string

        Returns:
            Hardness value (1=soft, 3=hard, etc.)
        """
        hardness_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
        return hardness_map.get(compound.upper(), 2)

    def _model_predict(self, features: pd.DataFrame) -> float:
        """
        Make a prediction using the trained model.

        Args:
            features: Feature DataFrame

        Returns:
            Predicted lap time in seconds
        """
        prediction = self.model.predict(features)[0]

        # Apply bounds to ensure prediction is realistic
        if prediction < 70:
            prediction = 70
        elif prediction > 120:
            prediction = 120

        return float(prediction)

    def _fallback_prediction(
        self,
        inputs: Dict[str, Any],
        driver_name: str,
        team_name: str,
        degradation_per_lap_s: float,  # NEW PARAMETER
        tire_wear_percentage: float,  # NEW PARAMETER
        grip_level: float,  # NEW PARAMETER
    ) -> float:
        """
        Fallback prediction method using heuristics when model is unavailable.

        Args:
            inputs: Input dictionary
            driver_name: Name of the driver
            team_name: Name of the team
            degradation_per_lap_s: Estimated lap time degradation from tire wear this lap (in seconds)
            tire_wear_percentage: Cumulative tire wear (0-100%)
            grip_level: Current grip level (0.0-1.0)

        Returns:
            Estimated lap time in seconds
        """
        base_lap_time = 90.0  # Generic base lap time for a dry track

        # Adjust for tire compound
        compound = inputs.get("compound", "MEDIUM")
        if compound == "SOFT":
            base_lap_time -= 1.5
        elif compound == "HARD":
            base_lap_time += 1.5
        elif compound == "INTERMEDIATE":
            base_lap_time += 7.0  # Penalty for wrong tire on dry track
        elif compound == "WET":
            base_lap_time += 15.0  # Penalty for wrong tire on dry track

        # --- Incorporate detailed tire degradation and grip ---
        # Direct impact of degradation from TireWearAgent
        base_lap_time += degradation_per_lap_s

        # Impact of overall tire wear percentage (beyond per-lap degradation, reflecting cumulative effect)
        # Apply a non-linear penalty for higher wear
        if tire_wear_percentage > 70:
            base_lap_time += (
                tire_wear_percentage - 70
            ) * 0.2  # Higher penalty for very high wear
        elif tire_wear_percentage > 40:
            base_lap_time += (
                tire_wear_percentage - 40
            ) * 0.1  # Moderate penalty for mid wear

        # Impact of grip level: lower grip means slower lap time
        # Grip level is between 0.0 and 1.0 (1.0 is full grip)
        # If grip_level is 0.8, it means 20% grip loss. Let's say 20% grip loss adds 1 second.
        if grip_level < 1.0:
            grip_loss_penalty = (
                1.0 - grip_level
            ) * 5.0  # Example: 0.1 grip loss = 0.5s penalty
            base_lap_time += grip_loss_penalty
        # --------------------------------------------------------

        # Placeholder for driver/team specific adjustments
        driver_adjustment = 0.0
        if "Hamilton" in driver_name and "Mercedes" in team_name:
            driver_adjustment = -0.5
        elif "Verstappen" in driver_name and "Red Bull" in team_name:
            driver_adjustment = -0.4
        elif "Leclerc" in driver_name and "Ferrari" in team_name:
            driver_adjustment = -0.3
        elif "Latifi" in driver_name:
            driver_adjustment = 0.5

        team_adjustment = 0.0
        if "Red Bull" in team_name:
            team_adjustment = -0.3
        elif "Mercedes" in team_name:
            team_adjustment = -0.2
        elif "Williams" in team_name:
            team_adjustment = 0.3

        base_lap_time += driver_adjustment + team_adjustment

        # Simulate some variability
        variability = np.random.normal(0, 0.5)
        predicted_time = base_lap_time + variability

        # Apply bounds
        return max(70.0, min(predicted_time, 150.0))

    def _determine_factors(
        self,
        inputs: Dict[str, Any],
        predicted_time: float,
        driver_name: str,
        team_name: str,
        degradation_per_lap_s: float,  # NEW PARAMETER
        tire_wear_percentage: float,  # NEW PARAMETER
        grip_level: float,  # NEW PARAMETER
    ) -> List[Dict[str, Any]]:
        """
        Determine key factors influencing the lap time prediction.

        Args:
            inputs: Input dictionary
            predicted_time: The predicted lap time
            driver_name: Name of the driver
            team_name: Name of the team
            degradation_per_lap_s: Estimated lap time degradation from tire wear this lap (in seconds)
            tire_wear_percentage: Cumulative tire wear (0-100%)
            grip_level: Current grip level (0.0-1.0)

        Returns:
            List of dictionaries, each representing a factor and its impact.
        """
        factors = []

        # Tire compound factor
        compound = inputs.get("compound", "MEDIUM")
        factors.append(
            {
                "name": "Tire Compound",
                "value": compound,
                "impact": "high" if compound in ["SOFT", "WET"] else "medium",
            }
        )

        # Tire age factor (can be seen as a proxy for wear, but now we have more direct metrics)
        tire_age = inputs.get("tire_age", 0)
        if tire_age > 15:
            impact = "high"
        elif tire_age > 5:
            impact = "medium"
        else:
            impact = "low"
        factors.append({"name": "Tire Age (laps)", "value": tire_age, "impact": impact})

        # --- NEW FACTORS FROM TIRE AGENTS ---
        if degradation_per_lap_s > 0.1:
            factors.append(
                {
                    "name": "Tire Degradation (this lap)",
                    "value": f"{degradation_per_lap_s:.2f}s",
                    "impact": "high" if degradation_per_lap_s > 0.5 else "medium",
                }
            )

        if tire_wear_percentage > 50:
            factors.append(
                {
                    "name": "Cumulative Tire Wear",
                    "value": f"{tire_wear_percentage:.1f}%",
                    "impact": "high" if tire_wear_percentage > 70 else "medium",
                }
            )

        if grip_level < 0.9:
            factors.append(
                {
                    "name": "Tire Grip Level",
                    "value": f"{grip_level:.2f}",
                    "impact": "high" if grip_level < 0.8 else "medium",
                }
            )

        # Circuit/Event factor
        event = inputs.get("event", inputs.get("circuit_id", "unknown") + " Grand Prix")
        factors.append({"name": "Circuit", "value": event, "impact": "high"})

        # Example: Add driver/team as a factor if they were used in fallback
        if not self.model:
            if "Hamilton" in driver_name:
                factors.append(
                    {
                        "name": "Driver Skill (Hamilton)",
                        "impact": "Significant positive",
                        "value": "-0.5s (estimated)",
                    }
                )
            if "Red Bull" in team_name:
                factors.append(
                    {
                        "name": "Team Performance (Red Bull)",
                        "impact": "Positive",
                        "value": "-0.3s (estimated)",
                    }
                )

        return factors
