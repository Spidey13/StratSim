"""
Lap time prediction agent that forecasts lap times based on various factors.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the default model path relative to the project root
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "laptime_xgboost_pipeline_tuned_v1.joblib"
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
    ):
        """
        Initialize the lap time prediction agent.

        Args:
            name: Agent name
            model_path: Path to the trained lap time prediction model.
                        Defaults to the standard tuned model path.
        """
        super().__init__(name)
        self.model_path = model_path
        self.model = self._load_model(self.model_path) if self.model_path else None
        self.features = [
            "TyreLife",
            "CompoundHardness",
            "TrackTemp_Avg",
            "AirTemp_Avg",
            "Humidity_Avg",
            "Rainfall",
            "Event",  # Updated to use Event instead of Circuit
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

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to predict lap times.

        Args:
            inputs: Dictionary containing:
                - circuit_id: ID of the circuit
                - compound: Tire compound
                - tire_age: Age of tires in laps
                - weather: Weather conditions
                - driver: Driver ID
                - event: Event/Grand Prix name

        Returns:
            Dictionary containing:
                - predicted_laptime: Predicted lap time in seconds
                - confidence: Confidence level of prediction
                - factors: Factors influencing the prediction
        """
        # Extract required inputs
        circuit_id = inputs.get("circuit_id")
        compound = inputs.get("compound", "MEDIUM")
        tire_age = inputs.get("tire_age", 0)
        weather = inputs.get("weather", {})
        driver = inputs.get("driver")
        event = inputs.get("event", inputs.get("circuit_id", "unknown") + " Grand Prix")

        # If we have a trained model, use it
        if self.model:
            try:
                # Prepare features for prediction
                features = self._prepare_features(inputs)
                predicted_time = self._model_predict(features)
                confidence = 0.95  # Placeholder
            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}")
                predicted_time = self._fallback_prediction(inputs)
                confidence = 0.7  # Lower confidence for fallback
        else:
            # Fallback to heuristic model
            predicted_time = self._fallback_prediction(inputs)
            confidence = 0.7

        # Determine factors influencing lap time
        factors = self._determine_factors(inputs, predicted_time)

        return {
            "predicted_laptime": predicted_time,
            "confidence": confidence,
            "factors": factors,
        }

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
                "TrackTemp_Avg": [inputs.get("weather", {}).get("track_temp", 30)],
                "AirTemp_Avg": [inputs.get("weather", {}).get("air_temp", 20)],
                "Humidity_Avg": [inputs.get("weather", {}).get("humidity", 50)],
                "Rainfall": [inputs.get("weather", {}).get("rainfall", 0)],
                "Event": [
                    inputs.get(
                        "event", inputs.get("circuit_id", "unknown") + " Grand Prix"
                    )
                ],
            }
        )

        return features

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
        # Most F1 lap times are between 70-120 seconds
        if prediction < 70:
            prediction = 70
        elif prediction > 120:
            prediction = 120

        return float(prediction)

    def _fallback_prediction(self, inputs: Dict[str, Any]) -> float:
        """
        Fallback prediction method using heuristics when model is unavailable.

        Args:
            inputs: Input dictionary

        Returns:
            Estimated lap time in seconds
        """
        # Base lap time depends on circuit (placeholder values)
        circuit_base_times = {
            "monza": 85.0,  # Fast circuit
            "monaco": 75.0,  # Slow circuit
            "silverstone": 90.0,  # Medium circuit
            "spa": 104.0,  # Longest circuit
            "singapore": 98.0,  # Street circuit
        }

        circuit = inputs.get("circuit_id", "unknown").lower()
        base_time = circuit_base_times.get(circuit, 90.0)  # Default if unknown

        # Adjust for tire compound
        compound = inputs.get("compound", "MEDIUM").upper()
        compound_factor = {
            "SOFT": -0.5,  # Faster
            "MEDIUM": 0.0,  # Neutral
            "HARD": 0.8,  # Slower
            "INTERMEDIATE": 2.0,  # Much slower
            "WET": 4.0,  # Very slow
        }

        # Adjust for tire age (degradation)
        tire_age = inputs.get("tire_age", 0)
        wear_factor = 0.05 * tire_age  # 0.05 seconds per lap of age

        # Adjust for weather
        weather = inputs.get("weather", {})
        rainfall = weather.get("rainfall", 0)
        weather_factor = rainfall * 2.0  # 2 seconds slower per unit of rainfall

        # Adjust for temperature effects
        track_temp = weather.get("track_temp", 30)
        temp_factor = 0.0
        if track_temp < 20:  # Cold track
            temp_factor = 0.8  # Slower
        elif track_temp > 50:  # Very hot track
            temp_factor = 0.5  # Slower due to overheating

        # Adjust for humidity
        humidity = weather.get("humidity", 50)
        humidity_factor = 0.0
        if humidity > 80 and track_temp > 30:
            humidity_factor = (
                0.3  # Higher humidity makes the track slower in hot conditions
            )

        # Calculate final time
        adjusted_time = (
            base_time
            + compound_factor.get(compound, 0.0)
            + wear_factor
            + weather_factor
            + temp_factor
            + humidity_factor
        )

        return adjusted_time

    def _determine_factors(
        self, inputs: Dict[str, Any], predicted_time: float
    ) -> List[Dict[str, Any]]:
        """
        Determine the factors influencing the lap time prediction.

        Args:
            inputs: Input dictionary
            predicted_time: Predicted lap time

        Returns:
            List of factors with their impact
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

        # Tire age factor
        tire_age = inputs.get("tire_age", 0)
        if tire_age > 15:
            impact = "high"
        elif tire_age > 5:
            impact = "medium"
        else:
            impact = "low"

        factors.append({"name": "Tire Age", "value": tire_age, "impact": impact})

        # Weather factors
        weather = inputs.get("weather", {})
        if weather.get("rainfall", 0) > 0:
            factors.append(
                {"name": "Rainfall", "value": weather.get("rainfall"), "impact": "high"}
            )

        # Track temperature
        track_temp = weather.get("track_temp", 30)
        temp_impact = "low"
        if track_temp < 20 or track_temp > 45:
            temp_impact = "high"
        elif 20 <= track_temp < 25 or 40 <= track_temp <= 45:
            temp_impact = "medium"

        factors.append(
            {"name": "Track Temperature", "value": track_temp, "impact": temp_impact}
        )

        # Air temperature
        air_temp = weather.get("air_temp", 20)
        factors.append(
            {"name": "Air Temperature", "value": air_temp, "impact": "medium"}
        )

        # Humidity
        humidity = weather.get("humidity", 50)
        humidity_impact = "low"
        if humidity > 80:
            humidity_impact = "medium"
        factors.append(
            {"name": "Humidity", "value": humidity, "impact": humidity_impact}
        )

        # Circuit/Event factor
        event = inputs.get("event", inputs.get("circuit_id", "unknown") + " Grand Prix")
        factors.append({"name": "Circuit", "value": event, "impact": "high"})

        return factors
