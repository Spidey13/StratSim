"""
Tire wear modeling agent that estimates tire degradation and performance.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import logging
import joblib

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TireWearAgent(BaseAgent):
    """
    Agent responsible for modeling tire wear and degradation
    based on compound, circuit, driving style, and conditions.
    """

    def __init__(self, name: str = "TireWearAgent", model_path: Optional[str] = None):
        """
        Initialize the tire wear modeling agent.

        Args:
            name: Agent name
            model_path: Path to the trained tire wear model
        """
        super().__init__(name)
        self.model = self._load_model(model_path) if model_path else None

        # Define base degradation rates per compound (percentage per lap)
        self.base_degradation = {
            "SOFT": 1.8,  # Faster degradation
            "MEDIUM": 1.2,  # Medium degradation
            "HARD": 0.8,  # Slower degradation
            "INTERMEDIATE": 2.0,  # Fast in mixed conditions
            "WET": 0.5,  # Very slow in full wet conditions
        }

    def _load_model(self, model_path: str):
        """
        Load the trained tire wear model.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object
        """
        try:
            logger.info(f"Loading tire wear model from {model_path}")
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to estimate tire wear.

        Args:
            inputs: Dictionary containing:
                - compound: Tire compound
                - tire_age: Age of tires in laps
                - circuit_id: ID of the circuit
                - driver_id: Driver ID
                - weather: Weather conditions

        Returns:
            Dictionary containing:
                - tire_wear: Estimated tire wear percentage (0-100)
                - grip_level: Estimated remaining grip (0-1)
                - optimal_window: Whether the tire is in optimal temp window
                - wear_factors: Factors contributing to wear
        """
        # Extract inputs
        compound = inputs.get("compound", "MEDIUM")
        tire_age = inputs.get("tire_age", 0)
        circuit_id = inputs.get("circuit_id", "unknown")
        driver_id = inputs.get("driver_id", "unknown")
        weather = inputs.get("weather", {})

        # Use model if available
        if self.model:
            try:
                features = self._prepare_features(inputs)
                wear_pct = self._model_predict(features)
                confidence = 0.95
            except Exception as e:
                logger.error(f"Model prediction error: {str(e)}")
                wear_pct = self._calculate_wear_fallback(inputs)
                confidence = 0.7
        else:
            # Fallback to heuristic calculation
            wear_pct = self._calculate_wear_fallback(inputs)
            confidence = 0.7

        # Calculate remaining grip based on wear
        # Non-linear relationship - grip falls off more quickly at high wear
        grip_remaining = max(0, 1 - (wear_pct / 100) ** 1.5)

        # Determine if tire is in optimal temperature window
        # (simplified logic - in real F1, this would be more complex)
        track_temp = weather.get("track_temp", 30)
        optimal_window = self._is_in_optimal_temp_window(compound, track_temp)

        # Determine factors affecting wear
        wear_factors = self._determine_wear_factors(inputs, wear_pct)

        return {
            "tire_wear": wear_pct,
            "grip_level": grip_remaining,
            "optimal_window": optimal_window,
            "confidence": confidence,
            "wear_factors": wear_factors,
        }

    def _prepare_features(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for model prediction.

        Args:
            inputs: Input dictionary

        Returns:
            DataFrame with prepared features
        """
        # Create a single row dataframe with all features
        features = pd.DataFrame(
            {
                "TyreLife": [inputs.get("tire_age", 0)],
                "CompoundHardness": [
                    self._get_compound_hardness(inputs.get("compound", "MEDIUM"))
                ],
                "TrackTemp": [inputs.get("weather", {}).get("track_temp", 30)],
                "AirTemp": [inputs.get("weather", {}).get("air_temp", 20)],
                "Humidity": [inputs.get("weather", {}).get("humidity", 50)],
                "Rainfall": [inputs.get("weather", {}).get("rainfall", 0)],
                "Circuit": [inputs.get("circuit_id", "unknown")],
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
            Predicted tire wear percentage
        """
        prediction = self.model.predict(features)[0]
        return float(prediction)

    def _calculate_wear_fallback(self, inputs: Dict[str, Any]) -> float:
        """
        Calculate tire wear without a model using heuristics.

        Args:
            inputs: Input dictionary

        Returns:
            Estimated tire wear percentage
        """
        compound = inputs.get("compound", "MEDIUM").upper()
        tire_age = inputs.get("tire_age", 0)
        weather = inputs.get("weather", {})

        # Base degradation per lap based on compound
        base_rate = self.base_degradation.get(compound, 1.2)

        # Circuit abrasiveness factor (default to medium)
        circuit_id = inputs.get("circuit_id", "unknown").lower()
        circuit_factors = {
            "monaco": 0.8,  # Low abrasion
            "monza": 1.3,  # High speed, medium abrasion
            "barcelona": 1.5,  # High abrasion
            "silverstone": 1.4,  # High energy loading
        }
        circuit_factor = circuit_factors.get(circuit_id, 1.0)

        # Weather impact on degradation
        track_temp = weather.get("track_temp", 30)
        rainfall = weather.get("rainfall", 0)

        # Temperature impact
        # Higher temps increase wear, except for wet tires
        if compound in ["WET", "INTERMEDIATE"]:
            temp_factor = max(0.5, 1.0 - (track_temp - 20) * 0.03)
        else:
            temp_factor = 1.0 + max(0, (track_temp - 25) * 0.03)

        # Rain impact
        # Rain reduces wear for wet/inter, increases for slicks
        if rainfall > 0:
            if compound in ["WET", "INTERMEDIATE"]:
                rain_factor = max(0.5, 1.0 - rainfall * 0.2)
            else:
                rain_factor = 1.0 + rainfall * 0.5
        else:
            rain_factor = 1.0

        # Calculate total wear
        # First few laps have a break-in period with lower wear
        if tire_age <= 2:
            break_in_factor = 0.7
        else:
            break_in_factor = 1.0

        wear_per_lap = (
            base_rate * circuit_factor * temp_factor * rain_factor * break_in_factor
        )
        total_wear = min(100, wear_per_lap * tire_age)

        return total_wear

    def _is_in_optimal_temp_window(self, compound: str, track_temp: float) -> bool:
        """
        Determine if tire is in its optimal temperature window.

        Args:
            compound: Tire compound
            track_temp: Track temperature in Celsius

        Returns:
            Boolean indicating if tire is in optimal temperature window
        """
        # Optimal temperature windows for each compound
        optimal_ranges = {
            "SOFT": (35, 50),
            "MEDIUM": (30, 45),
            "HARD": (25, 40),
            "INTERMEDIATE": (15, 25),
            "WET": (5, 20),
        }

        compound_range = optimal_ranges.get(compound.upper(), (25, 40))
        return compound_range[0] <= track_temp <= compound_range[1]

    def _determine_wear_factors(
        self, inputs: Dict[str, Any], wear_pct: float
    ) -> List[Dict[str, Any]]:
        """
        Determine factors contributing to tire wear.

        Args:
            inputs: Input dictionary
            wear_pct: Calculated wear percentage

        Returns:
            List of factors with their impact
        """
        factors = []

        # Compound factor
        compound = inputs.get("compound", "MEDIUM")
        if compound in ["SOFT"]:
            impact = "high"
        elif compound in ["MEDIUM"]:
            impact = "medium"
        else:
            impact = "low"

        factors.append({"name": "Compound", "value": compound, "impact": impact})

        # Circuit factor
        circuit_id = inputs.get("circuit_id", "unknown")
        circuit_abrasion = {
            "monaco": "low",
            "barcelona": "high",
            "silverstone": "high",
            "monza": "medium",
        }

        factors.append(
            {
                "name": "Circuit",
                "value": circuit_id,
                "impact": circuit_abrasion.get(circuit_id.lower(), "medium"),
            }
        )

        # Weather factors
        weather = inputs.get("weather", {})
        track_temp = weather.get("track_temp", 30)
        rainfall = weather.get("rainfall", 0)

        if track_temp > 40:
            temp_impact = "high"
        elif track_temp > 30:
            temp_impact = "medium"
        else:
            temp_impact = "low"

        factors.append(
            {"name": "Track Temperature", "value": track_temp, "impact": temp_impact}
        )

        if rainfall > 0:
            factors.append(
                {
                    "name": "Rainfall",
                    "value": rainfall,
                    "impact": "high"
                    if compound not in ["WET", "INTERMEDIATE"]
                    else "low",
                }
            )

        return factors
