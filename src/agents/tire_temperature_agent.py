"""
Agent for simulating tire temperature dynamics based on driver behavior and weather.
"""

from typing import Dict, Any
import numpy as np
import logging
from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TireTemperatureAgent(BaseAgent):
    """
    Agent responsible for simulating tire temperature dynamics considering:
    - Driver characteristics (aggression, tire management)
    - Weather conditions (track temp, air temp, rainfall)
    - Compound-specific thermal properties
    """

    def __init__(self, name: str = "TireTemperatureAgent"):
        super().__init__(name)

        # Thermal characteristics by compound
        self.compound_thermal_properties = {
            "SOFT": {
                "heat_retention": 0.85,  # How well the tire retains heat
                "temp_sensitivity": 1.2,  # How quickly it responds to temperature changes
                "optimal_operating_temp": 95.0,  # Center of optimal window
            },
            "MEDIUM": {
                "heat_retention": 0.80,
                "temp_sensitivity": 1.0,
                "optimal_operating_temp": 90.0,
            },
            "HARD": {
                "heat_retention": 0.75,
                "temp_sensitivity": 0.8,
                "optimal_operating_temp": 85.0,
            },
            "INTERMEDIATE": {
                "heat_retention": 0.70,
                "temp_sensitivity": 0.9,
                "optimal_operating_temp": 70.0,
            },
            "WET": {
                "heat_retention": 0.65,
                "temp_sensitivity": 0.7,
                "optimal_operating_temp": 50.0,
            },
        }

        # Weather impact coefficients
        self.weather_coefficients = {
            "track_temp_influence": 0.6,  # How much track temp affects tire temp
            "air_temp_influence": 0.4,  # How much air temp affects tire temp
            "rain_cooling_factor": 0.3,  # How much rain cools the tires
        }

        # Driver behavior impact coefficients
        self.driver_coefficients = {
            "aggression_heat_factor": 15.0,  # Max temperature increase from aggressive driving
            "management_cooling_factor": 10.0,  # Max temperature reduction from good tire management
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate tire temperature based on various factors.

        Args:
            inputs: Dictionary containing:
                - compound: Tire compound
                - current_temp: Current tire temperature
                - track_temp: Track temperature
                - air_temp: Air temperature
                - rainfall: Rainfall amount
                - driver_characteristics: Driver characteristics dict
                - is_pit_lap: Whether this is a pit lap
                - lap_number: Current lap number

        Returns:
            Dictionary with updated tire temperature and status
        """
        compound = inputs.get("compound", "MEDIUM")
        current_temp = inputs.get("current_temp", 80.0)
        track_temp = inputs.get("track_temp", 30.0)
        air_temp = inputs.get("air_temp", 25.0)
        rainfall = inputs.get("rainfall", 0.0)
        driver_chars = inputs.get("driver_characteristics", {})
        is_pit_lap = inputs.get("is_pit_lap", False)

        # Define optimal temperature windows per compound
        optimal_temp_windows = {
            "SOFT": (90, 110),
            "MEDIUM": (85, 105),
            "HARD": (80, 100),
            "INTERMEDIATE": (65, 85),
            "WET": (55, 75),
        }

        # Get optimal range for current compound
        optimal_range = optimal_temp_windows.get(compound, (85, 105))
        optimal_temp = (optimal_range[0] + optimal_range[1]) / 2

        # Calculate ambient influence (weighted average of track and air temp)
        ambient_influence = (
            (track_temp * 0.6) + (air_temp * 0.4) + 60.0
        )  # Base temp boost to get into F1 operating range

        # Driver heat generation based on characteristics
        aggression = driver_chars.get("aggression", 1.0)
        consistency = driver_chars.get("consistency", 1.0)
        driver_heat = (
            25.0 * aggression * consistency
        )  # More aggressive drivers generate more heat

        # Cooling effects
        tire_management = driver_chars.get("tire_management", 1.0)
        management_cooling = (
            -15.0 * tire_management
        )  # Better tire management means better temperature control

        # Rain cooling effect
        rain_cooling = -rainfall * 10.0 if rainfall > 0 else 0.0

        # Calculate temperature change
        if is_pit_lap:
            # Reset temperature on pit stop but maintain some heat
            new_temp = ambient_influence + (driver_heat * 0.5)
        else:
            # Normal lap temperature calculation
            target_temp = ambient_influence + driver_heat + management_cooling
            temp_diff = target_temp - current_temp
            new_temp = current_temp + (temp_diff * 0.3) + rain_cooling  # Gradual change

        # Ensure temperature stays within physical limits
        new_temp = max(min(new_temp, 140.0), 60.0)  # Adjusted temperature bounds

        # Calculate temperature status
        is_cold = new_temp < optimal_range[0]
        is_overheating = new_temp > optimal_range[1]
        temp_delta = abs(new_temp - optimal_temp)

        return {
            "tire_temp": new_temp,
            "is_cold": is_cold,
            "is_overheating": is_overheating,
            "optimal_temp": optimal_temp,
            "temp_delta": temp_delta,
            "temp_status": {
                "is_cold": is_cold,
                "is_overheating": is_overheating,
                "optimal_range": optimal_range,
                "current_temp": new_temp,
            },
        }
