"""
Agent for calculating tire grip levels with advanced modeling of track evolution,
temperature effects, and wear-based variations.
"""

from typing import Dict, Any
import numpy as np
import logging
from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)


class GripModelAgent(BaseAgent):
    """
    Agent responsible for calculating tire grip levels with consideration for:
    - Track evolution over the race
    - Temperature effects on different compounds
    - Non-linear grip loss at high wear
    - Weather impacts on grip
    """

    def __init__(self, name: str = "GripModelAgent"):
        super().__init__(name)

        # Track evolution parameters
        self.track_evolution = {
            "base_grip": 0.92,  # Starting grip level (92% of maximum)
            "max_evolution": 0.08,  # Maximum additional grip from evolution (up to 100%)
            "evolution_rate": 0.002,  # Grip increase per lap from racing line
            "rain_reset_factor": 0.5,  # How much evolution is lost in rain
        }

        # Temperature windows for each compound
        self.temp_windows = {
            "SOFT": {
                "optimal": (90, 100),  # Optimal working range
                "working": (80, 110),  # Wider working range
                "critical_low": 60,
                "critical_high": 130,
            },
            "MEDIUM": {
                "optimal": (85, 95),
                "working": (75, 105),
                "critical_low": 55,
                "critical_high": 125,
            },
            "HARD": {
                "optimal": (80, 90),
                "working": (70, 100),
                "critical_low": 50,
                "critical_high": 120,
            },
            "INTERMEDIATE": {
                "optimal": (60, 80),
                "working": (40, 100),
                "critical_low": 30,
                "critical_high": 110,
            },
            "WET": {
                "optimal": (40, 60),
                "working": (30, 80),
                "critical_low": 20,
                "critical_high": 90,
            },
        }

        # High wear variation parameters
        self.high_wear_effects = {
            "threshold": 70.0,  # When high wear effects start
            "variation_scale": 0.02,  # Scale of random variations
            "cliff_point": 85.0,  # When severe performance drop begins
            "max_grip_loss": 0.7,  # Maximum grip loss at 100% wear
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate grip levels considering all factors.

        Args:
            inputs: Dictionary containing:
                - compound: Tire compound
                - tire_wear: Current tire wear percentage
                - tire_temp: Current tire temperature
                - track_temp: Track temperature
                - current_lap: Current lap number
                - total_laps: Total race laps
                - weather: Weather conditions
                - driver_characteristics: Driver-specific factors

        Returns:
            Dictionary containing:
                - grip_level: Overall grip level (0.0-1.0)
                - track_evolution: Current track evolution level
                - temp_performance: Temperature-based performance
                - wear_grip: Wear-based grip level
                - in_optimal_window: Whether tire is in optimal temperature window
        """
        # Extract inputs
        compound = inputs.get("compound", "MEDIUM")
        tire_wear = inputs.get("tire_wear", 0.0)
        tire_temp = inputs.get("tire_temp", 90.0)
        track_temp = inputs.get("track_temp", 30.0)
        current_lap = inputs.get("current_lap", 1)
        total_laps = inputs.get("total_laps", 50)
        weather = inputs.get("weather", {})
        driver_chars = inputs.get("driver_characteristics", {})

        # Calculate track evolution
        track_grip = self._calculate_track_evolution(current_lap, total_laps, weather)

        # Calculate temperature effects
        temp_performance = self._calculate_temperature_performance(
            compound, tire_temp, track_temp
        )

        # Calculate wear-based grip with high wear variations
        wear_grip = self._calculate_wear_grip(compound, tire_wear, driver_chars)

        # Combine all effects
        # Track evolution provides the baseline
        # Temperature and wear effects multiply from there
        base_grip = track_grip
        temp_multiplier = temp_performance["grip_multiplier"]
        wear_multiplier = wear_grip["grip_multiplier"]

        # Calculate final grip level
        grip_level = base_grip * temp_multiplier * wear_multiplier

        # Ensure grip stays within bounds
        grip_level = max(0.3, min(1.0, grip_level))

        # Log detailed calculations
        logger.debug(
            f"Grip calculation:\n"
            f"  Track evolution: {track_grip:.3f}\n"
            f"  Temperature multiplier: {temp_multiplier:.3f}\n"
            f"  Wear multiplier: {wear_multiplier:.3f}\n"
            f"  Final grip: {grip_level:.3f}"
        )

        return {
            "grip_level": grip_level,
            "track_evolution": track_grip,
            "temp_performance": temp_performance,
            "wear_grip": wear_grip,
            "in_optimal_window": temp_performance["in_optimal_window"],
        }

    def _calculate_track_evolution(
        self, current_lap: int, total_laps: int, weather: Dict[str, Any]
    ) -> float:
        """
        Calculate track grip evolution over the race distance.
        """
        # Base evolution from rubber laid down
        evolution_progress = min(1.0, current_lap / (total_laps * 0.7))
        natural_evolution = self.track_evolution["base_grip"] + (
            self.track_evolution["max_evolution"] * evolution_progress
        )

        # Additional evolution from racing line
        racing_line_evolution = min(
            0.03,  # Cap at 3% additional grip
            current_lap * self.track_evolution["evolution_rate"],
        )

        # Weather effects
        rainfall = weather.get("rainfall", 0)
        if rainfall > 0:
            # Rain washes away rubber
            evolution_loss = rainfall * self.track_evolution["rain_reset_factor"]
            # More grip is lost in heavy rain
            current_grip = max(
                self.track_evolution["base_grip"], natural_evolution - evolution_loss
            )
        else:
            current_grip = natural_evolution + racing_line_evolution

        return current_grip

    def _calculate_temperature_performance(
        self, compound: str, tire_temp: float, track_temp: float
    ) -> Dict[str, Any]:
        """
        Calculate grip based on tire and track temperatures.
        """
        temp_window = self.temp_windows.get(compound, self.temp_windows["MEDIUM"])

        # Calculate distance from optimal range
        optimal_low, optimal_high = temp_window["optimal"]
        working_low, working_high = temp_window["working"]

        # Initialize performance metrics
        in_optimal_window = optimal_low <= tire_temp <= optimal_high
        in_working_window = working_low <= tire_temp <= working_high

        # Calculate temperature performance
        if in_optimal_window:
            temp_performance = 1.0
        elif in_working_window:
            # Linear falloff within working range
            if tire_temp < optimal_low:
                temp_performance = 0.9 + (
                    0.1 * (tire_temp - working_low) / (optimal_low - working_low)
                )
            else:  # tire_temp > optimal_high
                temp_performance = 0.9 + (
                    0.1 * (working_high - tire_temp) / (working_high - optimal_high)
                )
        else:
            # Severe performance drop outside working range
            if tire_temp < working_low:
                temp_performance = max(0.5, 0.9 * (tire_temp / working_low))
            else:  # tire_temp > working_high
                temp_performance = max(0.5, 0.9 * (working_high / tire_temp))

        # Track temperature influence
        track_effect = 1.0
        if track_temp < temp_window["critical_low"]:
            track_effect = max(
                0.7, 1.0 - (temp_window["critical_low"] - track_temp) * 0.02
            )
        elif track_temp > temp_window["critical_high"]:
            track_effect = max(
                0.7, 1.0 - (track_temp - temp_window["critical_high"]) * 0.02
            )

        # Combine tire and track temperature effects
        grip_multiplier = temp_performance * track_effect

        return {
            "grip_multiplier": grip_multiplier,
            "in_optimal_window": in_optimal_window,
            "in_working_window": in_working_window,
            "temp_performance": temp_performance,
            "track_effect": track_effect,
        }

    def _calculate_wear_grip(
        self, compound: str, wear: float, driver_chars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate grip based on tire wear with increased variation at high wear.
        """
        # Get driver characteristics
        consistency = driver_chars.get("consistency", 1.0)
        tire_management = driver_chars.get("tire_management", 1.0)

        # Base grip loss calculation
        if wear < self.high_wear_effects["threshold"]:
            # Linear grip loss until threshold
            base_grip = 1.0 - (wear / 100) * 0.8
        else:
            # Non-linear loss after threshold
            base_wear_loss = self.high_wear_effects["threshold"] / 100 * 0.8
            remaining_wear = wear - self.high_wear_effects["threshold"]
            remaining_wear_factor = remaining_wear / (
                100 - self.high_wear_effects["threshold"]
            )

            # More pronounced effect after cliff point
            if wear > self.high_wear_effects["cliff_point"]:
                cliff_excess = wear - self.high_wear_effects["cliff_point"]
                cliff_factor = (
                    cliff_excess / (100 - self.high_wear_effects["cliff_point"])
                ) ** 2
                high_wear_loss = remaining_wear_factor * (1 + cliff_factor)
            else:
                high_wear_loss = remaining_wear_factor

            base_grip = 1.0 - base_wear_loss - high_wear_loss

        # Add wear-based variations
        if wear > self.high_wear_effects["threshold"]:
            # Increase variation at high wear
            variation_scale = self.high_wear_effects["variation_scale"]
            # Less consistent drivers have more variation
            variation_scale *= 2 - consistency
            # Better tire management reduces variation
            variation_scale *= 2 - tire_management

            variation = np.random.normal(0, variation_scale)
            grip_multiplier = base_grip * (1 + variation)
        else:
            grip_multiplier = base_grip

        # Ensure grip multiplier stays within bounds
        grip_multiplier = max(
            1.0 - self.high_wear_effects["max_grip_loss"], min(1.0, grip_multiplier)
        )

        return {
            "grip_multiplier": grip_multiplier,
            "base_grip": base_grip,
            "high_wear_active": wear > self.high_wear_effects["threshold"],
            "cliff_active": wear > self.high_wear_effects["cliff_point"],
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the agent."""
        return {
            "name": self.name,
            "description": "Calculates tire grip levels with advanced modeling of track evolution, temperature effects, and wear variations.",
            "inputs": [
                "compound",
                "tire_wear",
                "tire_temp",
                "track_temp",
                "current_lap",
                "total_laps",
                "weather",
                "driver_characteristics",
            ],
            "outputs": [
                "grip_level",
                "track_evolution",
                "temp_performance",
                "wear_grip",
                "in_optimal_window",
            ],
        }
