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
        temp_performance = self._calculate_temp_performance(
            compound, tire_temp, track_temp, driver_chars
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

    def _calculate_temp_performance(
        self,
        compound: str,
        tire_temp: float,
        track_temp: float,
        driver_chars: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Calculate grip impact from tire temperature.

        Args:
            compound: Tire compound
            tire_temp: Current tire temperature
            track_temp: Track temperature
            driver_chars: Driver characteristics

        Returns:
            Dictionary with temperature performance details
        """
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
        working_range = (optimal_range[0] - 10, optimal_range[1] + 10)

        # Calculate base grip based on temperature
        if optimal_range[0] <= tire_temp <= optimal_range[1]:
            # In optimal window - full grip
            temp_performance = 1.0
            in_optimal_window = True
            in_working_window = True
        elif working_range[0] <= tire_temp <= working_range[1]:
            # In working window but not optimal
            if tire_temp < optimal_range[0]:
                # Cold tires
                delta = (tire_temp - working_range[0]) / (
                    optimal_range[0] - working_range[0]
                )
                temp_performance = 0.7 + (0.3 * delta)
            else:
                # Hot tires
                delta = (working_range[1] - tire_temp) / (
                    working_range[1] - optimal_range[1]
                )
                temp_performance = 0.7 + (0.3 * delta)
            in_optimal_window = False
            in_working_window = True
        else:
            # Outside working window
            if tire_temp < working_range[0]:
                # Very cold
                temp_performance = max(0.4, 0.7 * (tire_temp / working_range[0]))
            else:
                # Very hot
                temp_performance = max(0.4, 0.7 * (working_range[1] / tire_temp))
            in_optimal_window = False
            in_working_window = False

        # Track temperature effect
        track_delta = abs(track_temp - optimal_temp)
        track_effect = max(0.7, 1.0 - (track_delta / 100.0))

        # Driver adaptability
        consistency = driver_chars.get("consistency", 1.0)
        tire_management = driver_chars.get("tire_management", 1.0)
        driver_adaptation = (consistency + tire_management) / 2

        # Better drivers can extract more grip from suboptimal temperatures
        if not in_optimal_window:
            temp_performance = temp_performance + (
                (1.0 - temp_performance) * (driver_adaptation - 1.0) * 0.3
            )

        # Calculate final grip multiplier
        grip_multiplier = temp_performance * track_effect

        return {
            "grip_multiplier": float(grip_multiplier),
            "in_optimal_window": in_optimal_window,
            "in_working_window": in_working_window,
            "temp_performance": float(temp_performance),
            "track_effect": float(track_effect),
        }

    def _calculate_wear_grip(
        self, compound: str, wear: float, driver_chars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate grip based on tire wear with compound-specific characteristics.
        """
        # Get driver characteristics
        consistency = driver_chars.get("consistency", 1.0)
        tire_management = driver_chars.get("tire_management", 1.0)

        # Compound-specific characteristics
        compound_chars = {
            "SOFT": {
                "initial_grip": 1.02,  # Highest peak grip
                "wear_resistance": 0.7,  # Wears faster
                "cliff_point": 65.0,  # Earlier cliff
                "cliff_severity": 1.3,  # More severe drop-off
            },
            "MEDIUM": {
                "initial_grip": 1.0,  # Baseline grip
                "wear_resistance": 0.85,  # Balanced wear
                "cliff_point": 75.0,  # Standard cliff
                "cliff_severity": 1.1,  # Standard drop-off
            },
            "HARD": {
                "initial_grip": 0.98,  # Lower peak grip
                "wear_resistance": 1.0,  # Most wear resistant
                "cliff_point": 85.0,  # Later cliff
                "cliff_severity": 0.9,  # More gradual drop-off
            },
            "INTERMEDIATE": {
                "initial_grip": 0.95,
                "wear_resistance": 0.8,
                "cliff_point": 70.0,
                "cliff_severity": 1.2,
            },
            "WET": {
                "initial_grip": 0.90,
                "wear_resistance": 0.75,
                "cliff_point": 60.0,
                "cliff_severity": 1.4,
            },
        }

        # Get compound characteristics (default to medium if unknown)
        chars = compound_chars.get(compound, compound_chars["MEDIUM"])

        # Calculate base grip loss based on wear
        # Better tire management reduces grip loss
        effective_wear = wear * (1.1 - (tire_management * 0.1))

        if effective_wear < chars["cliff_point"] * 0.5:
            # First half of tire life - gradual grip loss
            base_grip_loss = (effective_wear / chars["cliff_point"]) ** (
                1.2 * chars["wear_resistance"]
            )
        elif effective_wear < chars["cliff_point"]:
            # Approaching cliff - accelerating grip loss
            progress_to_cliff = (effective_wear - (chars["cliff_point"] * 0.5)) / (
                chars["cliff_point"] * 0.5
            )
            base_grip_loss = 0.5 + (progress_to_cliff * 0.3)  # More pronounced loss
        else:
            # After cliff point - severe grip loss
            base_grip_loss = 0.8  # Significant base loss at cliff
            excess_wear = effective_wear - chars["cliff_point"]
            # More severe drop-off based on compound characteristics
            cliff_loss = (excess_wear / (100 - chars["cliff_point"])) ** chars[
                "cliff_severity"
            ]
            base_grip_loss += cliff_loss * 0.2  # Additional loss after cliff

        # Calculate grip level
        grip_level = chars["initial_grip"] * (1.0 - base_grip_loss)

        # Add consistency-based variations
        # Less consistent drivers have more grip variations
        variation_scale = 0.02 * (2.0 - consistency)
        variation = np.random.normal(0, variation_scale)
        grip_level *= 1.0 + variation

        # Ensure grip stays within bounds
        grip_level = max(0.3, min(chars["initial_grip"], grip_level))

        return {
            "grip_multiplier": grip_level,
            "base_grip_loss": base_grip_loss,
            "high_wear_active": effective_wear > chars["cliff_point"],
            "cliff_active": effective_wear > chars["cliff_point"],
            "compound_characteristics": chars,
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
