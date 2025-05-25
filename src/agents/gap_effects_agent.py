"""
Agent for calculating the effects of gaps between cars, including dirty air, DRS, and fuel load.
"""

from typing import Dict, Any
import numpy as np
import logging
from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)


class GapEffectsAgent(BaseAgent):
    """
    Agent responsible for calculating the effects of gaps between cars,
    including dirty air effects, DRS activation, and fuel load impact.
    """

    def __init__(self, name: str = "GapEffectsAgent"):
        super().__init__(name)

        # Dirty air effect parameters
        self.dirty_air_zones = {
            "severe": {"range": (0, 1.0), "effect": 0.015},  # 0-1.0s: 1.5% time loss
            "strong": {
                "range": (1.0, 2.0),
                "effect": 0.010,
            },  # 1.0-2.0s: 1.0% time loss
            "moderate": {
                "range": (2.0, 3.0),
                "effect": 0.005,
            },  # 2.0-3.0s: 0.5% time loss
            "light": {"range": (3.0, 4.0), "effect": 0.002},  # 3.0-4.0s: 0.2% time loss
        }

        # DRS parameters
        self.drs_detection_range = 1.0  # DRS detection zone gap threshold (seconds)
        self.drs_effect = -0.005  # -0.5% lap time when DRS is active
        self.drs_zones_per_lap = 2  # Default number of DRS zones

        # Fuel load parameters
        self.fuel_weight_penalty = 0.03  # 0.03s per lap per 10kg of fuel
        self.fuel_consumption = 1.8  # kg per lap
        self.initial_fuel_load = 110  # kg at race start

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the combined effects of dirty air, DRS, and fuel load on lap time.

        Args:
            inputs: Dictionary containing:
                - gap_to_ahead: Gap to car ahead in seconds
                - current_lap: Current lap number
                - base_lap_time: Base lap time without effects
                - track_characteristics: Dictionary with track-specific parameters
                - driver_characteristics: Dictionary with driver-specific parameters

        Returns:
            Dictionary containing:
                - lap_time_delta: Combined time effect in seconds
                - dirty_air_effect: Time lost due to dirty air
                - drs_active: Whether DRS was available
                - drs_effect: Time gained from DRS
                - fuel_effect: Time lost due to fuel load
        """
        gap_to_ahead = inputs.get("gap_to_ahead", 10.0)  # Default to clean air
        current_lap = inputs.get("current_lap", 1)
        base_lap_time = inputs.get("base_lap_time", 90.0)
        track_chars = inputs.get("track_characteristics", {})
        driver_chars = inputs.get("driver_characteristics", {})

        # Calculate dirty air effect
        dirty_air_effect = self._calculate_dirty_air_effect(
            gap_to_ahead, base_lap_time, track_chars
        )

        # Calculate DRS effect
        drs_effect = self._calculate_drs_effect(
            gap_to_ahead, base_lap_time, track_chars
        )

        # Calculate fuel load effect
        fuel_effect = self._calculate_fuel_load_effect(current_lap, base_lap_time)

        # Combine all effects
        total_effect = dirty_air_effect + drs_effect + fuel_effect

        # Log detailed calculations
        logger.debug(
            f"Gap effects calculation:\n"
            f"  Gap to ahead: {gap_to_ahead:.3f}s\n"
            f"  Dirty air effect: {dirty_air_effect:.3f}s\n"
            f"  DRS effect: {drs_effect:.3f}s\n"
            f"  Fuel load effect: {fuel_effect:.3f}s\n"
            f"  Total effect: {total_effect:.3f}s"
        )

        return {
            "lap_time_delta": total_effect,
            "dirty_air_effect": dirty_air_effect,
            "drs_active": drs_effect < 0,  # DRS is active if there's a negative effect
            "drs_effect": drs_effect,
            "fuel_effect": fuel_effect,
        }

    def _calculate_dirty_air_effect(
        self, gap_ahead: float, base_lap_time: float, track_chars: Dict[str, Any]
    ) -> float:
        """
        Calculate the time loss due to dirty air effect.
        Returns the time penalty in seconds.
        """
        # Get track-specific dirty air sensitivity (default to 1.0)
        track_sensitivity = track_chars.get("dirty_air_sensitivity", 1.0)

        # Find applicable dirty air zone
        effect = 0.0
        for zone in self.dirty_air_zones.values():
            if zone["range"][0] <= gap_ahead <= zone["range"][1]:
                effect = zone["effect"] * track_sensitivity
                break

        # Calculate time loss
        time_loss = base_lap_time * effect

        logger.debug(
            f"Dirty air calculation:\n"
            f"  Gap ahead: {gap_ahead:.3f}s\n"
            f"  Track sensitivity: {track_sensitivity:.3f}\n"
            f"  Effect multiplier: {effect:.3f}\n"
            f"  Time loss: {time_loss:.3f}s"
        )

        return time_loss

    def _calculate_drs_effect(
        self, gap_ahead: float, base_lap_time: float, track_chars: Dict[str, Any]
    ) -> float:
        """
        Calculate the time gain from DRS if available.
        Returns the time effect in seconds (negative for time gain).
        """
        # Check if gap is within DRS detection range
        if gap_ahead > self.drs_detection_range:
            return 0.0

        # Get track-specific DRS effectiveness (default to 1.0)
        track_drs_effect = track_chars.get("drs_effectiveness", 1.0)

        # Get number of DRS zones for this track
        drs_zones = track_chars.get("drs_zones", self.drs_zones_per_lap)

        # Calculate DRS effect scaled by number of zones and track effectiveness
        time_gain = base_lap_time * self.drs_effect * track_drs_effect * (drs_zones / 2)

        logger.debug(
            f"DRS calculation:\n"
            f"  Gap ahead: {gap_ahead:.3f}s\n"
            f"  DRS zones: {drs_zones}\n"
            f"  Track effectiveness: {track_drs_effect:.3f}\n"
            f"  Time gain: {time_gain:.3f}s"
        )

        return time_gain

    def _calculate_fuel_load_effect(
        self, current_lap: int, base_lap_time: float
    ) -> float:
        """
        Calculate the time loss due to fuel load.
        Returns the time penalty in seconds.
        """
        # Calculate remaining fuel based on lap number
        fuel_consumed = self.fuel_consumption * (current_lap - 1)
        current_fuel = max(0, self.initial_fuel_load - fuel_consumed)

        # Calculate time penalty (increases with more fuel)
        time_loss = (current_fuel / 10) * self.fuel_weight_penalty

        logger.debug(
            f"Fuel load calculation:\n"
            f"  Current lap: {current_lap}\n"
            f"  Remaining fuel: {current_fuel:.1f}kg\n"
            f"  Time loss: {time_loss:.3f}s"
        )

        return time_loss

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the agent."""
        return {
            "name": self.name,
            "description": "Calculates the effects of gaps between cars, including dirty air, DRS, and fuel load impacts.",
            "inputs": [
                "gap_to_ahead",
                "current_lap",
                "base_lap_time",
                "track_characteristics",
                "driver_characteristics",
            ],
            "outputs": [
                "lap_time_delta",
                "dirty_air_effect",
                "drs_active",
                "drs_effect",
                "fuel_effect",
            ],
        }
