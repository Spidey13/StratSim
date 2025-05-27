# src/agents/vehicle_dynamics_agent.py

from typing import Dict, Any
import numpy as np
import logging
from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)


class VehicleDynamicsAgent(BaseAgent):
    """
    Agent responsible for calculating vehicle performance and lap times
    based on car setup, tire condition, and track state.
    """

    def __init__(self, name: str = "VehicleDynamicsAgent"):
        """
        Initialize the VehicleDynamicsAgent.

        Args:
            name: The name of the agent.
        """
        super().__init__(name)

        # Base lap time for a perfectly optimized lap (seconds)
        self.base_lap_time = (
            80.0  # Starting point for calculations (typical F1 mid-length circuit)
        )

        # Segment time distributions (percentage of lap time)
        self.segment_distribution = {
            "S1": 0.33,  # Sector 1 - typically more high-speed
            "S2": 0.33,  # Sector 2 - mixed corners
            "S3": 0.34,  # Sector 3 - typically more technical
        }

        # Corner types and their grip sensitivity (reduced impact)
        self.corner_sensitivity = {
            "high_speed": 0.4,  # High speed corners most affected by grip
            "medium_speed": 0.3,  # Medium speed corners
            "low_speed": 0.2,  # Low speed corners
            "straight": 0.1,  # Minimal grip impact on straights
        }

        # Minimum sector time multipliers (tightened range)
        self.min_sector_multiplier = {
            "S1": 0.85,  # Can't go faster than 85% of base sector time
            "S2": 0.85,  # More consistent across sectors
            "S3": 0.85,
        }

        # Performance factors (reduced sensitivity)
        self.perf_factors = {
            "tire_temp_optimal": 90.0,  # Optimal tire temp (C)
            "track_temp_optimal": 35.0,  # Optimal track temp (C)
            "temp_sensitivity": 0.001,  # Reduced temperature sensitivity
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate lap time and performance metrics based on current conditions.

        Args:
            inputs: Dictionary containing:
                - tire_condition: Dict with wear%, grip%, compound
                - track_state: Dict with temperature, grip, weather
                - car_setup: Dict with wing levels, suspension, etc.

        Returns:
            Dictionary containing:
                - predicted_lap_time: Float (seconds)
                - sector_times: Dict of sector predictions
                - performance_factors: Dict of contributing factors
        """
        # Extract tire data
        tire_data = inputs.get("tire_condition", {})
        tire_wear = tire_data.get("wear", 0.0)
        grip_level = tire_data.get("grip", 1.0)
        compound = tire_data.get("compound", "MEDIUM")

        # Extract track state
        track_state = inputs.get("track_state", {})
        track_temp = track_state.get(
            "track_temp", self.perf_factors["track_temp_optimal"]
        )
        track_grip = track_state.get("grip", 1.0)

        # Calculate base performance factors
        tire_performance = self._calculate_tire_performance(
            tire_wear, grip_level, compound
        )
        track_performance = self._calculate_track_performance(track_temp, track_grip)

        # Calculate sector times with compound-specific and corner-type effects
        sector_times = {}
        total_time = 0.0

        for sector, distribution in self.segment_distribution.items():
            base_sector_time = self.base_lap_time * distribution

            # Apply corner-specific grip sensitivity
            if sector == "S1":  # High speed sector
                grip_impact = self._calculate_grip_impact(grip_level, "high_speed")
            elif sector == "S2":  # Mixed sector
                grip_impact = self._calculate_grip_impact(grip_level, "medium_speed")
            else:  # Technical sector
                grip_impact = self._calculate_grip_impact(grip_level, "low_speed")

            # Calculate sector performance
            sector_perf = tire_performance * track_performance * grip_impact

            # Apply minimum sector time limit
            min_time = base_sector_time * self.min_sector_multiplier[sector]
            sector_time = max(min_time, base_sector_time / sector_perf)

            sector_times[sector] = sector_time
            total_time += sector_time

            # Log detailed sector calculation
            logger.debug(
                f"Sector {sector} calculation:\n"
                f"  Base time: {base_sector_time:.3f}s\n"
                f"  Grip impact: {grip_impact:.3f}\n"
                f"  Final time: {sector_time:.3f}s"
            )

        # Add some minor random variation (±0.5%)
        variation = np.random.normal(0, 0.005)
        total_time *= 1 + variation

        # Ensure lap time never goes below theoretical minimum
        min_possible_time = self.base_lap_time * 0.75
        total_time = max(min_possible_time, total_time)

        # Log overall performance factors
        logger.debug(
            f"Lap time calculation:\n"
            f"  Base time: {self.base_lap_time:.3f}s\n"
            f"  Tire performance: {tire_performance:.3f}\n"
            f"  Track performance: {track_performance:.3f}\n"
            f"  Grip level: {grip_level:.3f}\n"
            f"  Final time: {total_time:.3f}s"
        )

        return {
            "predicted_lap_time": total_time,
            "sector_times": sector_times,
            "performance_factors": {
                "tire_performance": tire_performance,
                "track_performance": track_performance,
                "grip_level": grip_level,
                "variation": variation,
            },
        }

    def _calculate_tire_performance(
        self, wear: float, grip: float, compound: str
    ) -> float:
        """
        Calculate tire performance factor based on wear, grip and compound.
        Returns a multiplier where 1.0 is optimal performance.
        """
        # Base compound performance (realistic F1 deltas)
        compound_base = {
            "SOFT": 1.01,  # ~0.8s faster than medium
            "MEDIUM": 1.0,  # Baseline
            "HARD": 0.99,  # ~0.8s slower than medium
            "INTERMEDIATE": 0.90,
            "WET": 0.85,
        }.get(compound, 1.0)

        # Non-linear wear effect (more gradual impact)
        wear_effect = 1.0 - (wear / 100) ** 1.2

        # Grip level has linear impact on performance
        grip_effect = 0.7 + (0.3 * grip)  # Max 30% impact from grip

        # Combine factors with balanced weighting
        performance = compound_base * (0.4 * wear_effect + 0.6 * grip_effect)

        # Add cliff effect when wear is very high (>85%)
        if wear > 85:
            cliff_factor = 1.0 - ((wear - 85) / 15) ** 1.5  # More gradual drop-off
            performance *= cliff_factor

        # Log detailed calculation
        logger.debug(
            f"Tire performance calculation:\n"
            f"  Compound: {compound} (base: {compound_base:.3f})\n"
            f"  Wear: {wear:.1f}% (effect: {wear_effect:.3f})\n"
            f"  Grip: {grip:.3f} (effect: {grip_effect:.3f})\n"
            f"  Final performance: {performance:.3f}"
        )

        return performance

    def _calculate_track_performance(
        self, track_temp: float, track_grip: float
    ) -> float:
        """
        Calculate track-related performance factor.
        Returns a multiplier where 1.0 is optimal performance.
        """
        # Temperature effect (linear falloff from optimal)
        temp_delta = abs(track_temp - self.perf_factors["track_temp_optimal"])
        temp_effect = max(
            0.85, 1.0 - (temp_delta * self.perf_factors["temp_sensitivity"])
        )

        # Track grip has linear impact (max 20% variation)
        grip_effect = 0.8 + (0.2 * track_grip)

        # Log detailed calculation
        logger.debug(
            f"Track performance calculation:\n"
            f"  Track temp: {track_temp:.1f}C (delta: {temp_delta:.1f}C, effect: {temp_effect:.3f})\n"
            f"  Track grip: {track_grip:.3f} (effect: {grip_effect:.3f})"
        )

        # Combine effects (weighted average favoring grip)
        return 0.3 * temp_effect + 0.7 * grip_effect

    def _calculate_grip_impact(self, grip_level: float, corner_type: str) -> float:
        """
        Calculate how grip affects different types of corners.
        """
        sensitivity = self.corner_sensitivity.get(corner_type, 0.5)

        # Non-linear grip impact
        grip_impact = 1.0 - (sensitivity * (1.0 - grip_level) ** 1.5)

        # Ensure reasonable bounds
        return max(0.5, min(1.2, grip_impact))

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the agent."""
        return {
            "name": self.name,
            "description": "Calculates vehicle speeds and differentials based on grip and other factors.",
            "inputs": [
                "tire_condition",
                "track_state",
                "car_setup",
            ],
            "outputs": [
                "predicted_lap_time",
                "sector_times",
                "performance_factors",
            ],
        }
