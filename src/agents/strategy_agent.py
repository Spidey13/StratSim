"""
Strategy planning agent that makes pit stop and tire compound decisions.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyAgent(BaseAgent):
    """
    Agent responsible for making race strategy decisions,
    including when to pit and which tire compounds to use.
    """

    def __init__(self, name: str = "StrategyAgent"):
        """
        Initialize the strategy planning agent.

        Args:
            name: Agent name
        """
        super().__init__(name)

        # Default pit window size (laps)
        self.pit_window_size = 5

        # Compound performance characteristics (relative pace)
        self.compound_pace = {
            "SOFT": 1.0,  # Baseline (fastest)
            "MEDIUM": 1.01,  # 1% slower than soft
            "HARD": 1.02,  # 2% slower than soft
            "INTERMEDIATE": 1.05,  # 5% slower in dry, fastest in light rain
            "WET": 1.10,  # 10% slower in dry, fastest in heavy rain
        }

        # Tire life characteristics (estimated laps at good performance)
        self.tire_life = {
            "SOFT": 20,
            "MEDIUM": 30,
            "HARD": 40,
            "INTERMEDIATE": 25,
            "WET": 30,
        }

        # Strategy state
        self.planned_stops = {}  # Planned pit stops by driver
        self.last_decisions = {}  # Last decision made for each driver

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process race information to make strategy decisions.

        Args:
            inputs: Dictionary containing:
                - lap: Current lap number
                - total_laps: Total race laps
                - driver_id: Driver identifier
                - driver_state: Current driver state
                - weather: Current weather conditions
                - track_state: Current track conditions

        Returns:
            Dictionary containing:
                - pit_decision: Boolean indicating whether to pit
                - new_compound: Compound to change to if pitting
                - expected_benefit: Expected time benefit of decision
                - reasoning: Explanation of the decision
        """
        # Extract inputs
        lap = inputs.get("lap", 0)
        total_laps = inputs.get("total_laps", 0)
        driver_id = inputs.get("driver_id", "unknown")
        driver_state = inputs.get("driver_state", {})
        weather = inputs.get("weather", {})
        track_state = inputs.get("track_state", {})

        # Get current tire information
        current_compound = driver_state.get("current_compound", "MEDIUM")
        tire_age = driver_state.get("tire_age", 0)

        # Calculate wear-based pit threshold
        wear_threshold = self._calculate_wear_threshold(current_compound, total_laps)

        # Check if weather conditions require immediate pit
        weather_pit, weather_compound = self._check_weather_pit(
            current_compound, weather, track_state
        )

        # Calculate optimal strategy
        if weather_pit:
            # Weather-based pit decision overrides
            pit_decision = True
            new_compound = weather_compound
            benefit = self._calculate_compound_benefit(
                current_compound, new_compound, tire_age, weather
            )
            reasoning = f"Weather conditions require switch to {new_compound}"
        else:
            # Normal strategy calculations
            pit_decision, new_compound, benefit, reasoning = self._calculate_strategy(
                lap, total_laps, driver_state, weather, track_state, wear_threshold
            )

        # Store decision for this driver
        self.last_decisions[driver_id] = {
            "lap": lap,
            "pit_decision": pit_decision,
            "current_compound": current_compound,
            "new_compound": new_compound,
            "tire_age": tire_age,
        }

        # Update planned stops if needed
        if not driver_id in self.planned_stops:
            self.planned_stops[driver_id] = []

        # If we're pitting, record it
        if pit_decision:
            self.planned_stops[driver_id].append(
                {
                    "lap": lap,
                    "old_compound": current_compound,
                    "new_compound": new_compound,
                    "reason": reasoning,
                }
            )

        return {
            "pit_decision": pit_decision,
            "new_compound": new_compound,
            "expected_benefit": benefit,
            "reasoning": reasoning,
        }

    def _calculate_wear_threshold(self, compound: str, total_laps: int) -> float:
        """
        Calculate the tire age threshold for pitting based on compound.

        Args:
            compound: Tire compound
            total_laps: Total race laps

        Returns:
            Threshold in laps
        """
        base_life = self.tire_life.get(compound, 25)

        # Adjust threshold based on race length
        # For shorter races, can push tires longer as percentage of race
        if total_laps < 40:
            return base_life * 0.9  # Push closer to limit
        elif total_laps > 60:
            return base_life * 0.8  # More conservative strategy
        else:
            return base_life * 0.85  # Standard threshold

    def _check_weather_pit(
        self,
        current_compound: str,
        weather: Dict[str, Any],
        track_state: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check if weather conditions necessitate a pit stop.

        Args:
            current_compound: Current tire compound
            weather: Weather conditions
            track_state: Track conditions

        Returns:
            Tuple of (pit_decision, new_compound)
        """
        condition = weather.get("condition", "Dry")
        rainfall = weather.get("rainfall", 0)

        # Check for rain when on slicks
        if rainfall > 0 and current_compound in ["SOFT", "MEDIUM", "HARD"]:
            if rainfall >= 2:
                # Heavy rain, need wet tires
                return True, "WET"
            else:
                # Light rain, need intermediates
                return True, "INTERMEDIATE"

        # Check for drying track when on rain tires
        if rainfall == 0 and current_compound in ["INTERMEDIATE", "WET"]:
            if track_state.get("dry_line", False):
                # Track has a dry line, switch to slicks
                return True, "MEDIUM"  # Default to medium for safety

        # No weather-based pit needed
        return False, current_compound

    def _calculate_strategy(
        self,
        lap: int,
        total_laps: int,
        driver_state: Dict[str, Any],
        weather: Dict[str, Any],
        track_state: Dict[str, Any],
        wear_threshold: float,
    ) -> Tuple[bool, str, float, str]:
        """
        Calculate optimal strategy decision.

        Args:
            lap: Current lap
            total_laps: Total race laps
            driver_state: Driver state
            weather: Weather conditions
            track_state: Track conditions
            wear_threshold: Tire wear threshold

        Returns:
            Tuple of (pit_decision, new_compound, expected_benefit, reasoning)
        """
        current_compound = driver_state.get("current_compound", "MEDIUM")
        tire_age = driver_state.get("tire_age", 0)

        # Don't pit on first or last few laps (unless emergency)
        if lap < 3 or lap > total_laps - 3:
            return False, current_compound, 0.0, "Too early or too late to pit"

        # Calculate remaining laps
        remaining_laps = total_laps - lap

        # Check if tire age exceeds threshold
        if tire_age >= wear_threshold:
            # Determine optimal compound for remaining laps
            new_compound = self._select_optimal_compound(remaining_laps, weather)
            benefit = self._calculate_compound_benefit(
                current_compound, new_compound, tire_age, weather
            )
            return True, new_compound, benefit, "Tire age exceeded wear threshold"

        # Check if we should make a strategic stop
        if self._should_make_strategic_stop(lap, total_laps, driver_state, weather):
            new_compound = self._select_optimal_compound(remaining_laps, weather)
            benefit = self._calculate_compound_benefit(
                current_compound, new_compound, tire_age, weather
            )
            return (
                True,
                new_compound,
                benefit,
                "Strategic pit stop for optimal race time",
            )

        # Check if pit stop is within optimal window
        # This is a simplified model that assumes 2-stop strategy for medium+ length races
        if 0.3 <= lap / total_laps <= 0.4 or 0.6 <= lap / total_laps <= 0.7:
            # We're in a typical pit window
            if tire_age >= wear_threshold * 0.7:  # 70% of threshold
                new_compound = self._select_optimal_compound(remaining_laps, weather)
                benefit = self._calculate_compound_benefit(
                    current_compound, new_compound, tire_age, weather
                )
                return (
                    True,
                    new_compound,
                    benefit,
                    "Preemptive pit within optimal window",
                )

        # Default to no pit
        return False, current_compound, 0.0, "Current strategy optimal"

    def _select_optimal_compound(
        self, remaining_laps: int, weather: Dict[str, Any]
    ) -> str:
        """
        Select the optimal compound based on remaining laps and conditions.

        Args:
            remaining_laps: Number of laps remaining
            weather: Weather conditions

        Returns:
            Optimal compound
        """
        rainfall = weather.get("rainfall", 0)

        # If it's raining, choose appropriate rain tire
        if rainfall > 0:
            if rainfall >= 2:
                return "WET"
            else:
                return "INTERMEDIATE"

        # Dry conditions, choose based on remaining laps
        if remaining_laps <= 15:
            return "SOFT"  # Aggressive strategy for short stint
        elif remaining_laps <= 30:
            return "MEDIUM"  # Balanced choice for medium stint
        else:
            return "HARD"  # Conservative choice for long stint

    def _calculate_compound_benefit(
        self,
        current_compound: str,
        new_compound: str,
        tire_age: int,
        weather: Dict[str, Any],
    ) -> float:
        """
        Calculate the expected time benefit of switching compounds.

        Args:
            current_compound: Current tire compound
            new_compound: New tire compound
            tire_age: Current tire age
            weather: Weather conditions

        Returns:
            Expected benefit in seconds per lap
        """
        # Base pace difference between compounds
        current_pace = self.compound_pace.get(current_compound, 1.0)
        new_pace = self.compound_pace.get(new_compound, 1.0)

        # Adjust based on weather conditions
        rainfall = weather.get("rainfall", 0)
        if rainfall > 0:
            # Rain tires have advantage in wet conditions
            if current_compound in ["SOFT", "MEDIUM", "HARD"]:
                # Slicks in rain
                current_pace *= 1.0 + rainfall * 0.1  # 10% penalty per rain unit

            if new_compound == "WET" and rainfall >= 2:
                # Wet tires in heavy rain
                new_pace *= 0.9  # 10% advantage
            elif new_compound == "INTERMEDIATE" and 0 < rainfall < 2:
                # Intermediates in light rain
                new_pace *= 0.85  # 15% advantage
        else:
            # Dry conditions
            if current_compound in ["INTERMEDIATE", "WET"]:
                # Rain tires in dry
                current_pace *= 1.1  # 10% penalty

        # Account for tire wear
        # Simplified model: performance degrades linearly after 50% of tire life
        tire_life = self.tire_life.get(current_compound, 25)
        if tire_age > tire_life * 0.5:
            wear_factor = (tire_age - tire_life * 0.5) / (tire_life * 0.5)
            current_pace *= 1.0 + wear_factor * 0.05  # Up to 5% penalty at full wear

        # Calculate lap time difference (use 90s as baseline lap time)
        baseline_lap = 90.0  # seconds
        current_lap_time = baseline_lap * current_pace
        new_lap_time = baseline_lap * new_pace

        # Return benefit (negative means improvement)
        return new_lap_time - current_lap_time

    def _should_make_strategic_stop(
        self,
        lap: int,
        total_laps: int,
        driver_state: Dict[str, Any],
        weather: Dict[str, Any],
    ) -> bool:
        """
        Determine if a strategic stop would be beneficial.

        Args:
            lap: Current lap
            total_laps: Total race laps
            driver_state: Driver state
            weather: Weather conditions

        Returns:
            Boolean indicating if a strategic stop is beneficial
        """
        # Check if we already made a stop in the last few laps
        pit_stops = driver_state.get("pit_stops", [])
        if pit_stops:
            last_stop_lap = pit_stops[-1].get("lap", 0)
            if lap - last_stop_lap < 10:
                return False  # Too soon since last stop

        # Check for upcoming weather changes
        forecast = weather.get("forecast", [])
        for i in range(1, 6):  # Look ahead 5 laps
            future_lap = lap + i
            if future_lap > total_laps:
                break

            # Find forecast for this lap
            lap_forecast = next(
                (f for f in forecast if f.get("lap") == future_lap), None
            )
            if lap_forecast and lap_forecast.get("condition") != weather.get(
                "condition"
            ):
                # Weather change coming, consider strategic stop
                return True

        # Check if we're approaching a standard pit window
        # This assumes a typical 2-stop strategy
        if total_laps > 40:  # Only for longer races
            first_window = total_laps // 3
            second_window = (total_laps * 2) // 3

            # Check if we're approaching either window
            if (first_window - self.pit_window_size <= lap <= first_window) or (
                second_window - self.pit_window_size <= lap <= second_window
            ):
                # We're in a strategic window, consider tire age
                tire_age = driver_state.get("tire_age", 0)
                current_compound = driver_state.get("current_compound", "MEDIUM")
                wear_threshold = self._calculate_wear_threshold(
                    current_compound, total_laps
                )

                # If tires are at least 60% worn, it makes sense to stop in window
                if tire_age >= wear_threshold * 0.6:
                    return True

        return False
