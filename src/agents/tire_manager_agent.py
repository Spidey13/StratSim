"""
Tire manager agent that tracks tire state and provides strategic guidance.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import numpy as np
import json

from .base_agent import BaseAgent
from .tire_wear_agent import TireWearAgent
from .grip_model_agent import GripModelAgent
from .tire_temperature_agent import TireTemperatureAgent

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TireManagerAgent(BaseAgent):
    """
    Agent responsible for tracking tire state, updating wear,
    and providing strategic guidance for pit stops based on tire condition.
    """

    def __init__(
        self,
        name: str = "TireManagerAgent",
        tire_wear_model_path: Optional[str] = None,
    ):
        """
        Initialize the tire manager agent.

        Args:
            name: Agent name
            tire_wear_model_path: Optional path to tire wear model
        """
        super().__init__(name)

        # Initialize sub-agents
        self.tire_wear_agent = TireWearAgent(
            name="TireWearAgent", model_path=tire_wear_model_path
        )
        self.grip_model_agent = GripModelAgent(name="GripModelAgent")
        self.tire_temp_agent = TireTemperatureAgent(name="TireTemperatureAgent")

        # Initialize empty tire state
        self.reset_state()

        # Define cliff thresholds for each compound (percentage wear)
        self.cliff_thresholds = {
            "SOFT": 85.0,  # Increased from 65.0 - softs should last ~15-20 laps
            "MEDIUM": 90.0,  # Increased from 75.0 - mediums should last ~25-30 laps
            "HARD": 95.0,  # Increased from 85.0 - hards should last ~35-40 laps
            "INTERMEDIATE": 70.0,  # Wet conditions specific
            "WET": 60.0,  # Wet conditions specific
        }

        # Warning thresholds (percentage of cliff threshold)
        self.warning_threshold = 0.9  # Increased from 0.8 to allow longer stints

        # Threshold for what's considered a "short stint" for tire choice logic
        self.short_stint_threshold_laps = (
            10  # Reduced from 15 to prevent too early stops
        )

        # Defines how many total accumulated seconds of lap time degradation
        # equates to a tire being 100% worn
        self.total_degradation_s_for_100_percent_wear = {
            "SOFT": 5.0,  # Increased for more gradual wear (was 2.5)
            "MEDIUM": 6.0,  # Increased for more gradual wear (was 3.0)
            "HARD": 7.0,  # Increased for more gradual wear (was 3.5)
            "INTERMEDIATE": 4.0,  # Increased from 2.0
            "WET": 3.0,  # Increased from 1.5
        }
        self.default_total_degradation_s_for_100_wear = 5.0  # Increased from 2.5

        # Defines the maximum percentage of grip lost when a tire is 100% worn
        self.max_grip_loss_at_100_percent_wear = (
            0.7  # Increased from 0.6 for more impact at end of life
        )

        # Define compound-specific cliff points for grip calculation
        self.cliff_points = {
            "SOFT": 65,  # Decreased from 80 for earlier performance drop
            "MEDIUM": 75,  # Decreased from 85 for earlier performance drop
            "HARD": 85,  # Decreased from 90 for earlier performance drop
            "INTERMEDIATE": 50,
            "WET": 40,
        }
        self.default_cliff_point = 75  # Decreased from 85

        # Debug counters
        self._debug_counter = 0
        self._debug_log_frequency = 1  # Log every lap

    def reset_state(self) -> None:
        """Reset the tire state to default values."""
        self.state = {
            "driver_id": None,
            "current_compound": None,
            "tire_age": 0,
            "tire_wear": 0.0,
            "tire_temp": 80.0,  # Add initial tire temperature
            "grip_level": 1.0,
            "in_optimal_window": True,
            "tire_health": "GOOD",
            "prev_lap_degradation_s_for_wear_model": 0.0,
            "last_wear_rate": 0.0,
            "simulation_year": 2024,
            "pit_recommendation": None,
            "compounds_used": [],
            "tire_history": [],  # List of (compound, age) tuples for all stints
        }

    def _log_degradation_factors(
        self,
        inputs: Dict[str, Any],
        wear_result: Dict[str, Any],
        degradation_s: float,
        wear_increment: float,
        grip_result: Dict[str, Any],
    ) -> None:
        """Log detailed information about tire degradation and grip factors."""
        self._debug_counter += 1

        # Only log every _debug_log_frequency laps
        if self._debug_counter % self._debug_log_frequency == 0:
            # Detailed debug logging for tire state
            logger.debug(
                f"\n=== Tire State Analysis (Lap {inputs.get('current_lap', 0)}) ===\n"
                f"Compound: {self.state['current_compound']}, Age: {self.state['tire_age']}\n"
                f"Current Wear: {self.state['tire_wear']:.2f}%, Wear Increment: {wear_increment:.3f}%\n"
                f"Tire Temp: {self.state['tire_temp']:.1f}°C\n"
                f"Current Grip: {self.state['grip_level']:.3f}\n"
                f"Raw Degradation: {degradation_s:.3f}s\n"
                f"=== Grip Calculation Details ===\n"
                f"Track Evolution: {grip_result.get('track_evolution', 'N/A')}\n"
                f"Temperature Performance: {grip_result.get('temp_performance', {})}\n"
                f"Wear Grip: {grip_result.get('wear_grip', {})}\n"
                f"=== Weather Impact ===\n"
                f"Weather: {inputs.get('weather', {})}\n"
            )

    def _calculate_wear_from_degradation(
        self,
        degradation_s: float,
        compound: str,
        lap_in_stint: int,
        driver_chars: Dict[str, float] = None,
        grip_level: float = 1.0,
        temp_status: Dict[str, Any] = None,
    ) -> float:
        """
        Convert model's degradation prediction to wear percentage more accurately.
        Now includes grip and temperature effects on wear rate.
        """
        # Get driver characteristics or use defaults
        if driver_chars is None:
            driver_chars = {}
        tire_management = driver_chars.get("tire_management", 1.0)
        aggression = driver_chars.get("aggression", 1.0)
        consistency = driver_chars.get("consistency", 1.0)

        # Compound-specific base wear rates (per lap)
        base_wear_rates = {
            "SOFT": 0.8 * (1.5 - tire_management * 0.3),  # More affected by management
            "MEDIUM": 0.6 * (1.4 - tire_management * 0.25),
            "HARD": 0.4 * (1.3 - tire_management * 0.2),  # Less affected by management
            "INTERMEDIATE": 0.7,
            "WET": 0.9,
        }
        base_wear = base_wear_rates.get(compound, 0.6)

        # Aggressive driving increases wear linearly and then exponentially
        aggression_delta = max(0, aggression - 1.0)  # Only consider excess aggression
        aggression_factor = (
            1.0 + aggression_delta + (aggression_delta * aggression_delta)
        )
        base_wear *= aggression_factor

        # Stint phase effects
        if lap_in_stint <= 2:  # Warm-up phase
            stint_factor = 0.7 + (lap_in_stint * 0.15 * aggression)
        elif lap_in_stint <= 5:  # Early stint
            stint_factor = 1.0 + ((lap_in_stint - 2) * 0.05 * aggression)
        else:  # Mid to late stint
            stint_factor = 1.2 + ((lap_in_stint - 5) * 0.02 * aggression)

        # Cap stint factor based on tire management skill
        max_stint_factor = 2.0 - (tire_management * 0.5)
        stint_factor = min(stint_factor, max_stint_factor)

        # Grip effects on wear - more sliding = more wear
        grip_wear_multiplier = 1.0 + (1.0 - min(1.0, max(0.3, grip_level)))

        # Temperature effects
        temp_multiplier = 1.0
        if temp_status:
            if temp_status.get("is_overheating", False):
                temp_multiplier = 1.3 - (tire_management * 0.1)
            elif temp_status.get("is_cold", False):
                temp_multiplier = 0.85

        # Calculate base wear increment
        wear_increment = (
            base_wear * stint_factor * grip_wear_multiplier * temp_multiplier
        )

        # Add consistency-based variation
        variation_scale = 0.15 * (2.0 - consistency)
        variation = np.random.normal(0, variation_scale)
        wear_increment *= max(0.5, min(1.5, 1.0 + variation))  # Limit variation impact

        # Ensure wear stays within realistic bounds
        min_wear = base_wear * 0.5  # Minimum wear can't be too low
        max_wear = base_wear * 3.0 * aggression  # Maximum wear scales with aggression
        wear_increment = max(min_wear, min(max_wear, wear_increment))

        return float(wear_increment)  # Ensure we return a float

    def _calculate_grip_from_wear(self, wear_pct: float, compound: str) -> float:
        """
        Calculate grip level based on wear percentage with compound-specific cliff points.
        """
        # Ensure inputs are real numbers and within bounds
        wear_pct = float(max(0.0, min(100.0, wear_pct)))
        cliff_point = float(self.cliff_points.get(compound, self.default_cliff_point))

        # Store previous grip level for validation
        prev_grip = float(max(0.3, min(1.0, self.state.get("grip_level", 1.0))))

        # More gradual initial grip loss using linear interpolation
        if wear_pct < cliff_point * 0.5:  # First half of tire life
            wear_ratio = wear_pct / (cliff_point * 0.5)
            grip_loss = wear_ratio * 0.3  # Linear loss up to 30%
        elif wear_pct < cliff_point:  # Second half until cliff
            wear_ratio = (wear_pct - (cliff_point * 0.5)) / (cliff_point * 0.5)
            grip_loss = 0.3 + (wear_ratio * 0.3)  # Linear loss from 30% to 60%
        else:  # After cliff point
            base_loss = 0.6  # 60% loss at cliff
            extra_wear_ratio = min(1.0, (wear_pct - cliff_point) / (100 - cliff_point))
            cliff_loss = extra_wear_ratio * 0.3  # Additional 30% loss post-cliff
            grip_loss = base_loss + cliff_loss

        # Calculate final grip level
        grip_level = 1.0 - grip_loss

        # Add small random variation (±0.5%)
        grip_variation = np.random.normal(0, 0.005)
        grip_level = grip_level * (1.0 + max(-0.005, min(0.005, grip_variation)))

        # Ensure grip never increases and stays within bounds
        grip_level = float(max(0.3, min(prev_grip, min(1.0, grip_level))))

        return grip_level

    def _calculate_initial_grip(
        self,
        wear_pct: float,
        compound: str,
        tire_temp: float,
        track_temp: float = 30.0,
        weather: Dict[str, Any] = None,
    ) -> float:
        """
        Calculate an initial grip estimate based on current state.
        This is used to influence wear calculation before final grip is determined.
        """
        # Debug logging for input parameters
        logger.debug(
            f"\n=== Initial Grip Calculation Inputs ===\n"
            f"Current Wear: {wear_pct:.2f}%\n"
            f"Compound: {compound}\n"
            f"Tire Temperature: {tire_temp:.1f}°C\n"
            f"Track Temperature: {track_temp:.1f}°C\n"
            f"Weather: {weather}\n"
        )

        # Start with base grip of 1.0
        initial_grip = 1.0

        # Apply basic wear effect
        cliff_point = self.cliff_points.get(compound, self.default_cliff_point)
        if wear_pct < cliff_point:
            wear_reduction = (wear_pct / cliff_point) * 0.3
        else:
            base_reduction = 0.3
            post_cliff = (wear_pct - cliff_point) / (100 - cliff_point)
            wear_reduction = base_reduction + (post_cliff * 0.4)

        initial_grip *= 1.0 - wear_reduction

        # Debug logging for wear impact
        logger.debug(
            f"\n=== Wear Impact on Initial Grip ===\n"
            f"Cliff Point: {cliff_point}\n"
            f"Wear Reduction: {wear_reduction:.3f}\n"
            f"Grip After Wear: {initial_grip:.3f}\n"
        )

        # Apply basic temperature effect
        temp_windows = {
            "SOFT": (85, 105),
            "MEDIUM": (80, 100),
            "HARD": (75, 95),
            "INTERMEDIATE": (50, 90),
            "WET": (40, 70),
        }
        optimal_range = temp_windows.get(compound, (80, 100))

        if tire_temp < optimal_range[0]:
            temp_effect = 0.8 + (0.2 * tire_temp / optimal_range[0])
        elif tire_temp > optimal_range[1]:
            temp_effect = 0.8 + (0.2 * optimal_range[1] / tire_temp)
        else:
            temp_effect = 1.0

        initial_grip *= temp_effect

        # Debug logging for temperature impact
        logger.debug(
            f"\n=== Temperature Impact on Initial Grip ===\n"
            f"Optimal Range: {optimal_range}\n"
            f"Temperature Effect: {temp_effect:.3f}\n"
            f"Grip After Temperature: {initial_grip:.3f}\n"
        )

        # Apply weather effects if available
        weather_effect = 1.0
        if weather:
            if weather.get("rainfall", 0) > 0:
                if compound not in ["WET", "INTERMEDIATE"]:
                    weather_effect = 0.7

        initial_grip *= weather_effect

        # Debug logging for weather impact
        logger.debug(
            f"\n=== Weather Impact on Initial Grip ===\n"
            f"Weather Effect: {weather_effect:.3f}\n"
            f"Final Initial Grip: {initial_grip:.3f}\n"
        )

        return max(0.3, min(1.0, initial_grip))

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to update tire state and provide recommendations.

        Args:
            inputs: Dictionary containing:
                - current_lap: Current lap number
                - race_lap: Current race lap (can be different in qualifying)
                - circuit_id: ID of the circuit
                - driver_id: Driver ID
                - weather: Weather conditions
                - compound: New compound if pitting, None otherwise
                - is_pit_lap: Boolean indicating if this is a pit lap
                - laps_remaining: Number of laps remaining in race
                - strategy: Current race strategy
                - characteristics: Driver characteristics (optional)

        Returns:
            Dictionary containing updated tire state and recommendations
        """
        current_lap = inputs.get("current_lap", 0)
        driver_id = inputs.get("driver_id", "Unknown")

        # Reset state for new driver or if compound is different
        if self.state.get("driver_id") != driver_id or (
            inputs.get("compound") and inputs.get("is_pit_lap", False)
        ):
            self.reset_state()
            self.state["driver_id"] = driver_id

        # Initialize state if needed
        if self.state["current_compound"] is None:
            self.state["current_compound"] = inputs.get("current_compound", "MEDIUM")
            self.state["tire_age"] = inputs.get("tire_age", 0)
            self.state["tire_wear"] = inputs.get("tire_wear", 0.0)
            self.state["compounds_used"] = inputs.get(
                "compounds_used", [self.state["current_compound"]]
            )
            self.state["tire_history"] = inputs.get(
                "tire_history", [(self.state["current_compound"], 0)]
            )

        # Debug logging for tire state
        logger.debug(
            f"\n=== Tire State for Driver {driver_id} at Lap {current_lap} ==="
        )
        logger.debug(
            f"Initial state: wear={self.state['tire_wear']:.1f}%, age={self.state['tire_age']}, compound={self.state['current_compound']}"
        )

        # Extract driver characteristics
        driver_chars = inputs.get("characteristics", {})
        if not driver_chars:
            driver_chars = {
                "tire_management": 0.8 + (np.random.random() * 0.4),
                "aggression": 0.8 + (np.random.random() * 0.4),
                "consistency": 0.8 + (np.random.random() * 0.4),
                "wet_weather": 0.8 + (np.random.random() * 0.4),
            }

        # Handle pit stops and compound changes
        is_pit_lap = inputs.get("is_pit_lap", False)
        if is_pit_lap:
            new_compound = inputs.get("compound")
            if new_compound:
                logger.debug(
                    f"PIT STOP - Changing from {self.state['current_compound']} to {new_compound}"
                )
                logger.debug(
                    f"Pre-pit state: wear={self.state['tire_wear']:.1f}%, age={self.state['tire_age']}"
                )

                # Save current tire state to history before changing
                if self.state["current_compound"] is not None:
                    self.state["tire_history"].append(
                        (self.state["current_compound"], self.state["tire_age"])
                    )
                    if (
                        self.state["current_compound"]
                        not in self.state["compounds_used"]
                    ):
                        self.state["compounds_used"].append(
                            self.state["current_compound"]
                        )

                # Update to new compound
                self.state["current_compound"] = new_compound.upper()
                self.state["tire_age"] = 0
                self.state["tire_wear"] = 0.0
                self.state["prev_lap_degradation_s_for_wear_model"] = 0.0
                self.state["last_wear_rate"] = 0.0

                logger.debug(
                    f"Post-pit state: wear={self.state['tire_wear']:.1f}%, age={self.state['tire_age']}"
                )
        else:
            # Normal lap - increment tire age
            self.state["tire_age"] += 1
            logger.debug(
                f"Normal lap - Incrementing tire age to {self.state['tire_age']}"
            )

        # Calculate tire temperature
        temp_inputs = {
            "compound": self.state["current_compound"],
            "current_temp": self.state["tire_temp"],
            "track_temp": inputs.get("weather", {}).get("track_temp", 30.0),
            "air_temp": inputs.get("weather", {}).get("air_temp", 25.0),
            "rainfall": inputs.get("weather", {}).get("rainfall", 0.0),
            "driver_characteristics": driver_chars,
            "is_pit_lap": is_pit_lap,
            "lap_number": inputs.get("current_lap", 1),
        }
        temp_result = self.tire_temp_agent.process(temp_inputs)
        self.state["tire_temp"] = temp_result["tire_temp"]

        # Calculate initial grip estimate for wear calculation
        initial_grip = self._calculate_initial_grip(
            self.state["tire_wear"],
            self.state["current_compound"],
            self.state["tire_temp"],
            inputs.get("weather", {}).get("track_temp", 30.0),
            inputs.get("weather", {}),
        )

        # Prepare inputs for tire wear agent
        wear_inputs = {
            "Compound": self.state["current_compound"],
            "LapNumberInStint": self.state["tire_age"],
            "Event": inputs.get("circuit_id", "unknown"),
            "Driver": inputs.get("driver_id", "unknown"),
            "Team": inputs.get("team_id", "Unknown_Team"),
            "Year": inputs.get("year", self.state.get("simulation_year", 2024)),
            "weather": inputs.get("weather", {}),
            "SpeedST": inputs.get("SpeedST", 280.0),
            "SpeedI1": inputs.get("SpeedI1", 200.0),
            "SpeedI2": inputs.get("SpeedI2", 150.0),
            "PrevLapTimeDegradation_s": self.state[
                "prev_lap_degradation_s_for_wear_model"
            ],
        }

        # Calculate updated wear
        wear_result = self.tire_wear_agent.process(wear_inputs)
        degradation_this_lap_s = wear_result["estimated_degradation_per_lap_s"]

        # Calculate wear using new method with driver characteristics and initial grip
        wear_increment = self._calculate_wear_from_degradation(
            degradation_this_lap_s,
            self.state["current_compound"],
            self.state["tire_age"],
            driver_chars,
            initial_grip,  # Pass initial grip estimate
            temp_result["temp_status"],  # Pass temperature status
        )

        # Update accumulated wear
        self.state["tire_wear"] = min(100, self.state["tire_wear"] + wear_increment)

        # Use GripModelAgent for final grip calculations with updated wear
        grip_inputs = {
            "compound": self.state["current_compound"],
            "tire_wear": self.state["tire_wear"],
            "tire_temp": self.state["tire_temp"],
            "track_temp": inputs.get("weather", {}).get("track_temp", 30.0),
            "current_lap": inputs.get("current_lap", 1),
            "total_laps": inputs.get("total_laps", 50),
            "weather": inputs.get("weather", {}),
            "driver_characteristics": driver_chars,
        }
        grip_result = self.grip_model_agent.process(grip_inputs)

        # Update state with grip model results
        self.state["grip_level"] = grip_result["grip_level"]
        self.state["in_optimal_window"] = grip_result["in_optimal_window"]
        self.state["track_evolution"] = grip_result["track_evolution"]

        # Log detailed state analysis
        self._log_degradation_factors(
            inputs, wear_result, degradation_this_lap_s, wear_increment, grip_result
        )

        # Store the predicted degradation for next lap
        self.state["prev_lap_degradation_s_for_wear_model"] = degradation_this_lap_s
        self.state["last_wear_rate"] = wear_increment

        # Update tire health status
        self._update_tire_health()

        # Generate pit recommendation
        pit_recommendation = self._generate_pit_recommendation(
            inputs.get("laps_remaining", 0), inputs.get("strategy", {})
        )
        self.state["pit_recommendation"] = pit_recommendation

        return {
            "tire_wear": self.state["tire_wear"],
            "grip_level": self.state["grip_level"],
            "in_optimal_window": self.state["in_optimal_window"],
            "track_evolution": self.state["track_evolution"],
            "tire_age": self.state["tire_age"],
            "current_compound": self.state["current_compound"],
            "pit_recommendation": pit_recommendation,
            "estimated_degradation_s_this_lap": degradation_this_lap_s,
            "wear_increment": wear_increment,
            "tire_health": self.state["tire_health"],
            "tire_temp": self.state["tire_temp"],  # Add tire temperature to output
            "temp_status": {  # Add temperature status information
                "is_overheating": temp_result["is_overheating"],
                "is_cold": temp_result["is_cold"],
                "optimal_temp": temp_result["optimal_temp"],
                "temp_delta": temp_result["temp_delta"],
            },
            "temp_performance": grip_result["temp_performance"],
            "wear_grip": grip_result["wear_grip"],
            "compounds_used": self.state["compounds_used"],
            "tire_history": self.state["tire_history"],
        }

    def _update_tire_health(self) -> None:
        """Update the tire health status based on current wear."""
        compound = self.state["current_compound"]
        wear_pct = self.state["tire_wear"]
        cliff = self.cliff_thresholds.get(compound, 75.0)
        warning = cliff * self.warning_threshold

        if wear_pct < 20:
            health = "new"
        elif wear_pct < warning:
            health = "good"
        elif wear_pct < cliff:
            health = "marginal"
        elif wear_pct < cliff + 10:
            health = "critical"
        else:
            health = "expired"

        self.state["tire_health"] = health

    def _generate_pit_recommendation(
        self, laps_remaining: int, strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate pit stop recommendation based on current tire state and race context.
        """
        should_pit = False
        reason = "No pit recommended"
        urgency = 0  # 0: None, 1: Advisory, 2: Recommended, 3: Urgent
        new_compound_suggestion = None

        # Get driver characteristics
        driver_chars = strategy.get("driver_characteristics", {})
        tire_management = driver_chars.get("tire_management", 1.0)
        aggression = driver_chars.get("aggression", 1.0)
        consistency = driver_chars.get("consistency", 1.0)

        # Adjust thresholds based on driver characteristics
        compound = self.state["current_compound"]
        base_cliff = self.cliff_thresholds.get(compound, 75.0)

        # Better tire management allows pushing closer to the cliff
        cliff_adjustment = (tire_management - 1.0) * 10  # ±10% based on tire management
        # More aggressive drivers need earlier stops
        cliff_adjustment -= (aggression - 1.0) * 5  # ±5% based on aggression
        # More consistent drivers can push closer to limits
        cliff_adjustment += (consistency - 1.0) * 5  # ±5% based on consistency

        # Apply adjustments to cliff threshold
        adjusted_cliff = base_cliff + cliff_adjustment
        # Ensure cliff stays within reasonable bounds
        adjusted_cliff = min(100, max(60, adjusted_cliff))

        # Adjust warning threshold based on driver characteristics
        base_warning = self.warning_threshold
        warning_adjustment = (
            tire_management - 1.0
        ) * 0.05  # ±5% based on tire management
        warning_adjustment -= (aggression - 1.0) * 0.03  # ±3% based on aggression
        adjusted_warning = base_warning + warning_adjustment
        # Ensure warning threshold stays within reasonable bounds
        adjusted_warning = min(0.95, max(0.8, adjusted_warning))

        wear_pct = self.state["tire_wear"]
        warning_point = adjusted_cliff * adjusted_warning

        # Adjust minimum stint length based on driver characteristics
        base_min_stint = 10
        stint_adjustment = int(
            (tire_management - 1.0) * 3
        )  # ±3 laps based on tire management
        stint_adjustment -= int((aggression - 1.0) * 2)  # ±2 laps based on aggression
        min_stint_laps = max(5, base_min_stint + stint_adjustment)

        # Check if tires are beyond or near cliff
        if wear_pct >= adjusted_cliff:
            should_pit = True
            reason = (
                f"Tires past cliff threshold ({wear_pct:.1f}% >= {adjusted_cliff:.1f}%)"
            )
            urgency = 3  # Urgent
        # Check if tires are in the warning zone (approaching cliff)
        elif wear_pct >= warning_point:
            should_pit = True
            reason = (
                f"Tires approaching cliff ({wear_pct:.1f}% >= {warning_point:.1f}%)"
            )
            urgency = 2  # Recommended

        # Apply minimum stint length logic
        if (
            self.state["tire_age"] < min_stint_laps and urgency < 3
        ):  # Don't override urgent cliff pit
            # Only consider overriding non-urgent pits if stint is too short
            if should_pit:
                should_pit = False
                reason = f"Stint too short ({self.state['tire_age']} < {min_stint_laps} laps). Holding off on pit."
                urgency = 0

        if should_pit:
            # Call the more sophisticated _recommend_compound method
            new_compound_suggestion = self._recommend_compound(laps_remaining, strategy)

        return {
            "should_pit": should_pit,
            "reason": reason,
            "urgency": urgency,
            "new_compound_suggestion": new_compound_suggestion,
            "current_wear_pct": wear_pct,
            "cliff_pct": adjusted_cliff,
            "warning_pct": warning_point,
            "tire_age": self.state["tire_age"],
            "driver_adjusted_thresholds": {
                "cliff": adjusted_cliff,
                "warning_ratio": adjusted_warning,
                "min_stint_laps": min_stint_laps,
            },
        }

    def _estimate_laps_until_cliff(self) -> int:
        """
        Estimate laps until tire performance cliff.

        Returns:
            Estimated number of laps until cliff
        """
        compound = self.state["current_compound"]
        current_wear = self.state["tire_wear"]
        cliff = self.cliff_thresholds.get(compound, 75.0)
        wear_rate = max(0.1, self.state["last_wear_rate"])  # Prevent division by zero

        laps_until_cliff = int((cliff - current_wear) / wear_rate)
        return max(0, laps_until_cliff)

    def _recommend_compound(
        self, laps_remaining: int, strategy_input: Dict[str, Any]
    ) -> str:
        """
        Recommend tire compound based on remaining laps and strategy.
        Enhanced logic considering weather, mandatory compounds, and stint length.

        Args:
            laps_remaining: Number of laps remaining in the race for this stint.
            strategy_input: Dictionary containing strategy information like:
                - weather_forecast: Output from WeatherAgent (includes current_weather & forecast).
                - total_laps: Total laps in the race.
                - available_slick_compounds: List of slicks for the event (e.g., ["SOFT", "MEDIUM", "HARD"]).
        Returns:
            Recommended compound string (e.g., "MEDIUM", "SOFT", "HARD", "INTERMEDIATE", "WET").
        """
        weather_data = strategy_input.get("weather_forecast", {})
        current_weather_details = weather_data.get(
            "current_weather", weather_data
        )  # Handles if weather_data is already current_weather

        rainfall = current_weather_details.get(
            "Rainfall_mm", current_weather_details.get("rainfall", 0.0)
        )

        # Weather check first
        if rainfall > 2.0:  # Threshold for WET tires
            return "WET"
        elif rainfall > 0.1:  # Threshold for INTERMEDIATE tires (e.g. light rain/damp)
            return "INTERMEDIATE"

        # Dry compound logic
        total_laps = strategy_input.get("total_laps", 50)
        if total_laps <= 0:
            total_laps = 50  # Avoid division by zero

        available_event_slicks = strategy_input.get(
            "available_slick_compounds", ["SOFT", "MEDIUM", "HARD"]
        )
        if not available_event_slicks:
            available_event_slicks = ["SOFT", "MEDIUM", "HARD"]  # Fallback

        current_stint_compound = self.state["current_compound"]
        compounds_already_used_set = set(self.state.get("compounds_used", []))

        slick_compounds_used_in_race = {
            c for c in compounds_already_used_set if c in available_event_slicks
        }

        must_fulfill_mandatory_slick_rule = False
        if len(available_event_slicks) >= 2:
            if len(slick_compounds_used_in_race) < 2:
                if not slick_compounds_used_in_race:
                    pass
                else:
                    must_fulfill_mandatory_slick_rule = True

        stint_length_pct = laps_remaining / total_laps if total_laps > 0 else 0
        next_compound = None

        potential_next_slicks = [
            c for c in available_event_slicks if c != current_stint_compound
        ]
        if (
            not potential_next_slicks
            and current_stint_compound in available_event_slicks
            and len(available_event_slicks) > 0
        ):
            potential_next_slicks = list(available_event_slicks)

        if must_fulfill_mandatory_slick_rule and laps_remaining > 3:
            options_for_mandatory = [
                c
                for c in available_event_slicks
                if c not in slick_compounds_used_in_race
            ]
            if not options_for_mandatory:
                must_fulfill_mandatory_slick_rule = False
            else:
                if stint_length_pct > 0.55 and "HARD" in options_for_mandatory:
                    next_compound = "HARD"
                elif stint_length_pct > 0.25 and "MEDIUM" in options_for_mandatory:
                    next_compound = "MEDIUM"
                elif "SOFT" in options_for_mandatory:
                    next_compound = "SOFT"
                elif "MEDIUM" in options_for_mandatory:
                    next_compound = "MEDIUM"
                elif "HARD" in options_for_mandatory:
                    next_compound = "HARD"
                else:
                    next_compound = options_for_mandatory[0]

        if not next_compound:
            if not potential_next_slicks and available_event_slicks:
                potential_next_slicks = list(available_event_slicks)

            if len(potential_next_slicks) == 1:
                next_compound = potential_next_slicks[0]
            elif potential_next_slicks:
                if stint_length_pct > 0.55 and "HARD" in potential_next_slicks:
                    next_compound = "HARD"
                elif stint_length_pct > 0.25 and "MEDIUM" in potential_next_slicks:
                    next_compound = "MEDIUM"
                elif "SOFT" in potential_next_slicks:
                    next_compound = "SOFT"
                elif "MEDIUM" in potential_next_slicks:
                    next_compound = "MEDIUM"
                elif "HARD" in potential_next_slicks:
                    next_compound = "HARD"
                else:
                    next_compound = potential_next_slicks[0]

        if not next_compound:
            if "MEDIUM" in available_event_slicks:
                next_compound = "MEDIUM"
            elif "HARD" in available_event_slicks:
                next_compound = "HARD"
            elif "SOFT" in available_event_slicks:
                next_compound = "SOFT"
            elif available_event_slicks:
                next_compound = available_event_slicks[0]
            else:
                next_compound = "MEDIUM"

        return next_compound

    def _calculate_wear_impact(
        self, current_wear: float, compound_characteristics: Dict
    ) -> Dict:
        """
        Calculate the impact of tire wear on grip.

        Args:
            current_wear: Current tire wear percentage (0-100)
            compound_characteristics: Tire compound characteristics

        Returns:
            Dictionary with wear impact details
        """
        # Ensure wear is positive and within bounds
        current_wear = max(0.0, min(100.0, float(current_wear)))

        cliff_point = compound_characteristics["cliff_point"]
        cliff_severity = compound_characteristics["cliff_severity"]

        # Calculate base grip loss (linear component)
        base_grip_loss = (current_wear / 100.0) * 0.5  # Maximum 50% grip loss from wear

        # Check for high wear effects (non-linear)
        high_wear_active = current_wear > (
            cliff_point * 0.7
        )  # Start non-linear effects before cliff
        cliff_active = current_wear > cliff_point

        # Apply additional grip loss for high wear
        if high_wear_active:
            wear_beyond_threshold = (current_wear - (cliff_point * 0.7)) / (
                cliff_point * 0.3
            )
            base_grip_loss += wear_beyond_threshold * 0.2  # Additional 20% max loss

        # Apply cliff effect if active
        if cliff_active:
            wear_beyond_cliff = (current_wear - cliff_point) / (100 - cliff_point)
            base_grip_loss += (
                wear_beyond_cliff * cliff_severity * 0.3
            )  # Severe grip loss

        # Calculate final grip multiplier (ensure it stays positive)
        grip_multiplier = max(0.1, 1.0 - base_grip_loss)

        return {
            "grip_multiplier": float(grip_multiplier),
            "base_grip_loss": float(base_grip_loss),
            "high_wear_active": bool(high_wear_active),
            "cliff_active": bool(cliff_active),
            "compound_characteristics": compound_characteristics,
        }

    def _calculate_grip_level(
        self,
        initial_grip: float,
        wear_impact: Dict,
        temp_impact: Dict,
        weather_impact: float,
    ) -> float:
        """
        Calculate the final grip level considering all factors.

        Args:
            initial_grip: Base grip level (0-1)
            wear_impact: Impact of tire wear
            temp_impact: Impact of temperature
            weather_impact: Impact of weather conditions

        Returns:
            Final grip level (0-1)
        """
        # Ensure all inputs are real numbers and within bounds
        initial_grip = max(0.0, min(1.0, float(initial_grip)))
        weather_impact = max(0.0, min(1.0, float(weather_impact)))

        # Get grip multipliers
        wear_multiplier = float(wear_impact["grip_multiplier"])
        temp_multiplier = float(temp_impact["grip_multiplier"])

        # Ensure multipliers are within bounds
        wear_multiplier = max(0.1, min(1.0, wear_multiplier))
        temp_multiplier = max(0.1, min(1.0, temp_multiplier))

        # Calculate combined grip
        grip = initial_grip * wear_multiplier * temp_multiplier * weather_impact

        # Ensure final grip stays within realistic bounds
        return max(0.3, min(1.0, float(grip)))  # Minimum 30% grip

    def calculate_wear_increment(
        self,
        compound: str,
        current_wear: float,
        driving_style: float,
        track_condition: float,
        is_following: bool,
        is_defending: bool,
        is_attacking: bool,
    ) -> float:
        """
        Calculate tire wear increment for current lap.

        Args:
            compound: Tire compound
            current_wear: Current tire wear percentage
            driving_style: Driver's aggression level (0-1)
            track_condition: Track condition factor
            is_following: Whether following another car closely
            is_defending: Whether defending position
            is_attacking: Whether attacking position ahead

        Returns:
            Wear increment percentage
        """
        # Ensure inputs are within bounds
        current_wear = max(0.0, min(100.0, float(current_wear)))
        driving_style = max(0.0, min(1.0, float(driving_style)))
        track_condition = max(0.5, min(1.5, float(track_condition)))

        # Base wear rates per compound (realistic F1 values)
        base_rates = {
            "SOFT": 0.8,  # Higher wear rate
            "MEDIUM": 0.6,  # Balanced wear rate
            "HARD": 0.4,  # Lower wear rate
            "INTERMEDIATE": 0.7,
            "WET": 0.5,
        }

        # Get base wear rate for compound
        base_wear = base_rates.get(compound, 0.6)  # Default to medium if unknown

        # Apply driving style effect (exponential to punish aggressive driving)
        style_multiplier = 1.0 + (driving_style**2)

        # Apply track condition effect
        condition_effect = 1.0 / track_condition  # Worse conditions = more wear

        # Additional wear from racing situations
        racing_multiplier = 1.0
        if is_following:
            racing_multiplier *= 1.2  # 20% more wear when following
        if is_defending:
            racing_multiplier *= 1.15  # 15% more when defending
        if is_attacking:
            racing_multiplier *= 1.25  # 25% more when attacking

        # Calculate wear increment
        wear_increment = (
            base_wear * style_multiplier * condition_effect * racing_multiplier
        )

        # Apply progressive wear increase as tire ages
        if current_wear > 50:
            wear_increment *= 1.0 + ((current_wear - 50) / 100)

        # Ensure wear increment stays within realistic bounds
        return max(0.1, min(2.0, float(wear_increment)))
