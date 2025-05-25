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

        # Initialize tire wear agent for degradation predictions
        self.tire_wear_agent = TireWearAgent(
            name="TireWearAgent", model_path=tire_wear_model_path
        )
        self.grip_model_agent = GripModelAgent(name="GripModelAgent")

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
            "current_compound": "MEDIUM",
            "tire_age": 0,
            "tire_wear": 0.0,
            "grip_level": 1.0,
            "in_optimal_window": True,
            "tire_health": "new",  # new, good, marginal, critical, expired
            "last_wear_rate": 0.0,  # wear per lap (percentage)
            "prev_lap_degradation_s_for_wear_model": 0.0,  # ADDED: To feed TireWearAgent
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
            # Convert all values to JSON serializable types
            def make_serializable(obj):
                if isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                else:
                    return str(obj)  # Convert any other types to strings

            debug_info = {
                "lap_info": {
                    "lap_number": inputs.get("current_lap", 0),
                    "stint_lap": self.state["tire_age"],
                    "compound": self.state["current_compound"],
                },
                "wear_factors": {
                    "raw_degradation_s": float(degradation_s),  # Ensure float
                    "wear_increment_pct": float(wear_increment),  # Ensure float
                    "total_wear_pct": float(self.state["tire_wear"]),  # Ensure float
                },
                "grip_factors": make_serializable(grip_result),
                "compound_specific": {
                    "cliff_point": float(
                        self.cliff_points.get(
                            self.state["current_compound"], self.default_cliff_point
                        )
                    ),
                    "max_degradation_s": float(
                        self.total_degradation_s_for_100_percent_wear.get(
                            self.state["current_compound"],
                            self.default_total_degradation_s_for_100_wear,
                        )
                    ),
                },
                "model_factors": make_serializable(wear_result.get("wear_factors", {})),
                "weather_impact": make_serializable(inputs.get("weather", {})),
                "speed_metrics": {
                    "SpeedST": float(inputs.get("SpeedST", 0)),
                    "SpeedI1": float(inputs.get("SpeedI1", 0)),
                    "SpeedI2": float(inputs.get("SpeedI2", 0)),
                },
            }

            try:
                logger.debug(f"Tire Analysis:\n{json.dumps(debug_info, indent=2)}")
            except TypeError as e:
                logger.error(f"Failed to serialize debug info: {e}")
                # Log raw string representation as fallback
                logger.debug(f"Tire Analysis (raw):\n{str(debug_info)}")

    def _calculate_wear_from_degradation(
        self,
        degradation_s: float,
        compound: str,
        lap_in_stint: int,
        driver_chars: Dict[str, float] = None,
    ) -> float:
        """
        Convert model's degradation prediction to wear percentage more accurately.
        """
        # Get base conversion factor for compound
        base_conversion = self.total_degradation_s_for_100_percent_wear.get(
            compound, self.default_total_degradation_s_for_100_wear
        )

        # Get driver characteristics or use defaults
        if driver_chars is None:
            driver_chars = {}
        tire_management = driver_chars.get("tire_management", 1.0)
        aggression = driver_chars.get("aggression", 1.0)
        consistency = driver_chars.get("consistency", 1.0)

        # Modified stint factor calculation with driver characteristics
        # Initial phase (first 3 laps): reduced wear for tire warm-up
        if lap_in_stint <= 3:
            # More aggressive drivers heat tires up faster
            warmup_factor = 0.7 + (lap_in_stint * 0.1 * aggression)
            stint_factor = min(1.0, warmup_factor)
        else:
            # After warm-up: more gradual increase affected by driving style
            base_stint_factor = (
                1.0 + ((lap_in_stint - 3) / 40) ** 0.5
            )  # More gradual increase
            # Aggressive drivers increase wear faster but with less extreme effect
            stint_factor = min(1.3, base_stint_factor * (1 + (aggression - 1) * 0.2))

        # Calculate base wear with tire management skill
        base_wear = (degradation_s / base_conversion) * 100
        # Better tire management reduces base wear
        base_wear *= 1.5 - (tire_management * 0.5)  # Reduced impact of tire management

        # Compound-specific minimum and maximum wear per lap
        min_wear_per_lap = {
            "SOFT": 0.3 * (1.5 - tire_management * 0.5),  # Reduced from 0.4
            "MEDIUM": 0.2 * (1.5 - tire_management * 0.5),  # Reduced from 0.3
            "HARD": 0.15 * (1.5 - tire_management * 0.5),  # Reduced from 0.2
            "INTERMEDIATE": 0.4 * (1.5 - tire_management * 0.5),
            "WET": 0.5 * (1.5 - tire_management * 0.5),
        }.get(compound, 0.25 * (1.5 - tire_management * 0.5))

        max_wear_per_lap = {
            "SOFT": 2.5 * aggression,  # Reduced from 4.0
            "MEDIUM": 2.0 * aggression,  # Reduced from 3.0
            "HARD": 1.5 * aggression,  # Reduced from 2.5
            "INTERMEDIATE": 3.0 * aggression,
            "WET": 3.5 * aggression,
        }.get(compound, 2.0 * aggression)

        # Calculate wear with stint factor
        wear_increment = base_wear * stint_factor

        # Add randomness based on driver consistency
        # Less consistent drivers have more variation
        variation_scale = 0.02 * (2 - consistency)  # Reduced from 0.03
        wear_variation = np.random.normal(0, variation_scale)
        wear_increment *= 1 + wear_variation

        # Ensure wear stays within compound-specific bounds
        wear_increment = min(max_wear_per_lap, max(min_wear_per_lap, wear_increment))

        return wear_increment

    def _calculate_grip_from_wear(self, wear_pct: float, compound: str) -> float:
        """
        Calculate grip level based on wear percentage with compound-specific cliff points.
        """
        cliff_point = self.cliff_points.get(compound, self.default_cliff_point)

        # Store previous grip level for validation
        prev_grip = self.state.get("grip_level", 1.0)

        # More gradual initial grip loss
        if wear_pct < cliff_point * 0.5:  # First half of tire life
            grip_loss = (
                wear_pct / cliff_point
            ) ** 1.3  # Reduced from 1.5 for more gradual loss
        elif wear_pct < cliff_point:  # Second half until cliff
            mid_point_loss = (0.5) ** 1.3
            remaining_pct = (wear_pct - (cliff_point * 0.5)) / (cliff_point * 0.5)
            grip_loss = mid_point_loss + (remaining_pct * (1 - mid_point_loss))
        else:  # After cliff point
            base_loss = 1.0  # Full base loss at cliff
            extra_wear = wear_pct - cliff_point
            # More gradual post-cliff drop-off (2.5 power instead of 3.0)
            cliff_loss = (extra_wear / (100 - cliff_point)) ** 2.5
            grip_loss = base_loss * (1 + cliff_loss)

        # Calculate final grip level
        grip_level = 1.0 - (grip_loss * self.max_grip_loss_at_100_percent_wear)

        # Smaller random variations (±0.5% instead of ±1%)
        grip_variation = np.random.normal(0, 0.005)
        grip_level *= 1 + grip_variation

        # Ensure grip never increases and stays within bounds
        grip_level = min(prev_grip, max(0.3, min(1.0, grip_level)))

        return grip_level

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
        laps_remaining = inputs.get("laps_remaining", 0)

        # Get driver characteristics or use defaults
        driver_chars = inputs.get("characteristics", {})
        tire_management_factor = driver_chars.get("tire_management", 1.0)
        consistency_factor = driver_chars.get("consistency", 1.0)
        aggression_factor = driver_chars.get("aggression", 1.0)
        wet_weather_skill = driver_chars.get("wet_weather", 1.0)

        # Check if this is a pit lap dictated by StrategyAgent and new compound is chosen
        is_pit_lap_from_strategy = inputs.get("is_pit_lap", False)
        new_compound_from_strategy = inputs.get("new_compound", None)

        if is_pit_lap_from_strategy and new_compound_from_strategy:
            # Record previous tire in history if it was a valid stint
            if self.state["current_compound"] and self.state["tire_age"] > 0:
                self.state["tire_history"].append(
                    (self.state["current_compound"], self.state["tire_age"])
                )

            # Set new compound based on StrategyAgent's decision
            self.state["current_compound"] = new_compound_from_strategy.upper()
            self.state["tire_age"] = 0
            self.state["tire_wear"] = 0.0
            self.state["grip_level"] = 1.0
            self.state["in_optimal_window"] = True
            self.state["tire_health"] = "new"
            self.state["last_wear_rate"] = 0.0
            self.state["prev_lap_degradation_s_for_wear_model"] = 0.0

            if self.state["current_compound"] not in self.state["compounds_used"]:
                self.state["compounds_used"].append(self.state["current_compound"])

            logger.info(
                f"New {self.state['current_compound']} tires fitted (Strategy Decision)"
            )

        elif current_lap == 1 and not self.state["compounds_used"]:
            # First lap of the race, initialize based on starting compound
            raw_input_compound = inputs.get("current_compound")
            initial_compound = inputs.get("current_compound", "MEDIUM")
            self.state["current_compound"] = initial_compound.upper()
            self.state["tire_age"] = 0
            self.state["tire_wear"] = 0.0
            self.state["grip_level"] = 1.0
            self.state["in_optimal_window"] = True
            self.state["tire_health"] = "new"
            self.state["last_wear_rate"] = 0.0
            self.state["prev_lap_degradation_s_for_wear_model"] = 0.0

            if self.state["current_compound"] not in self.state["compounds_used"]:
                self.state["compounds_used"].append(self.state["current_compound"])
            logger.info(
                f"Initial {self.state['current_compound']} tires fitted for Lap 1"
            )
        else:
            # Normal lap, not pitting, not lap 1 init
            self.state["tire_age"] += 1

        # Store speed metrics for wear calculations
        self._last_speed_metrics = {
            "SpeedST": inputs.get("SpeedST", 0),
            "SpeedI1": inputs.get("SpeedI1", 0),
            "SpeedI2": inputs.get("SpeedI2", 0),
        }

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
        self.state["in_optimal_window"] = wear_result["optimal_window"]

        # Calculate wear using new method with driver characteristics
        wear_increment = self._calculate_wear_from_degradation(
            degradation_this_lap_s,
            self.state["current_compound"],
            self.state["tire_age"],
            driver_chars,
        )

        # Update accumulated wear
        self.state["tire_wear"] = min(100, self.state["tire_wear"] + wear_increment)

        # Use new GripModelAgent for grip calculations
        grip_inputs = {
            "compound": self.state["current_compound"],
            "tire_wear": self.state["tire_wear"],
            "tire_temp": inputs.get("tire_temp", 90.0),  # Default to reasonable temp
            "track_temp": inputs.get("track_temp", 30.0),
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

        # Log detailed degradation analysis
        self._log_degradation_factors(
            inputs, wear_result, degradation_this_lap_s, wear_increment, grip_result
        )

        # STORE the predicted degradation for THIS lap to be used as PREVIOUS for the NEXT lap
        self.state["prev_lap_degradation_s_for_wear_model"] = degradation_this_lap_s

        # Update last_wear_rate
        self.state["last_wear_rate"] = wear_increment

        # Update tire health status
        self._update_tire_health()

        # Generate pit recommendation
        pit_recommendation = self._generate_pit_recommendation(
            laps_remaining, inputs.get("strategy", {})
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
            "temp_performance": grip_result["temp_performance"],
            "wear_grip": grip_result["wear_grip"],
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
