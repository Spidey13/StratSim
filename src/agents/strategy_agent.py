"""
Strategy planning agent that makes pit stop and tire compound decisions.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        # Default pit window size (laps) - Now a base value that will be adjusted per driver
        self.base_pit_window_size = 5

        # Compound performance characteristics (relative pace)
        self.compound_pace = {
            "SOFT": 1.0,  # Baseline (fastest)
            "MEDIUM": 1.01,  # 1% slower than soft
            "HARD": 1.02,  # 2% slower than soft
            "INTERMEDIATE": 1.05,  # 5% slower in dry, fastest in light rain
            "WET": 1.10,  # 10% slower in dry, fastest in heavy rain
        }

        # Base tire life characteristics (estimated laps at good performance)
        # These will be adjusted based on driver characteristics
        self.base_tire_life = {
            "SOFT": 20,
            "MEDIUM": 30,
            "HARD": 40,
            "INTERMEDIATE": 25,
            "WET": 30,
        }

        # Strategy state
        self.planned_stops = {}  # Planned pit stops by driver
        self.last_decisions = {}  # Last decision made for each driver

        # Base pit stop decision thresholds
        # These will be adjusted based on driver characteristics
        self.base_pit_decision_thresholds = {
            "wear_threshold": 75.0,  # Base wear threshold
            "grip_threshold": 0.85,  # Base grip threshold
            "performance_delta_threshold": 1.5,  # Base performance delta threshold
            "minimum_stint_length": 12,  # Base minimum stint length
            "safety_margin_laps": 3,  # Base safety margin
        }

        # Compound selection weights
        self.compound_selection_weights = {
            "remaining_laps": 2.0,  # Higher weight on remaining race distance
            "track_temp": 1.5,  # Temperature impact on compound choice
            "position": 1.0,  # Track position consideration
            "wear_rate": 1.8,  # Higher weight on wear rate analysis
            "competitor_strategy": 1.2,  # Consider competitor strategies
        }

        # Expected stint lengths (laps) for each compound
        self.expected_stint_lengths = {
            "SOFT": 18,  # Increased from 12 - more realistic soft tire life
            "MEDIUM": 28,  # Increased from 18 - more realistic medium tire life
            "HARD": 38,  # Increased from 25 - more realistic hard tire life
            "INTERMEDIATE": 20,
            "WET": 15,
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process race information to make strategy decisions.

        Args:
            inputs: Dictionary containing:
                - lap: Current lap number (int)
                - total_laps: Total race laps (int)
                - driver_id: Driver identifier (str)
                - driver_state: Current driver state (dict), expected to contain:
                    - 'current_compound' (str)
                    - 'tire_age' (int)
                    - 'pit_stops' (list) - list of dicts for pit stop history
                - weather: Current weather conditions (dict)
                - track_state: Current track conditions (dict)

        Returns:
            Dictionary containing:
                - pit_decision: Boolean indicating whether to pit
                - new_compound: Compound to change to if pitting
                - expected_benefit: Expected time benefit of decision (float)
                - reasoning: Explanation of the decision (str)
        """
        # Extract inputs
        lap = inputs.get("lap", 0)
        total_laps = inputs.get("total_laps", 0)
        driver_id = inputs.get("driver_id", "unknown")
        driver_state = inputs.get("driver_state", {})
        weather = inputs.get("weather", {})
        track_state = inputs.get("track_state", {})
        pit_time_penalty = inputs.get("pit_time_penalty", 20.0)
        driver_chars = inputs.get("driver_characteristics", {})

        # Get accurate position data
        track_positions = track_state.get("positions", {})
        current_position = track_positions.get(
            driver_id, driver_state.get("position", 1)
        )

        # Calculate gaps using track_positions
        sorted_positions = sorted(
            [(pos, d_id) for d_id, pos in track_positions.items()]
        )
        current_idx = next(
            (i for i, (pos, d_id) in enumerate(sorted_positions) if d_id == driver_id),
            0,
        )

        gap_ahead = driver_state.get("gap_to_ahead", 0.0)
        gap_to_leader = driver_state.get("gap_to_leader", 0.0)

        # Debug position data
        logger.debug(
            f"\nPosition data for {driver_id} at lap {lap}:"
            f"\n  Track positions: {track_positions}"
            f"\n  Current position: P{current_position}"
            f"\n  Gap ahead: {gap_ahead:.1f}s"
            f"\n  Gap to leader: {gap_to_leader:.1f}s"
            f"\n  Total race time: {driver_state.get('total_race_time', 0.0):.3f}s"
        )

        # Get driver characteristics
        tire_management = driver_chars.get("tire_management", 1.0)
        aggression = driver_chars.get("aggression", 1.0)
        consistency = driver_chars.get("consistency", 1.0)

        # Log initial state
        logger.debug(
            f"\nStrategy evaluation for {driver_id} at lap {lap}/{total_laps}:"
            f"\n  Position: P{driver_state.get('position', '?')}"
            f"\n  Grid Position: P{driver_state.get('grid_position', '?')}"
            f"\n  Positions Gained: {driver_state.get('positions_gained', 0)}"
            f"\n  Gap to Leader: {driver_state.get('gap_to_leader', 0):.1f}s"
            f"\n  Gap to Ahead: {driver_state.get('gap_to_ahead', 0):.1f}s"
            f"\n  Current Compound: {driver_state.get('current_compound', 'UNKNOWN')}"
            f"\n  Tire Age: {driver_state.get('tire_age', 0)} laps"
            f"\n  Tire Wear: {driver_state.get('tire_wear', 0):.1f}%"
            f"\n  Grip Level: {driver_state.get('grip_level', 1.0):.3f}"
        )

        # Adjust thresholds based on driver characteristics
        adjusted_thresholds = self._get_adjusted_thresholds(
            tire_management, aggression, consistency
        )

        # Get current tire information from driver_state
        current_compound = driver_state.get("current_compound", "MEDIUM")
        tire_age = driver_state.get("tire_age", 0)
        pit_stops = driver_state.get("pit_stops", [])
        grip_level = driver_state.get("grip_level", 1.0)

        # Calculate position based on current performance
        new_position = current_position

        # Calculate current tire performance using existing metrics
        current_pace_factor = self.compound_pace.get(current_compound, 1.05)
        max_tire_life = self._get_adjusted_tire_life(
            current_compound, tire_management, aggression
        )
        tire_life_pct = tire_age / max_tire_life if max_tire_life > 0 else 1.0

        # Lose positions if:
        # 1. During pit stops
        # 2. Poor tire performance (high age, low grip)
        # 3. Suboptimal compound for conditions
        if pit_stops and lap in pit_stops:
            # Use pit_time_penalty to determine position loss
            position_loss = min(
                3, int(pit_time_penalty / 2)
            )  # ~2-3 positions based on pit time
            new_position = min(20, current_position + position_loss)
        elif tire_life_pct > 0.8 or grip_level < adjusted_thresholds["grip_threshold"]:
            # Significant tire wear or low grip
            new_position = min(20, current_position + 1)
        elif tire_age < 3 and grip_level > 0.9:
            # Fresh tires with good grip
            new_position = max(1, current_position - 1)

        # Update driver state with new position
        driver_state["position"] = new_position

        # Get pit recommendation from TireManagerAgent
        pit_recommendation = driver_state.get("pit_recommendation", {})
        tm_recommends_pit = pit_recommendation.get("should_pit", False)
        tm_recommended_compound = pit_recommendation.get("new_compound_suggestion")
        tm_pit_reason = pit_recommendation.get("reason", "No specific reason from TMA.")

        # Check if weather conditions require immediate pit
        weather_pit, weather_compound = self._check_weather_pit(
            current_compound, weather, track_state
        )

        # Check if we've exceeded the adjusted tire life limit
        tire_life_exceeded = (
            tire_age >= max_tire_life * adjusted_thresholds["wear_threshold_pct"]
        )

        # Calculate optimal strategy
        if weather_pit:
            # Weather-based pit decision overrides
            pit_decision = True
            new_compound = weather_compound
            benefit = self._calculate_compound_benefit(
                current_compound,
                new_compound,
                tire_age,
                weather,
                pit_time_penalty,
                driver_state,  # Pass driver_state
            )
            reasoning = f"Weather conditions require switch to {new_compound}"
        elif tire_life_exceeded:
            # Tire life limit exceeded
            pit_decision = True
            new_compound = self._select_optimal_compound(total_laps - lap, weather)
            benefit = self._calculate_compound_benefit(
                current_compound,
                new_compound,
                tire_age,
                weather,
                pit_time_penalty,
                driver_state,  # Pass driver_state
            )
            reasoning = f"Tire age ({tire_age}) exceeded adjusted compound life limit ({max_tire_life})"
        elif tm_recommends_pit:
            # TireManagerAgent recommends pitting
            pit_decision = True
            new_compound = tm_recommended_compound or self._select_optimal_compound(
                total_laps - lap, weather
            )
            benefit = self._calculate_compound_benefit(
                current_compound,
                new_compound,
                tire_age,
                weather,
                pit_time_penalty,
                driver_state,  # Pass driver_state
            )
            reasoning = tm_pit_reason
        else:
            # Check if we're in a strategic window
            in_pit_window = self._should_make_strategic_stop(
                lap, total_laps, driver_state, weather
            )
            if in_pit_window:
                pit_decision = True
                new_compound = self._select_optimal_compound(total_laps - lap, weather)
                benefit = self._calculate_compound_benefit(
                    current_compound,
                    new_compound,
                    tire_age,
                    weather,
                    pit_time_penalty,
                    driver_state,  # Pass driver_state
                )
                reasoning = "Strategic pit window"
            else:
                pit_decision = False
                new_compound = current_compound
                benefit = 0.0
                reasoning = "No immediate pit required"

        # Store decision for this driver
        self.last_decisions[driver_id] = {
            "lap": lap,
            "pit_decision": pit_decision,
            "current_compound": current_compound,
            "new_compound": new_compound,
            "tire_age": tire_age,
        }

        # Record pit stop if decided
        if not driver_id in self.planned_stops:
            self.planned_stops[driver_id] = []

        if pit_decision:
            self.planned_stops[driver_id].append(
                {
                    "lap": lap,
                    "old_compound": current_compound,
                    "new_compound": new_compound,
                    "reason": reasoning,
                }
            )

        # Add after weather pit check
        if not weather_pit:
            # Check end-of-race considerations
            avoid_pit, avoid_reason = self._evaluate_end_race_strategy(
                lap, total_laps, current_compound, tire_age
            )
            if avoid_pit:
                return {
                    "pit_decision": False,
                    "new_compound": current_compound,
                    "expected_benefit": 0.0,
                    "reasoning": avoid_reason,
                }

        return {
            "pit_decision": pit_decision,
            "new_compound": new_compound,
            "expected_benefit": benefit,
            "reasoning": reasoning,
        }

    def _get_adjusted_thresholds(
        self, tire_management: float, aggression: float, consistency: float
    ) -> Dict[str, float]:
        """
        Get adjusted thresholds based on driver characteristics.
        """
        # Base thresholds
        thresholds = self.base_pit_decision_thresholds.copy()

        # Adjust wear threshold
        wear_adjustment = (tire_management - 1.0) * 10  # ±10% based on tire management
        wear_adjustment -= (aggression - 1.0) * 5  # ±5% based on aggression
        wear_adjustment += (consistency - 1.0) * 5  # ±5% based on consistency
        thresholds["wear_threshold"] = min(
            90, max(60, thresholds["wear_threshold"] + wear_adjustment)
        )

        # Adjust grip threshold
        grip_adjustment = (tire_management - 1.0) * 0.05  # ±5% based on tire management
        grip_adjustment -= (aggression - 1.0) * 0.03  # ±3% based on aggression
        thresholds["grip_threshold"] = min(
            0.9, max(0.7, thresholds["grip_threshold"] + grip_adjustment)
        )

        # Adjust minimum stint length
        stint_adjustment = int(
            (tire_management - 1.0) * 3
        )  # ±3 laps based on tire management
        stint_adjustment -= int((aggression - 1.0) * 2)  # ±2 laps based on aggression
        thresholds["minimum_stint_length"] = max(
            8, thresholds["minimum_stint_length"] + stint_adjustment
        )

        # Add percentage threshold for tire life
        thresholds["wear_threshold_pct"] = (
            0.9 + (tire_management - 1.0) * 0.1
        )  # 80-100% of max life

        return thresholds

    def _get_adjusted_tire_life(
        self, compound: str, tire_management: float, aggression: float
    ) -> int:
        """
        Get adjusted tire life based on compound and driver characteristics.
        """
        base_life = self.base_tire_life.get(compound, 25)

        # Adjust based on driver characteristics
        life_adjustment = (
            tire_management - 1.0
        ) * 5  # ±5 laps based on tire management
        life_adjustment -= (aggression - 1.0) * 3  # ±3 laps based on aggression

        return max(15, int(base_life + life_adjustment))

    def _calculate_wear_threshold(self, compound: str, total_laps: int) -> float:
        """
        Calculate the tire age threshold for pitting based on compound.

        Args:
            compound: Tire compound
            total_laps: Total race laps

        Returns:
            Threshold in laps
        """
        base_life = self.base_tire_life.get(compound, 25)

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
            weather: Weather conditions (expected keys: 'rainfall', 'condition')
            track_state: Track conditions (expected keys: 'dry_line')

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
        pit_recommendation: Dict[str, Any],
        pit_time_penalty: float,
    ) -> Tuple[bool, str, float, str]:
        """
        Calculate optimal strategy decision.
        Now primarily driven by TireManager's recommendation.
        Original logic can be a fallback or for non-wear related decisions.

        Args:
            lap: Current lap
            total_laps: Total race laps
            driver_state: Driver state
            weather: Weather conditions
            track_state: Track conditions
            pit_recommendation: Recommendation from TireManagerAgent
            pit_time_penalty: Time lost for a pit stop

        Returns:
            Tuple of (pit_decision, new_compound, expected_benefit, reasoning)
        """
        current_compound = driver_state.get("current_compound", "MEDIUM")
        tire_age = driver_state.get("tire_age", 0)

        # Don't pit on first or last few laps (unless emergency, which TM should flag)
        if lap < 3 or lap > total_laps - 3:
            # Allow TM recommendation to override this if urgent
            if not pit_recommendation.get("should_pit", False):
                return (
                    False,
                    current_compound,
                    0.0,
                    "Too early or too late for a strategic pit.",
                )
            # If TM says pit, proceed even if early/late, TM reason will be used.

        tm_recommends_pit = pit_recommendation.get("should_pit", False)
        tm_recommended_compound = pit_recommendation.get("new_compound_suggestion")
        tm_pit_reason = pit_recommendation.get("reason", "Tire Manager recommendation")

        if tm_recommends_pit:
            new_compound = (
                tm_recommended_compound
                if tm_recommended_compound
                else self._select_optimal_compound(total_laps - lap, weather)
            )
            benefit = self._calculate_compound_benefit(
                current_compound,
                new_compound,
                tire_age,
                weather,
                pit_time_penalty,
                driver_state,
            )
            return True, new_compound, benefit, tm_pit_reason

        # Fallback: Original tire age logic (if we want to keep it)
        # wear_threshold_laps = self._calculate_wear_threshold(current_compound, total_laps) # Calculate it here
        # if tire_age >= wear_threshold_laps:
        #     new_compound = self._select_optimal_compound(total_laps - lap, weather)
        #     benefit = self._calculate_compound_benefit(
        #         current_compound, new_compound, tire_age, weather, pit_time_penalty
        #     )
        #     return True, new_compound, benefit, f"Tire age ({tire_age}) exceeded wear threshold ({wear_threshold_laps:.1f} laps)"

        # If no pit recommended by TM and not by old wear threshold (if active),
        # consider other strategic stops (e.g., undercut/overcut).
        # This part is currently not implemented in detail.
        # For now, if TM doesn't say pit, and weather is fine, we don't pit.

        return (
            False,
            current_compound,
            0.0,
            "No pit recommended by TireManager or other strategic triggers.",
        )

    def _select_optimal_compound(
        self, remaining_laps: int, weather: Dict[str, Any]
    ) -> str:
        """
        Select the optimal compound based on remaining laps and conditions.
        """
        rainfall = weather.get("rainfall", 0)

        # If it's raining, choose appropriate rain tire
        if rainfall > 0:
            if rainfall >= 2:
                return "WET"
            else:
                return "INTERMEDIATE"

        # For dry conditions, choose based on remaining laps and expected tire life
        # We want to ensure the tire can last the remaining stint
        if remaining_laps <= self.base_tire_life["SOFT"] * 0.9:  # Can soft tires last?
            return "SOFT"  # Aggressive strategy for short stint
        elif remaining_laps <= self.base_tire_life["MEDIUM"] * 0.9:  # Can mediums last?
            return "MEDIUM"  # Balanced choice for medium stint
        else:
            return "HARD"  # Conservative choice for long stint

    def _calculate_compound_benefit(
        self,
        current_compound: str,
        new_compound: str,
        tire_age: int,
        weather: Dict[str, Any],
        pit_time_penalty: float,
        driver_state: Dict[str, Any],
    ) -> float:
        """
        Calculate the expected time benefit of changing to a new compound.
        A positive benefit means pitting is advantageous.

        Args:
            current_compound: Current tire compound
            new_compound: New tire compound
            tire_age: Age of the current tires
            weather: Current weather conditions
            pit_time_penalty: Time lost for a pit stop (e.g., 20 seconds)
            driver_state: Current state of the driver including position info

        Returns:
            Estimated time benefit in seconds.
        """
        # Get position and gap information
        current_position = driver_state.get("position", 1)
        gap_ahead = driver_state.get("gap_to_ahead", 0.0)
        gap_to_leader = driver_state.get("gap_to_leader", 0.0)
        positions_gained = driver_state.get("positions_gained", 0)

        # Calculate race progress and position multiplier
        remaining_laps = weather.get("total_laps", 70) - weather.get("lap", 0)
        race_progress = 1 - (remaining_laps / weather.get("total_laps", 70))
        position_value_multiplier = 1 + (race_progress**2) * 2

        # Log position-based strategy considerations
        logger.debug(
            f"\nStrategy position analysis:"
            f"\n  Current Position: P{current_position}"
            f"\n  Gap Ahead: {gap_ahead:.1f}s"
            f"\n  Gap to Leader: {gap_to_leader:.1f}s"
            f"\n  Positions Gained: {positions_gained}"
            f"\n  Race Progress: {race_progress:.2%}"
        )

        # Additional position-based adjustments with logging
        original_multiplier = position_value_multiplier
        if current_position == 1:
            position_value_multiplier *= 1.5
            logger.debug(
                f"  Leading - Increased multiplier: {original_multiplier:.2f} -> {position_value_multiplier:.2f}"
            )
        elif gap_ahead < 3.0 and current_position > 1:
            position_value_multiplier *= 0.8
            logger.debug(
                f"  Close to car ahead - Decreased multiplier: {original_multiplier:.2f} -> {position_value_multiplier:.2f}"
            )
        elif gap_ahead > 5.0 and positions_gained > 0:
            position_value_multiplier *= 1.3
            logger.debug(
                f"  Clear air with positions gained - Increased multiplier: {original_multiplier:.2f} -> {position_value_multiplier:.2f}"
            )

        # Track-specific baseline lap time (should come from track database)
        baseline_lap_time = 90.0

        # Compound-specific pace factors (relative to optimal)
        # Reduced deltas between compounds for more realistic differences
        compound_pace = {
            "SOFT": 1.0,  # Baseline (fastest)
            "MEDIUM": 1.005,  # 0.5% slower than soft (was 1%)
            "HARD": 1.01,  # 1% slower than soft (was 2%)
            "INTERMEDIATE": 1.03,  # Wet conditions
            "WET": 1.06,  # Full wet conditions
        }

        # Tire degradation characteristics - reduced rates for more realistic behavior
        compound_deg_rates = {
            "SOFT": 0.002,  # 0.2% per lap (was 0.4%)
            "MEDIUM": 0.0015,  # 0.15% per lap (was 0.3%)
            "HARD": 0.001,  # 0.1% per lap (was 0.2%)
            "INTERMEDIATE": 0.003,
            "WET": 0.004,
        }

        # Temperature impact on compounds - more nuanced temperature windows
        track_temp = weather.get("track_temp", 30)
        temp_impact = {
            "SOFT": max(
                0, min(1.0, 1.0 - abs(track_temp - 35) * 0.008)
            ),  # Optimal ~35°C
            "MEDIUM": max(
                0, min(1.0, 1.0 - abs(track_temp - 30) * 0.006)
            ),  # Optimal ~30°C
            "HARD": max(
                0, min(1.0, 1.0 - abs(track_temp - 25) * 0.004)
            ),  # Optimal ~25°C
            "INTERMEDIATE": 1.0,  # Less temperature sensitive
            "WET": 1.0,  # Less temperature sensitive
        }

        # Add pit stop time variation based on track position
        base_pit_loss = pit_time_penalty
        if current_position > 10:
            # More traffic in the back, higher chance of losing time
            pit_time_variation = np.random.normal(1.5, 0.5)  # Additional 1-2s loss
        elif current_position > 5:
            pit_time_variation = np.random.normal(0.8, 0.3)  # 0.5-1.1s variation
        else:
            pit_time_variation = np.random.normal(0.5, 0.2)  # 0.3-0.7s variation

        adjusted_pit_penalty = base_pit_loss + pit_time_variation

        # Enhanced position-based adjustments
        position_value_multiplier = 1.0  # Base multiplier

        # Front-running position effects
        if current_position == 1:
            if gap_ahead > 5.0:
                position_value_multiplier = 0.9  # Conservative when leading with gap
                logger.debug(
                    "  Leading with comfortable gap - More conservative strategy"
                )
            else:
                position_value_multiplier = 1.2  # Defensive when leading closely
                logger.debug("  Leading but under pressure - Defensive strategy")
        elif current_position <= 3:
            if gap_ahead < 2.0:
                position_value_multiplier = 1.3  # Aggressive when fighting for podium
                logger.debug("  Podium fight - Aggressive strategy")

        # Midfield position effects
        elif current_position <= 10:
            if gap_ahead < 1.5:
                position_value_multiplier = 1.2  # Points-paying position battle
                logger.debug("  Points position battle - Aggressive strategy")
            elif gap_ahead > 4.0:
                position_value_multiplier = 0.95  # Holding position
                logger.debug("  Stable points position - Conservative strategy")

        # Calculate current tire performance with more realistic degradation
        current_deg_rate = compound_deg_rates.get(current_compound, 0.0015)
        current_base_pace = compound_pace.get(current_compound, 1.05)
        current_temp_factor = temp_impact.get(current_compound, 1.0)

        # Non-linear degradation effect (more gradual early, steeper late)
        wear_effect = 1.0 + (current_deg_rate * tire_age) ** 1.3

        # Current lap time calculation with temperature effects
        time_per_lap_current = (
            baseline_lap_time * current_base_pace * wear_effect / current_temp_factor
        )

        # Calculate new tire performance
        new_deg_rate = compound_deg_rates.get(new_compound, 0.0015)
        new_base_pace = compound_pace.get(new_compound, 1.05)
        new_temp_factor = temp_impact.get(new_compound, 1.0)

        # New tires start fresh but need warm-up
        warmup_penalty = 0.3  # 0.3s slower on first lap
        time_per_lap_new = (
            baseline_lap_time * new_base_pace / new_temp_factor + warmup_penalty
        )

        # Calculate per-lap benefit with more realistic gains
        per_lap_gain = time_per_lap_current - time_per_lap_new

        # Calculate stint length based on compound characteristics
        new_compound_ideal_life = self.base_tire_life.get(new_compound, 25)
        laps_on_new_tire = min(remaining_laps, new_compound_ideal_life)

        # Calculate total benefit over the stint with warm-up phase
        total_gain = 0
        for lap in range(laps_on_new_tire):
            if lap == 0:
                # First lap includes warm-up penalty
                lap_gain = per_lap_gain - warmup_penalty
            elif lap < 3:
                # Tires coming up to temperature
                lap_gain = per_lap_gain * (0.8 + 0.1 * lap)
            else:
                # Normal performance
                new_tire_wear_effect = 1.0 + (new_deg_rate * lap) ** 1.3
                lap_time_new = (
                    baseline_lap_time
                    * new_base_pace
                    * new_tire_wear_effect
                    / new_temp_factor
                )
                lap_time_old = (
                    time_per_lap_current * (1 + current_deg_rate * lap) ** 0.5
                )
                lap_gain = lap_time_old - lap_time_new

            total_gain += lap_gain

        # Adjust total gain based on position value multiplier
        total_gain *= position_value_multiplier

        # Final benefit calculation with variable pit stop penalty
        net_benefit = total_gain - adjusted_pit_penalty

        logger.debug(
            f"Benefit calc: Current: {current_compound} (Age: {tire_age}), New: {new_compound}. "
            f"Track Temp: {track_temp}°C, TempFactors: C={current_temp_factor:.3f}, N={new_temp_factor:.3f}. "
            f"WearEffect: {wear_effect:.3f}, PerLapGain: {per_lap_gain:.3f}s. "
            f"Position: P{current_position}, GapAhead: {gap_ahead:.1f}s, Gained: {positions_gained}, "
            f"RaceProgress: {race_progress:.2f}, PositionMultiplier: {position_value_multiplier:.2f}, "
            f"TotalGain: {total_gain:.3f}s, AdjustedPitPenalty: {adjusted_pit_penalty:.1f}s, NetBenefit: {net_benefit:.3f}s"
        )

        return net_benefit

    def _should_make_strategic_stop(
        self,
        lap: int,
        total_laps: int,
        driver_state: Dict[str, Any],
        weather: Dict[str, Any],
    ) -> bool:
        """
        Determine if we should make a strategic pit stop.
        """
        # Debug position calculation
        raw_position = driver_state.get("position", 1)
        track_positions = weather.get("track_positions", {})
        calculated_position = next(
            (
                pos
                for pos, d_id in track_positions.items()
                if d_id == driver_state.get("driver_id")
            ),
            raw_position,
        )

        logger.debug(
            f"\nPosition calculation in strategic stop:"
            f"\n  Raw position from state: {raw_position}"
            f"\n  Calculated from track: {calculated_position}"
            f"\n  Track position data: {track_positions}"
        )

        # Get driver characteristics and state
        driver_chars = driver_state.get("characteristics", {})
        tire_management = driver_chars.get("tire_management", 1.0)
        aggression = driver_chars.get("aggression", 1.0)

        # Get current state
        current_compound = driver_state.get("current_compound", "MEDIUM")
        tire_age = driver_state.get("tire_age", 0)
        tire_wear = driver_state.get("tire_wear", 0.0)
        current_position = driver_state.get("position", 1)
        gap_ahead = driver_state.get("gap_to_ahead", 0.0)
        grip_level = driver_state.get("grip_level", 1.0)
        pit_stops = driver_state.get("pit_stops", [])

        # Log strategic stop evaluation
        logger.debug(
            f"\nEvaluating strategic stop:"
            f"\n  Position: P{current_position}"
            f"\n  Gap Ahead: {gap_ahead:.1f}s"
            f"\n  Tire Age: {tire_age} laps"
            f"\n  Tire Wear: {tire_wear:.1f}%"
            f"\n  Compound: {current_compound}"
            f"\n  Lap: {lap}/{total_laps}"
        )

        # Calculate base stint length based on compound
        base_stint_length = self.base_tire_life.get(current_compound, 25)

        # Adjust stint length based on driver characteristics
        stint_adjustment = (
            tire_management - 1.0
        ) * 5  # ±5 laps based on tire management
        stint_adjustment -= (aggression - 1.0) * 3  # ±3 laps based on aggression
        adjusted_stint_length = max(15, base_stint_length + stint_adjustment)

        # Calculate optimal pit windows based on race length and mandatory pit rules
        laps_remaining = total_laps - lap
        mandatory_pit_needed = len(pit_stops) == 0

        # Early window: 30-45% race distance
        early_window_start = int(total_laps * 0.30)
        early_window_end = int(total_laps * 0.45)

        # Mid window: 45-60% race distance
        mid_window_start = int(total_laps * 0.45)
        mid_window_end = int(total_laps * 0.60)

        # Late window: 60-75% race distance
        late_window_start = int(total_laps * 0.60)
        late_window_end = int(total_laps * 0.75)

        # Add randomization based on driver characteristics
        window_variation = int(
            (np.random.random() - 0.5) * 6
        )  # ±3 laps random variation
        window_variation += int(
            (tire_management - 1.0) * 4
        )  # ±2 laps based on tire management
        window_variation -= int((aggression - 1.0) * 4)  # ±2 laps based on aggression

        early_window_start += window_variation
        early_window_end += window_variation
        mid_window_start += window_variation
        mid_window_end += window_variation
        late_window_start += window_variation
        late_window_end += window_variation

        # Determine if we're in a pit window
        in_early_window = early_window_start <= lap <= early_window_end
        in_mid_window = mid_window_start <= lap <= mid_window_end
        in_late_window = late_window_start <= lap <= late_window_end

        # Get position information
        current_position = driver_state.get("position", 1)
        gap_ahead = driver_state.get("gap_to_ahead", 0.0)
        gap_to_leader = driver_state.get("gap_to_leader", 0.0)

        # Adjust strategy based on position
        if current_position == 1 and gap_to_leader > 5.0:
            # Leading with good gap - be more conservative
            tire_age_appropriate = tire_age >= (adjusted_stint_length * 0.8)
        elif gap_ahead < 2.0 and current_position > 1:
            # Close to car ahead - be more aggressive
            tire_age_appropriate = tire_age >= (adjusted_stint_length * 0.6)
        else:
            # Standard case
            tire_age_appropriate = tire_age >= (adjusted_stint_length * 0.7)

        # Don't pit too close to the end unless critical
        if laps_remaining < 5:
            return False

        # Mandatory pit stop logic with position consideration
        if mandatory_pit_needed:
            if current_position == 1:
                # If leading, wait for a good window
                if (in_mid_window or in_late_window) and tire_age_appropriate:
                    return True
            else:
                # If not leading, more flexible with timing
                if in_mid_window or (in_late_window and laps_remaining > 10):
                    return True
            # Emergency pit if getting very late
            if lap > late_window_end and laps_remaining > 5:
                return True

        # Regular pit stop logic with position consideration
        elif tire_age_appropriate:
            if current_position == 1:
                # Leading - more conservative
                if in_mid_window and tire_wear > 65:
                    return True
                if in_late_window and tire_wear > 55:
                    return True
            else:
                # Not leading - standard strategy
                if in_early_window and tire_wear > 70:
                    return True
                if in_mid_window and tire_wear > 60:
                    return True
                if in_late_window and tire_wear > 50:
                    return True

        return False

    def _evaluate_pit_stop_need(self, car_state, track_state, race_state):
        """
        Enhanced pit stop evaluation with more realistic thresholds.
        """
        current_tire = car_state.get("current_tire", {})
        tire_wear = current_tire.get("wear", 0)
        tire_grip = current_tire.get("grip", 1.0)
        laps_on_tire = current_tire.get("laps", 0)
        compound = current_tire.get("compound", "UNKNOWN")

        # Get expected stint length for current compound
        expected_length = self.expected_stint_lengths.get(compound, 20)

        # Calculate wear rate and projected wear
        if laps_on_tire > 0:
            wear_rate = tire_wear / laps_on_tire
            projected_wear = tire_wear + (
                wear_rate * self.base_pit_decision_thresholds["safety_margin_laps"]
            )
        else:
            wear_rate = 0
            projected_wear = tire_wear

        # Enhanced pit stop triggers
        pit_triggers = {
            "wear_critical": projected_wear
            > self.base_pit_decision_thresholds["wear_threshold"],
            "grip_critical": tire_grip
            < self.base_pit_decision_thresholds["grip_threshold"],
            "stint_length_optimal": laps_on_tire
            >= expected_length * 0.85,  # Allow some flexibility
            "performance_delta": self._calculate_performance_delta(
                car_state, track_state
            )
            > self.base_pit_decision_thresholds["performance_delta_threshold"],
        }

        # Don't pit too early unless critical
        if laps_on_tire < self.base_pit_decision_thresholds["minimum_stint_length"]:
            return False, "Stint too short for pit stop"

        # Decision logic with detailed reason
        if pit_triggers["wear_critical"]:
            return True, f"Critical wear projection: {projected_wear:.1f}%"
        elif pit_triggers["grip_critical"] and pit_triggers["stint_length_optimal"]:
            return True, f"Low grip ({tire_grip:.2f}) and optimal stint length"
        elif pit_triggers["performance_delta"] and pit_triggers["stint_length_optimal"]:
            return True, "Performance delta exceeds threshold and stint length optimal"

        return False, "Current tire performance still acceptable"

    def _calculate_performance_delta(self, car_state, track_state):
        """
        Calculate the potential performance gain from a pit stop.
        """
        current_tire = car_state.get("current_tire", {})
        current_lap_time = car_state.get("last_lap_time", 0)

        # Estimate fresh tire performance
        fresh_tire_delta = {
            "SOFT": -1.2,  # Potential gain with fresh softs
            "MEDIUM": -0.8,  # Potential gain with fresh mediums
            "HARD": -0.5,  # Potential gain with fresh hards
        }.get(current_tire.get("compound", "UNKNOWN"), -0.8)

        # Factor in current tire degradation
        current_degradation = (1 - current_tire.get("grip", 1.0)) * 2.0

        # Calculate total potential gain
        performance_delta = current_degradation + fresh_tire_delta

        return performance_delta

    def _evaluate_end_race_strategy(
        self, lap: int, total_laps: int, current_compound: str, tire_age: int
    ) -> Tuple[bool, str]:
        """
        Evaluate if we should avoid pitting based on end-of-race considerations.

        Returns:
            Tuple of (should_avoid_pit, reason)
        """
        laps_remaining = total_laps - lap

        # Base cases to avoid pitting near race end
        if laps_remaining < 5:
            return True, "Too few laps remaining to benefit from pit stop"

        # Check if current tires can reasonably make it to the end
        compound_life = self.base_tire_life.get(current_compound, 25)
        estimated_life_left = compound_life - tire_age

        # If tires can make it to the end with acceptable performance
        if estimated_life_left >= laps_remaining * 1.1:  # 10% safety margin
            return True, "Current tires sufficient for remaining laps"

        # If very close to the end but tires are still manageable
        if laps_remaining < 8 and tire_age < compound_life * 0.9:
            return True, "Track position more valuable near race end"

        return False, "End race strategy check passed"
