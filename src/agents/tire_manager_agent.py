"""
Tire manager agent that tracks tire state and provides strategic guidance.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

from .base_agent import BaseAgent
from .tire_wear_agent import TireWearAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            name="TireWearHelper", model_path=tire_wear_model_path
        )

        # Initialize empty tire state
        self.reset_state()

        # Define cliff thresholds for each compound (percentage wear)
        self.cliff_thresholds = {
            "SOFT": 65.0,  # Softs degrade more quickly
            "MEDIUM": 75.0,  # Mediums have a longer life
            "HARD": 85.0,  # Hards are most durable
            "INTERMEDIATE": 60.0,  # Intermediates degrade quickly in changing conditions
            "WET": 40.0,  # Wets degrade quickly when track dries
        }

        # Warning thresholds (percentage of cliff threshold)
        self.warning_threshold = 0.8  # 80% of cliff threshold

    def reset_state(self) -> None:
        """Reset the tire state to default values."""
        self.state = {
            "current_compound": "MEDIUM",
            "tire_age": 0,
            "tire_wear": 0.0,
            "grip_level": 1.0,
            "in_optimal_window": True,
            "tire_health": "new",  # new, good, marginal, critical, expired
            "last_wear_rate": 0.0,  # wear per lap
            "pit_recommendation": None,
            "compounds_used": [],
            "tire_history": [],  # List of (compound, age) tuples for all stints
        }

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

        Returns:
            Dictionary containing updated tire state and recommendations
        """
        current_lap = inputs.get("current_lap", 0)
        compound = inputs.get("compound", None)  # New compound if pitting
        is_pit_lap = inputs.get("is_pit_lap", False)
        laps_remaining = inputs.get("laps_remaining", 0)

        # If pitting or first lap, set new compound
        if is_pit_lap or (current_lap == 1 and not self.state["compounds_used"]):
            if compound:
                # Record previous tire in history if not first lap
                if current_lap > 1:
                    self.state["tire_history"].append(
                        (self.state["current_compound"], self.state["tire_age"])
                    )

                # Set new compound
                self.state["current_compound"] = compound.upper()
                self.state["tire_age"] = 0
                self.state["tire_wear"] = 0.0
                self.state["grip_level"] = 1.0
                self.state["in_optimal_window"] = True
                self.state["tire_health"] = "new"
                self.state["last_wear_rate"] = 0.0

                # Track compounds used
                if compound.upper() not in self.state["compounds_used"]:
                    self.state["compounds_used"].append(compound.upper())

                logger.info(f"New {compound} tires fitted")
            else:
                logger.warning("Pit lap indicated but no compound specified")
        else:
            # Update tire age if not pitting
            self.state["tire_age"] += 1

        # Prepare inputs for tire wear agent
        wear_inputs = {
            "compound": self.state["current_compound"],
            "tire_age": self.state["tire_age"],
            "circuit_id": inputs.get("circuit_id", "unknown"),
            "driver_id": inputs.get("driver_id", "unknown"),
            "weather": inputs.get("weather", {}),
        }

        # Calculate updated wear
        wear_result = self.tire_wear_agent.process(wear_inputs)

        # Update state with wear information
        self.state["tire_wear"] = wear_result["tire_wear"]
        self.state["grip_level"] = wear_result["grip_level"]
        self.state["in_optimal_window"] = wear_result["optimal_window"]

        # Calculate wear rate (change in wear per lap)
        if self.state["tire_age"] > 1:
            self.state["last_wear_rate"] = wear_result["tire_wear"] - self.state.get(
                "tire_wear", 0
            )

        # Update tire health status
        self._update_tire_health()

        # Generate pit recommendation
        pit_recommendation = self._generate_pit_recommendation(
            laps_remaining, inputs.get("strategy", {})
        )
        self.state["pit_recommendation"] = pit_recommendation

        return {
            "compound": self.state["current_compound"],
            "tire_age": self.state["tire_age"],
            "tire_wear": self.state["tire_wear"],
            "grip_level": self.state["grip_level"],
            "tire_health": self.state["tire_health"],
            "in_optimal_window": self.state["in_optimal_window"],
            "wear_factors": wear_result["wear_factors"],
            "wear_rate": self.state["last_wear_rate"],
            "pit_recommendation": pit_recommendation,
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
        Generate pit stop recommendations based on tire state.

        Args:
            laps_remaining: Number of laps remaining
            strategy: Current race strategy information

        Returns:
            Dictionary with pit recommendation details
        """
        compound = self.state["current_compound"]
        wear_pct = self.state["tire_wear"]
        health = self.state["tire_health"]
        cliff = self.cliff_thresholds.get(compound, 75.0)

        # Default values
        should_pit = False
        urgency = "none"
        reason = None
        recommended_compound = None

        # Check if we're approaching cliff
        if health == "critical":
            should_pit = True
            urgency = "high"
            reason = f"Critical tire wear ({wear_pct:.1f}%)"
        elif health == "marginal":
            should_pit = laps_remaining > 5  # Don't pit if few laps remaining
            urgency = "medium"
            reason = f"Marginal tire wear ({wear_pct:.1f}%)"

        # If we should pit, recommend a compound
        if should_pit:
            recommended_compound = self._recommend_compound(laps_remaining, strategy)

        return {
            "should_pit": should_pit,
            "urgency": urgency,
            "reason": reason,
            "laps_until_cliff": self._estimate_laps_until_cliff(),
            "recommended_compound": recommended_compound,
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

    def _recommend_compound(self, laps_remaining: int, strategy: Dict[str, Any]) -> str:
        """
        Recommend tire compound based on remaining laps and strategy.

        Args:
            laps_remaining: Number of laps remaining
            strategy: Current race strategy information

        Returns:
            Recommended compound for next stint
        """
        # Get weather info from strategy if available
        weather = strategy.get("weather_forecast", {}).get("current", {})
        rainfall = weather.get("rainfall", 0)

        # If it's raining, recommend appropriate rain tire
        if rainfall > 2.0:
            return "WET"
        elif rainfall > 0:
            return "INTERMEDIATE"

        # For dry conditions, base on laps remaining
        remaining_pct = laps_remaining / strategy.get("total_laps", 50)
        used_compounds = self.state["compounds_used"]

        # Check if we need to use a mandatory compound
        mandatory_compounds = strategy.get("mandatory_compounds", ["MEDIUM", "HARD"])
        unused_mandatory = [c for c in mandatory_compounds if c not in used_compounds]

        if unused_mandatory and laps_remaining > 5:
            # Prioritize unused mandatory compound with appropriate durability
            if remaining_pct > 0.4 and "HARD" in unused_mandatory:
                return "HARD"
            elif remaining_pct > 0.2 and "MEDIUM" in unused_mandatory:
                return "MEDIUM"
            else:
                return unused_mandatory[0]  # Use any mandatory compound

        # No mandatory constraints, optimize for performance
        if laps_remaining > 25:
            return "HARD"
        elif laps_remaining > 15:
            return "MEDIUM"
        else:
            return "SOFT"
