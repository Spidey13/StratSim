"""
Main simulation loop orchestrating the F1 race strategy simulation.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from pathlib import Path

from ..agents.base_agent import BaseAgent
from ..agents.lap_time_agent import LapTimeAgent
from ..agents.tire_wear_agent import TireWearAgent
from ..agents.weather_agent import WeatherAgent
from ..agents.strategy_agent import StrategyAgent
from src.config.settings import (
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_WEATHER_UPDATE_INTERVAL,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceSimulator:
    """
    Main simulation loop for F1 race strategy simulation.
    This class orchestrates interaction between agents and handles
    the state of the race simulation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the race simulator with configuration.

        Args:
            config: Simulation configuration including:
                - circuit: Circuit information
                - race_laps: Total number of laps
                - weather_condition: Weather condition
                - available_compounds: Available tire compounds
                - driver_info: Driver information
        """
        self.config = config
        self.circuit = config.get("circuit", "unknown")
        self.race_laps = config.get("race_laps", 50)
        self.current_lap = 0
        self.race_history = []
        self.is_running = False

        # Initialize agents
        self.agents = self._initialize_agents()

        # Race state
        self.state = {
            "lap": 0,
            "drivers": {},
            "weather": {},
            "track_state": {},
            "events": [],
        }

        # Initialize race state
        self._initialize_race_state()

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """
        Initialize the agent ecosystem.

        Returns:
            Dictionary mapping agent names to instances
        """
        lap_time_agent = LapTimeAgent(name="LapTimeAgent")
        tire_wear_agent = TireWearAgent(name="TireWearAgent")
        weather_agent = WeatherAgent(name="WeatherAgent")
        strategy_agent = StrategyAgent(name="StrategyAgent")

        return {
            "lap_time": lap_time_agent,
            "tire_wear": tire_wear_agent,
            "weather": weather_agent,
            "strategy": strategy_agent,
        }

    def _initialize_race_state(self) -> None:
        """Initialize the race state based on configuration."""
        # Initialize driver information
        drivers = self.config.get("driver_info", {})
        for driver_id, driver_info in drivers.items():
            starting_compound = driver_info.get("starting_compound", "MEDIUM")
            self.state["drivers"][driver_id] = {
                "current_compound": starting_compound,
                "tire_age": 0,
                "lap_times": [],
                "pit_stops": [],
                "position": driver_info.get("starting_position", 0),
                "total_race_time": 0.0,
            }

        # Initialize weather based on configuration
        weather_condition = self.config.get("weather_condition", "Dry")
        self.state["weather"] = self._initialize_weather(weather_condition)

        # Initialize track state
        self.state["track_state"] = {
            "grip_level": 1.0,  # 1.0 is baseline
            "rubbered_in": False,
            "dry_line": weather_condition == "Dry",
            "wet_patches": weather_condition != "Dry",
        }

    def _initialize_weather(self, weather_condition: str) -> Dict[str, Any]:
        """
        Initialize weather based on condition string.

        Args:
            weather_condition: String describing weather

        Returns:
            Weather state dictionary
        """
        if weather_condition == "Dry":
            return {
                "condition": "Dry",
                "rainfall": 0,
                "air_temp": 25,
                "track_temp": 35,
                "humidity": 40,
                "wind_speed": 5,
                "forecast": [
                    {"lap": i, "condition": "Dry"} for i in range(1, self.race_laps + 1)
                ],
            }
        elif weather_condition == "Light Rain":
            return {
                "condition": "Light Rain",
                "rainfall": 1,
                "air_temp": 18,
                "track_temp": 22,
                "humidity": 80,
                "wind_speed": 10,
                "forecast": [
                    {"lap": i, "condition": "Light Rain"}
                    for i in range(1, self.race_laps + 1)
                ],
            }
        elif weather_condition == "Heavy Rain":
            return {
                "condition": "Heavy Rain",
                "rainfall": 3,
                "air_temp": 15,
                "track_temp": 18,
                "humidity": 95,
                "wind_speed": 15,
                "forecast": [
                    {"lap": i, "condition": "Heavy Rain"}
                    for i in range(1, self.race_laps + 1)
                ],
            }
        elif weather_condition == "Variable":
            # Create a mixed weather scenario
            forecasts = []
            for i in range(1, self.race_laps + 1):
                if i < self.race_laps * 0.3:
                    forecasts.append({"lap": i, "condition": "Dry"})
                elif i < self.race_laps * 0.4:
                    forecasts.append({"lap": i, "condition": "Light Rain"})
                elif i < self.race_laps * 0.7:
                    forecasts.append({"lap": i, "condition": "Heavy Rain"})
                else:
                    forecasts.append({"lap": i, "condition": "Light Rain"})

            return {
                "condition": "Dry",  # Starting condition
                "rainfall": 0,
                "air_temp": 22,
                "track_temp": 30,
                "humidity": 60,
                "wind_speed": 8,
                "forecast": forecasts,
            }
        else:
            # Default to dry
            return {
                "condition": "Dry",
                "rainfall": 0,
                "air_temp": 25,
                "track_temp": 35,
                "humidity": 40,
                "wind_speed": 5,
                "forecast": [
                    {"lap": i, "condition": "Dry"} for i in range(1, self.race_laps + 1)
                ],
            }

    def run_simulation(self, callback=None) -> List[Dict[str, Any]]:
        """
        Run the full race simulation.

        Args:
            callback: Optional callback function to report progress

        Returns:
            List of lap data dictionaries
        """
        self.is_running = True
        self.race_history = []

        try:
            # Simulate lap by lap
            for lap in range(1, self.race_laps + 1):
                if not self.is_running:
                    logger.info("Simulation stopped.")
                    break

                # Update current lap
                self.current_lap = lap
                self.state["lap"] = lap

                # Simulate this lap
                lap_data = self.simulate_lap()

                # Add to history
                self.race_history.append(lap_data)

                # Call progress callback if provided
                if callback:
                    callback(lap, self.race_laps, lap_data)

                # Small delay for visualization purposes
                time.sleep(0.05)

            logger.info(
                f"Simulation completed: {self.current_lap}/{self.race_laps} laps"
            )
            return self.race_history

        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            self.is_running = False
            return self.race_history

    def simulate_lap(self) -> Dict[str, Any]:
        """
        Simulate a single lap of the race.

        Returns:
            Dictionary with lap simulation data
        """
        lap_data = {"lap": self.current_lap, "drivers": {}, "weather": {}, "events": []}

        try:
            # 1. Update weather for this lap
            weather_input = {
                "lap": self.current_lap,
                "current_weather": self.state["weather"],
                "forecast": self.state["weather"].get("forecast", []),
            }
            weather_result = self.agents["weather"].process(weather_input)
            self.state["weather"] = weather_result.get("weather", self.state["weather"])
            lap_data["weather"] = self.state["weather"]

            # 2. Process each driver
            for driver_id, driver_state in self.state["drivers"].items():
                # Update tire age
                driver_state["tire_age"] += 1

                # Prepare input for strategy agent
                strategy_input = {
                    "lap": self.current_lap,
                    "total_laps": self.race_laps,
                    "driver_id": driver_id,
                    "driver_state": driver_state,
                    "weather": self.state["weather"],
                    "track_state": self.state["track_state"],
                }

                # Get strategy decision
                strategy_result = self.agents["strategy"].process(strategy_input)
                pit_decision = strategy_result.get("pit_decision", False)

                # Record pit stop if decided
                if pit_decision:
                    new_compound = strategy_result.get(
                        "new_compound", driver_state["current_compound"]
                    )
                    pit_time = 20.0  # Simplified pit time

                    # Record pit stop
                    pit_stop = {
                        "lap": self.current_lap,
                        "old_compound": driver_state["current_compound"],
                        "new_compound": new_compound,
                        "time_lost": pit_time,
                    }
                    driver_state["pit_stops"].append(pit_stop)

                    # Update driver state
                    driver_state["current_compound"] = new_compound
                    driver_state["tire_age"] = 0

                    # Add pit event
                    lap_data["events"].append(
                        {
                            "type": "pit_stop",
                            "driver": driver_id,
                            "lap": self.current_lap,
                            "details": pit_stop,
                        }
                    )

                # Prepare input for tire wear agent
                tire_input = {
                    "lap": self.current_lap,
                    "compound": driver_state["current_compound"],
                    "tire_age": driver_state["tire_age"],
                    "driver_id": driver_id,
                    "circuit_id": self.circuit,
                    "weather": self.state["weather"],
                }

                # Get tire wear prediction
                tire_result = self.agents["tire_wear"].process(tire_input)

                # Prepare input for lap time agent
                lap_time_input = {
                    "circuit_id": self.circuit,
                    "driver": driver_id,
                    "compound": driver_state["current_compound"],
                    "tire_age": driver_state["tire_age"],
                    "weather": self.state["weather"],
                    "tire_wear": tire_result.get("tire_wear", 0),
                }

                # Get lap time prediction
                lap_time_result = self.agents["lap_time"].process(lap_time_input)
                predicted_time = lap_time_result.get("predicted_laptime", 90.0)

                # Add pit time if pitted
                if pit_decision:
                    predicted_time += pit_time

                # Update driver state
                driver_state["lap_times"].append(predicted_time)
                driver_state["total_race_time"] += predicted_time

                # Record lap data for this driver
                lap_data["drivers"][driver_id] = {
                    "lap_time": predicted_time,
                    "compound": driver_state["current_compound"],
                    "tire_age": driver_state["tire_age"],
                    "tire_wear": tire_result.get("tire_wear", 0),
                    "pit_this_lap": pit_decision,
                    "total_race_time": driver_state["total_race_time"],
                }

            # 3. Calculate positions based on total race time
            sorted_drivers = sorted(
                self.state["drivers"].items(), key=lambda x: x[1]["total_race_time"]
            )

            # Update positions
            for pos, (driver_id, _) in enumerate(sorted_drivers, 1):
                self.state["drivers"][driver_id]["position"] = pos
                lap_data["drivers"][driver_id]["position"] = pos

            return lap_data

        except Exception as e:
            logger.error(f"Error simulating lap {self.current_lap}: {str(e)}")
            return {"lap": self.current_lap, "error": str(e)}

    def stop_simulation(self) -> None:
        """Stop the running simulation."""
        self.is_running = False
        logger.info("Simulation stop requested.")

    def get_results(self) -> Dict[str, Any]:
        """
        Get the final simulation results.

        Returns:
            Dictionary with simulation results
        """
        if not self.race_history:
            return {"status": "No simulation data available"}

        # Extract final state
        final_state = self.race_history[-1] if self.race_history else {}

        # Calculate statistics
        driver_results = {}
        for driver_id, driver_state in self.state["drivers"].items():
            lap_times = driver_state["lap_times"]

            driver_results[driver_id] = {
                "position": driver_state["position"],
                "total_time": driver_state["total_race_time"],
                "avg_lap_time": sum(lap_times) / len(lap_times) if lap_times else 0,
                "fastest_lap": min(lap_times) if lap_times else 0,
                "pit_stops": len(driver_state["pit_stops"]),
                "pit_stop_details": driver_state["pit_stops"],
            }

        # Sort by position
        sorted_results = dict(
            sorted(driver_results.items(), key=lambda x: x[1]["position"])
        )

        return {
            "race_completed": self.current_lap >= self.race_laps,
            "laps_completed": self.current_lap,
            "total_laps": self.race_laps,
            "results": sorted_results,
            "weather_summary": self.state["weather"],
            "circuit": self.circuit,
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export race history to a pandas DataFrame.

        Returns:
            DataFrame with race data
        """
        if not self.race_history:
            return pd.DataFrame()

        # Flatten race history into rows
        rows = []

        for lap_data in self.race_history:
            lap = lap_data.get("lap", 0)
            weather = lap_data.get("weather", {})

            for driver_id, driver_lap in lap_data.get("drivers", {}).items():
                row = {
                    "Lap": lap,
                    "Driver": driver_id,
                    "LapTime": driver_lap.get("lap_time", 0),
                    "Compound": driver_lap.get("compound", "UNKNOWN"),
                    "TireAge": driver_lap.get("tire_age", 0),
                    "TireWear": driver_lap.get("tire_wear", 0),
                    "Position": driver_lap.get("position", 0),
                    "PitStop": driver_lap.get("pit_this_lap", False),
                    "TotalTime": driver_lap.get("total_race_time", 0),
                    "Weather": weather.get("condition", "UNKNOWN"),
                    "Rainfall": weather.get("rainfall", 0),
                    "TrackTemp": weather.get("track_temp", 0),
                    "AirTemp": weather.get("air_temp", 0),
                }
                rows.append(row)

        return pd.DataFrame(rows)


class SimulationOrchestrator:
    def __init__(
        self,
        lap_time_agent: LapTimeAgent,
        tire_wear_agent: TireWearAgent,
        weather_agent: WeatherAgent,
        strategy_agent: StrategyAgent,
        max_steps: int = DEFAULT_SIMULATION_STEPS,
        weather_update_interval: int = DEFAULT_WEATHER_UPDATE_INTERVAL,
    ):
        self.agents = {
            "lap_time": lap_time_agent,
            "tire_wear": tire_wear_agent,
            "weather": weather_agent,
            "strategy": strategy_agent,
        }
        self.max_steps = max_steps
        self.weather_update_interval = weather_update_interval
        self.current_step = 0
        self.state = {}

    def initialize_simulation(self, initial_conditions: Dict):
        """Initialize the simulation with starting conditions."""
        self.state = initial_conditions.copy()
        for agent in self.agents.values():
            agent.initialize(self.state)
        logger.info("Simulation initialized with conditions: %s", initial_conditions)

    def step(self) -> Dict:
        """Execute one step of the simulation."""
        if self.current_step >= self.max_steps:
            raise StopIteration("Maximum simulation steps reached")

        # Update weather periodically
        if self.current_step % self.weather_update_interval == 0:
            weather_update = self.agents["weather"].update(self.state)
            self.state.update(weather_update)

        # Update tire wear
        tire_update = self.agents["tire_wear"].update(self.state)
        self.state.update(tire_update)

        # Calculate lap time
        lap_time = self.agents["lap_time"].predict(self.state)
        self.state["lap_time"] = lap_time

        # Get strategy recommendations
        strategy = self.agents["strategy"].decide(self.state)
        self.state.update(strategy)

        self.current_step += 1
        return self.state.copy()

    def run_simulation(self) -> List[Dict]:
        """Run the complete simulation and return the state history."""
        history = []
        try:
            while True:
                state = self.step()
                history.append(state)
        except StopIteration:
            logger.info("Simulation completed after %d steps", self.current_step)
        return history

    def save_results(self, output_path: Path):
        """Save simulation results to a file."""
        # TODO: Implement result saving logic
        pass
