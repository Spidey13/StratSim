"""
Main simulation loop orchestrating the F1 race strategy simulation.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
import time
from pathlib import Path
import sys
import numpy as np

# Adjust imports to be relative to the project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from agents.base_agent import BaseAgent
from agents.lap_time_agent import LapTimeAgent
from agents.tire_manager_agent import TireManagerAgent
from agents.weather_agent import WeatherAgent
from agents.strategy_agent import StrategyAgent
from agents.vehicle_dynamics_agent import VehicleDynamicsAgent
from agents.gap_effects_agent import GapEffectsAgent
from config.settings import (
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_WEATHER_UPDATE_INTERVAL,
)

# Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure logger for this module is set to DEBUG


class RaceSimulator:
    """
    Main simulation loop for F1 race strategy simulation.
    This class orchestrates interaction between agents and handles
    the state of the race simulation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the race simulator with configuration.
        """
        self.config = config
        self.circuit_name = config.get("circuit", "unknown")
        self.race_laps = config.get("race_laps", 50)
        self.current_lap = 0
        self.race_history = []
        self.is_running = False
        self.pit_time_penalty = config.get("pit_time_penalty", 20.0)
        self.agents = self._initialize_agents()
        self.state = {
            "lap": 0,
            "drivers": {},
            "weather": {},
            "track_state": {},
            "events": [],
            "simulation_year": config.get("year", 2024),
        }
        self._initialize_race_state()

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """
        Initialize the agent ecosystem.
        """
        lap_time_agent = LapTimeAgent(name="LapTimeAgent")
        # Initialize TireManagerAgent, which internally uses TireWearAgent
        tire_manager_agent = TireManagerAgent(name="TireManagerAgent")
        weather_agent = WeatherAgent(
            name="WeatherAgent", api_key=self.config.get("weather_api_key")
        )
        strategy_agent = StrategyAgent(name="StrategyAgent")
        vehicle_dynamics_agent = VehicleDynamicsAgent(name="VehicleDynamicsAgent")
        gap_effects_agent = GapEffectsAgent(name="GapEffectsAgent")

        return {
            "lap_time": lap_time_agent,
            "tire_manager": tire_manager_agent,
            "weather": weather_agent,
            "strategy": strategy_agent,
            "vehicle_dynamics": vehicle_dynamics_agent,
            "gap_effects": gap_effects_agent,
        }

    def _initialize_race_state(self) -> None:
        """Initialize the state of the race, drivers, and weather."""
        # Initialize weather and track state
        weather_condition = self.config.get("weather_condition", "Dry")
        self.state["weather"] = self._initialize_weather(weather_condition)
        self.state["track_state"] = {
            "dry_line": weather_condition == "Dry",
            "track_temp": self.state["weather"]["track_temp"],
            "air_temp": self.state["weather"]["air_temp"],
        }

        logger.debug(f"Initialized Weather: {self.state['weather']}")
        logger.debug(f"Initialized Track State: {self.state['track_state']}")

        # Get sorted drivers by grid position
        sorted_drivers = sorted(
            self.config.get("drivers", {}).items(),
            key=lambda x: x[1].get("grid_position", 999),
        )

        # Initialize driver states
        self.state["drivers"] = {}
        position = 1

        for driver_id_code_from_config, driver_specific_config_data in sorted_drivers:
            # Get driver characteristics or generate random ones
            driver_characteristics = driver_specific_config_data.get(
                "characteristics", {}
            )
            if not driver_characteristics:
                driver_characteristics = {
                    "tire_management": 0.8 + (np.random.random() * 0.4),
                    "aggression": 0.8 + (np.random.random() * 0.4),
                    "consistency": 0.8 + (np.random.random() * 0.4),
                    "wet_weather": 0.8 + (np.random.random() * 0.4),
                }

            # Get initial compound
            actual_starting_compound = driver_specific_config_data.get(
                "starting_compound", "MEDIUM"
            )

            # Initialize speeds
            initial_speeds = {
                "SpeedST": 280.0,  # Starting speed
                "SpeedI1": 200.0,  # Intermediate 1 speed
                "SpeedI2": 150.0,  # Intermediate 2 speed
                "SpeedFL": 280.0,  # Finish line speed
                "SpeedST_Diff": 0.0,
                "SpeedI1_Diff": 0.0,
                "SpeedI2_Diff": 0.0,
                "SpeedFL_Diff": 0.0,
            }

            # Initialize tire temperature based on compound
            initial_tire_temp = {
                "SOFT": 90.0,
                "MEDIUM": 85.0,
                "HARD": 80.0,
                "INTERMEDIATE": 65.0,
                "WET": 55.0,
            }.get(actual_starting_compound, 85.0)

            # Create driver state
            driver_state = {
                "driver_name": driver_specific_config_data.get(
                    "driver_name", "Unknown Driver"
                ),
                "team_name": driver_specific_config_data.get(
                    "team_name", "Unknown Team"
                ),
                "current_compound": actual_starting_compound,
                "tire_age": 0,
                "tire_wear": 0.0,
                "tire_temp": initial_tire_temp,
                "grip_level": 1.0,
                "lap_times": [],
                "pit_stops": [],
                "position": position,
                "grid_position": position,
                "total_race_time": 0.0,
                "prev_lap_degradation_s": 0.0,
                "estimated_degradation_s_this_lap": 0.0,
                "driver_id_code": driver_id_code_from_config,
                "team_id_code": driver_specific_config_data.get(
                    "team_id", driver_specific_config_data.get("team_name", "UNKNOWN")
                ),
                "characteristics": driver_characteristics,
                "in_optimal_window": True,
                "tire_health": "GOOD",
                "track_evolution": 0.92,  # Starting grip level
                "compounds_used": [actual_starting_compound],
                "tire_history": [
                    (actual_starting_compound, 0)
                ],  # (compound, age) tuples
                **initial_speeds,
            }

            self.state["drivers"][driver_id_code_from_config] = driver_state
            position += 1

            logger.debug(
                f"Initialized Driver State for {driver_id_code_from_config}: {driver_state}"
            )

        logger.info(f"Initialized {len(self.state['drivers'])} drivers")

    def _initialize_weather(self, weather_condition: str) -> Dict[str, Any]:
        """Initialize weather based on condition string."""
        if weather_condition == "Dry":
            return {
                "condition": "Dry",
                "rainfall": 0.0,
                "track_temp": 35.0,  # Reasonable track temperature for dry conditions
                "air_temp": 25.0,  # Reasonable air temperature for dry conditions
                "humidity": 50.0,  # Moderate humidity
                "wind_speed": 5.0,  # Light wind
            }
        elif weather_condition == "Wet":
            return {
                "condition": "Wet",
                "rainfall": 2.0,  # Moderate rain
                "track_temp": 20.0,  # Lower track temperature due to rain
                "air_temp": 18.0,  # Lower air temperature in wet conditions
                "humidity": 85.0,  # High humidity
                "wind_speed": 15.0,  # Stronger winds
            }
        else:
            return {
                "condition": "Mixed",
                "rainfall": 0.5,  # Light rain
                "track_temp": 25.0,  # Intermediate track temperature
                "air_temp": 22.0,  # Intermediate air temperature
                "humidity": 70.0,  # Moderate to high humidity
                "wind_speed": 10.0,  # Moderate wind
            }

    def run_simulation(self, callback=None) -> List[Dict[str, Any]]:
        """
        Run the full race simulation.
        """
        self.is_running = True
        self.race_history = []
        self._initialize_race_state()

        for lap_num in range(1, self.race_laps + 1):
            if not self.is_running:
                logger.info("Simulation stopped.")
                break
            self.current_lap = lap_num
            self.state["lap"] = lap_num
            lap_data = self.simulate_lap()
            self.race_history.append(lap_data)
            if callback:
                callback(lap_num, self.race_laps, lap_data)
            time.sleep(0.05)

        logger.info(f"Simulation completed: {self.current_lap}/{self.race_laps} laps")
        self.is_running = False
        return self.race_history

    def _update_driver_positions(self) -> None:
        """Update driver positions based on total race times."""
        # Sort drivers by total race time
        sorted_drivers = sorted(
            self.state["drivers"].items(), key=lambda x: x[1]["total_race_time"]
        )

        # Update positions and calculate gaps
        leader_time = sorted_drivers[0][1]["total_race_time"]
        prev_time = leader_time

        for position, (driver_id, driver_state) in enumerate(sorted_drivers, 1):
            # Update position
            self.state["drivers"][driver_id]["position"] = position

            # Calculate gaps
            gap_to_leader = driver_state["total_race_time"] - leader_time
            gap_ahead = (
                driver_state["total_race_time"] - prev_time if position > 1 else 0.0
            )

            # Update gaps in driver state
            self.state["drivers"][driver_id]["gap_to_leader"] = gap_to_leader
            self.state["drivers"][driver_id]["gap_ahead"] = gap_ahead

            # Store current time for next iteration
            prev_time = driver_state["total_race_time"]

            # Debug log
            logger.debug(
                f"Updated position for {driver_id}: P{position}, "
                f"Gap to leader: {gap_to_leader:.3f}s, "
                f"Gap ahead: {gap_ahead:.3f}s"
            )

    def simulate_lap(self) -> Dict[str, Any]:
        """Simulate one lap of the race."""
        logger.debug(f"\n=== Starting Lap {self.current_lap} Simulation ===")

        # Process weather at the start of each lap
        weather_input = {
            "lap": self.current_lap,
            "weather": self.state["weather"],
            "circuit_id": self.circuit_name,
        }
        weather_output = self.agents["weather"].process(weather_input)
        self.state["weather"] = weather_output["weather"]

        lap_specific_outputs = {}  # To store results like performance_factors, gap_effects, etc.

        # Step 1: Process each driver's lap, update their states in self.state["drivers"]
        # (e.g., tire wear, total_race_time), and collect lap-specific outputs.
        for driver_id, driver_state_ref in self.state["drivers"].items():
            # Get current tire state for logging or pre-agent logic if needed
            current_tire_state_log = {
                "wear": driver_state_ref.get("tire_wear", 0),
                "age": driver_state_ref.get("tire_age", 0),
                "compound": driver_state_ref.get("current_compound", "UNKNOWN"),
            }
            logger.debug(f"\n--- Processing {driver_id} ---")
            logger.debug(
                f"Starting tire state: wear={current_tire_state_log['wear']:.1f}%, age={current_tire_state_log['age']}, compound={current_tire_state_log['compound']}"
            )

            # Strategy Agent
            strategy_input = {
                "lap": self.current_lap,
                "total_laps": self.race_laps,
                "weather": self.state["weather"],
                "track_state": self.state["track_state"],
                "circuit_id": self.circuit_name,
                "driver_id": driver_id,
                "driver_state": driver_state_ref,  # Pass reference to live state
                "pit_time_penalty": self.pit_time_penalty,
                "driver_characteristics": driver_state_ref.get("characteristics", {}),
                "track_positions": {
                    d_id: self.state["drivers"][d_id][
                        "position"
                    ]  # Use current positions
                    for d_id in self.state["drivers"]
                },
            }
            strategy_result = self.agents["strategy"].process(strategy_input)
            pit_decision = strategy_result.get("pit_decision", False)
            new_compound = strategy_result.get("new_compound")

            if pit_decision:
                logger.debug(
                    f"PIT DECISION - Driver {driver_id} pitting for {new_compound}"
                )
                # Log pre-pit tire state directly from driver_state_ref as it's live
                logger.debug(
                    f"Pre-pit tire state: wear={driver_state_ref['tire_wear']:.1f}%, age={driver_state_ref['tire_age']}"
                )

            # Tire Manager Agent
            tire_manager_input = {
                "circuit_id": self.circuit_name,
                "driver_id": driver_id,
                "team_id": driver_state_ref.get("team_id_code", "Unknown_Team"),
                "year": self.state.get("simulation_year", 2024),
                "weather": self.state["weather"],
                "current_lap": self.current_lap,
                "laps_remaining": self.race_laps - self.current_lap,
                "is_pit_lap": pit_decision,
                "compound": new_compound if pit_decision else None,
                **driver_state_ref,  # Spread the current state
            }
            tire_manager_output = self.agents["tire_manager"].process(
                tire_manager_input
            )
            driver_state_ref.update(
                tire_manager_output
            )  # Update self.state["drivers"][driver_id]

            # Vehicle Dynamics Agent
            vehicle_dynamics_input = {
                "tire_condition": {
                    "wear": driver_state_ref["tire_wear"],
                    "grip": driver_state_ref["grip_level"],
                    "compound": driver_state_ref["current_compound"],
                },
                "track_state": {
                    "temperature": self.state["weather"]["track_temp"],
                    "grip": driver_state_ref[
                        "track_evolution"
                    ],  # This was from driver_state
                    "weather": self.state["weather"],
                },
                "car_setup": driver_state_ref.get("car_setup", {}),
            }
            dynamics_result = self.agents["vehicle_dynamics"].process(
                vehicle_dynamics_input
            )
            base_laptime = dynamics_result["predicted_lap_time"]

            # Gap Effects Agent
            gap_effects_input = {
                "gap_ahead": driver_state_ref.get("gap_ahead", 10.0),
                "base_lap_time": base_laptime,
                "track_characteristics": {
                    "dirty_air_sensitivity": 1.0,
                    "drs_zones": 2,
                },
                "current_lap": self.current_lap,
                "fuel_load": max(0, 100 - (self.current_lap * 2)),
            }
            gap_effects = self.agents["gap_effects"].process(gap_effects_input)
            predicted_laptime = base_laptime + gap_effects["lap_time_delta"]

            if pit_decision:
                predicted_laptime += self.pit_time_penalty
                driver_state_ref["pit_stops"].append(
                    self.current_lap
                )  # Update self.state
                logger.debug(
                    f"Post-pit tire state: wear={driver_state_ref['tire_wear']:.1f}%, age={driver_state_ref['tire_age']}, compound={driver_state_ref['current_compound']}"
                )

            # Update total race time & lap times in self.state["drivers"][driver_id]
            driver_state_ref["lap_times"].append(predicted_laptime)
            driver_state_ref["total_race_time"] += predicted_laptime

            # Store lap-specific calculated values (not part of core driver_state but needed for history)
            lap_specific_outputs[driver_id] = {
                "performance_factors": dynamics_result.get("performance_factors", {}),
                "gap_effects_details": gap_effects,  # Renamed to avoid clash if gap_effects is also a top-level key
                "lap_time_calculated": predicted_laptime,  # Renamed to avoid clash with "lap_times" list in state
                "base_lap_time_calculated": base_laptime,  # Renamed
                "pit_this_lap_decision": pit_decision,  # Renamed
            }
            logger.debug(
                f"Final tire state for {driver_id} after processing: wear={driver_state_ref['tire_wear']:.1f}%, age={driver_state_ref['tire_age']}, compound={driver_state_ref['current_compound']}\\n"
            )

        # Step 2: All drivers' states (esp. total_race_time) are updated for the current lap.
        # Now, update their official positions based on these times.
        self._update_driver_positions()

        # Step 3: Construct lap_data_for_history using the now fully updated self.state["drivers"]
        # (which includes the correct end-of-lap positions) and the collected lap-specific outputs.
        lap_data_for_history = {
            "lap": self.current_lap,
            "weather": self.state["weather"].copy(),
            "track_state": self.state["track_state"].copy(),
            "drivers": {},
        }

        for driver_id, final_driver_state_ref in self.state["drivers"].items():
            # final_driver_state_ref is self.state["drivers"][driver_id]
            # It contains the updated position, total_race_time, tire_wear, etc.

            driver_lap_record = (
                final_driver_state_ref.copy()
            )  # Make a copy for the history record

            # Add/overwrite with the specific calculated values for this lap from lap_specific_outputs
            # These were the values previously spread into the history record directly.
            lap_calcs = lap_specific_outputs.get(driver_id, {})

            driver_lap_record.update(
                lap_calcs.get("performance_factors", {})
            )  # Spread dict
            driver_lap_record.update(
                lap_calcs.get("gap_effects_details", {})
            )  # Spread dict
            driver_lap_record["lap_time"] = lap_calcs.get(
                "lap_time_calculated"
            )  # Specific key for the single lap's time
            driver_lap_record["base_lap_time"] = lap_calcs.get(
                "base_lap_time_calculated"
            )
            driver_lap_record["pit_this_lap"] = lap_calcs.get("pit_this_lap_decision")
            # current_compound, tire_age, tire_wear are already in final_driver_state_ref

            lap_data_for_history["drivers"][driver_id] = driver_lap_record

        return lap_data_for_history

    def stop_simulation(self) -> None:
        """Stop the running simulation."""
        self.is_running = False
        # logger.info("Simulation stop requested.")

    def get_results(self) -> Dict[str, Any]:
        """Get the final simulation results."""
        initial_weather = {}
        final_weather = {}
        full_forecast = []

        if self.race_history:
            initial_weather_lap_entry = self.race_history[0].get("weather", {})
            final_weather_lap_entry = self.race_history[-1].get("weather", {})

            initial_weather = {
                "condition": initial_weather_lap_entry.get("condition", "N/A"),
                "rainfall": initial_weather_lap_entry.get("rainfall", "N/A"),
                "air_temp": initial_weather_lap_entry.get("air_temp", "N/A"),
                "track_temp": initial_weather_lap_entry.get("track_temp", "N/A"),
                "humidity": initial_weather_lap_entry.get("humidity", "N/A"),
                "wind_speed": initial_weather_lap_entry.get("wind_speed", "N/A"),
            }
            final_weather = {
                "condition": final_weather_lap_entry.get("condition", "N/A"),
                "rainfall": final_weather_lap_entry.get("rainfall", "N/A"),
                "air_temp": final_weather_lap_entry.get("air_temp", "N/A"),
                "track_temp": final_weather_lap_entry.get("track_temp", "N/A"),
                "humidity": final_weather_lap_entry.get("humidity", "N/A"),
                "wind_speed": final_weather_lap_entry.get("wind_speed", "N/A"),
            }
            # The forecast is part of each weather entry in history, and should be the same.
            # It's also set during _initialize_weather and stored in self.state["weather"] initially.
            full_forecast = initial_weather_lap_entry.get("forecast", [])
        elif self.state.get(
            "weather"
        ):  # Fallback if race_history is empty but state was initialized
            initial_weather_state = self.state["weather"]
            initial_weather = {
                "condition": initial_weather_state.get("condition", "N/A"),
                "rainfall": initial_weather_state.get("rainfall", "N/A"),
                "air_temp": initial_weather_state.get("air_temp", "N/A"),
                "track_temp": initial_weather_state.get("track_temp", "N/A"),
                "humidity": initial_weather_state.get("humidity", "N/A"),
                "wind_speed": initial_weather_state.get("wind_speed", "N/A"),
            }
            final_weather = initial_weather  # If no history, final is same as initial
            full_forecast = initial_weather_state.get("forecast", [])

        weather_summary_data = {
            "initial_track_temp": initial_weather.get("track_temp", "N/A"),
            "initial_air_temp": initial_weather.get("air_temp", "N/A"),
            "initial_humidity": initial_weather.get("humidity", "N/A"),
            "condition": final_weather.get("condition", "N/A"),  # Final condition
            "rainfall": final_weather.get("rainfall", "N/A"),  # Final rainfall
            "wind_speed": final_weather.get("wind_speed", "N/A"),  # Final wind_speed
            "track_temp": final_weather.get("track_temp", "N/A"),  # Final track_temp
            "air_temp": final_weather.get("air_temp", "N/A"),  # Final air_temp
            "forecast": full_forecast,
        }

        return {
            "status": "Completed",
            "results": self.state["drivers"],
            "history": self.race_history,
            "weather_summary": weather_summary_data,  # Added weather summary
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export race history to a pandas DataFrame.
        """
        if not self.race_history:
            return pd.DataFrame()
        rows = []
        for lap_entry in self.race_history:
            lap = lap_entry.get("lap", 0)
            weather_at_lap = lap_entry.get("weather", {})
            for driver_id, driver_lap_data in lap_entry.get("drivers", {}).items():
                rows.append(
                    {
                        "Lap": lap,
                        "Driver": driver_id,  # This is the column that streamlit_app.py expects
                        "DriverName": driver_lap_data.get("driver_name", "N/A"),
                        "TeamName": driver_lap_data.get("team_name", "N/A"),
                        "LapTime": driver_lap_data.get("lap_time", 0),
                        "Compound": driver_lap_data.get("current_compound", "UNKNOWN"),
                        "TireAge": driver_lap_data.get("tire_age", 0),
                        "TireWearPercentage": driver_lap_data.get("tire_wear", 0),
                        "EstimatedDegradationPerLap_s": driver_lap_data.get(
                            "estimated_degradation_s_this_lap", 0.0
                        ),
                        "GripLevel": driver_lap_data.get("grip_level", 1.0),
                        "Position": driver_lap_data.get("position", 0),
                        "PitStop": driver_lap_data.get("pit_this_lap", False),
                        "TotalTime": driver_lap_data.get("total_race_time", 0),
                        "SpeedST": driver_lap_data.get("SpeedST", 0.0),
                        "SpeedI1": driver_lap_data.get("SpeedI1", 0.0),
                        "SpeedI2": driver_lap_data.get("SpeedI2", 0.0),
                        "SpeedFL": driver_lap_data.get("SpeedFL", 0.0),
                        "SpeedST_Diff": driver_lap_data.get("SpeedST_Diff", 0.0),
                        "SpeedI1_Diff": driver_lap_data.get("SpeedI1_Diff", 0.0),
                        "SpeedI2_Diff": driver_lap_data.get("SpeedI2_Diff", 0.0),
                        "SpeedFL_Diff": driver_lap_data.get("SpeedFL_Diff", 0.0),
                        "WeatherCondition": weather_at_lap.get("condition", "UNKNOWN"),
                        "Rainfall": weather_at_lap.get("rainfall", 0),
                        "TrackTemp": weather_at_lap.get("track_temp", 0),
                        "AirTemp": weather_at_lap.get("air_temp", 0),
                    }
                )
        return pd.DataFrame(rows)
