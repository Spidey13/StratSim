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
        # Initialize overall race state
        self.state = {
            "drivers": {},
            "weather": self.config.get("weather_condition", "Dry"),
            "track_status": {},
            "lap_times": {},
            "race_results": {},
            "safety_car": {"status": False, "laps_out": 0},
            "virtual_safety_car": {"status": False, "laps_out": 0},
            "red_flag": False,
            "current_lap": 0,
            "total_laps": self.config.get("race_laps", 50),
            "circuit_name": self.config.get("circuit", "Unknown Circuit"),
            "year": self.config.get("year", 2023),
        }

        drivers_config_from_sim_config = self.config.get("driver_info", {})

        # Sort drivers by their starting positions
        sorted_drivers = sorted(
            drivers_config_from_sim_config.items(),
            key=lambda x: x[1].get(
                "starting_position", 999
            ),  # Use high default for unspecified
        )

        # Initialize state for each driver based on config, maintaining grid order
        for position, (
            driver_id_code_from_config,
            driver_specific_config_data,
        ) in enumerate(sorted_drivers, 1):
            actual_starting_compound = driver_specific_config_data.get(
                "starting_compound", "MEDIUM"
            )

            # Initial speed values
            initial_speeds = {
                "SpeedST": driver_specific_config_data.get("SpeedST", 280.0),
                "SpeedI1": driver_specific_config_data.get("SpeedI1", 200.0),
                "SpeedI2": driver_specific_config_data.get("SpeedI2", 150.0),
                "SpeedFL": driver_specific_config_data.get("SpeedFL", 260.0),
            }

            # Get driver characteristics
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

            self.state["drivers"][driver_id_code_from_config] = {
                "driver_name": driver_specific_config_data.get(
                    "driver_name", "Unknown Driver"
                ),
                "team_name": driver_specific_config_data.get(
                    "team_name", "Unknown Team"
                ),
                "current_compound": actual_starting_compound,
                "tire_age": 0,
                "tire_wear": 0.0,
                "grip_level": 1.0,
                "lap_times": [],
                "pit_stops": [],
                "position": position,  # Use the enumerated position
                "grid_position": position,  # Store original grid position
                "total_race_time": 0.0,
                "prev_lap_degradation_s": 0.0,
                "driver_id_code": driver_id_code_from_config,
                "team_id_code": driver_specific_config_data.get(
                    "team_id", driver_specific_config_data.get("team_name", "UNKNOWN")
                ),
                "characteristics": driver_characteristics,
                **initial_speeds,
            }

        weather_condition = self.config.get("weather_condition", "Dry")
        self.state["weather"] = self._initialize_weather(weather_condition)
        self.state["track_state"] = {
            "dry_line": weather_condition == "Dry",
        }

    def _initialize_weather(self, weather_condition: str) -> Dict[str, Any]:
        """Initialize weather based on condition string."""
        # This function can be expanded with more detailed initial weather states
        # For now, it sets up a basic structure with a forecast
        base_weather = {}
        if weather_condition == "Dry":
            base_weather = {
                "condition": "Dry",
                "rainfall": 0,
                "air_temp": 25,
                "track_temp": 35,
                "humidity": 40,
                "wind_speed": 5,
            }
        elif weather_condition == "Light Rain":
            base_weather = {
                "condition": "Light Rain",
                "rainfall": 1,
                "air_temp": 18,
                "track_temp": 22,
                "humidity": 80,
                "wind_speed": 10,
            }
        elif weather_condition == "Heavy Rain":
            base_weather = {
                "condition": "Heavy Rain",
                "rainfall": 3,
                "air_temp": 15,
                "track_temp": 18,
                "humidity": 95,
                "wind_speed": 15,
            }
        else:  # Variable and default
            base_weather = {
                "condition": "Dry",
                "rainfall": 0,
                "air_temp": 22,
                "track_temp": 30,
                "humidity": 60,
                "wind_speed": 8,
            }

        # Generate a forecast
        forecast = []
        if weather_condition == "Variable":
            for i in range(1, self.race_laps + 1):
                if i < self.race_laps * 0.4:
                    forecast.append({"lap": i, "condition": "Dry"})
                else:
                    forecast.append({"lap": i, "condition": "Light Rain"})
        else:
            for i in range(1, self.race_laps + 1):
                forecast.append({"lap": i, "condition": base_weather["condition"]})

        base_weather["forecast"] = forecast
        return base_weather

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

    def simulate_lap(self) -> Dict[str, Any]:
        """
        Simulate one lap of the race.
        """
        # Process weather at the start of each lap
        weather_input = {
            "lap": self.current_lap,
            "weather": self.state["weather"],
            "circuit_id": self.circuit_name,
        }
        weather_output = self.agents["weather"].process(weather_input)
        self.state["weather"] = weather_output["weather"]

        lap_data_for_history = {
            "lap": self.current_lap,
            "weather": self.state["weather"],
            "track_state": self.state["track_state"],
            "drivers": {},
        }

        # Calculate positions and gaps before processing drivers
        sorted_drivers = sorted(
            self.state["drivers"].items(), key=lambda item: item[1]["total_race_time"]
        )

        # Update track positions dictionary
        track_positions = {}
        leader_time = sorted_drivers[0][1]["total_race_time"]

        for pos, (driver_id, driver_data) in enumerate(sorted_drivers, 1):
            track_positions[driver_id] = pos

            # Calculate gaps
            gap_to_leader = driver_data["total_race_time"] - leader_time
            gap_ahead = 0.0
            if pos > 1:
                gap_ahead = (
                    driver_data["total_race_time"]
                    - sorted_drivers[pos - 2][1]["total_race_time"]
                )

            # Update driver state with position and gaps
            self.state["drivers"][driver_id].update(
                {
                    "position": pos,
                    "gap_to_leader": gap_to_leader,
                    "gap_to_ahead": gap_ahead,
                    "positions_gained": self.state["drivers"][driver_id][
                        "grid_position"
                    ]
                    - pos,
                }
            )

        # Update track state with position data
        self.state["track_state"]["positions"] = track_positions

        # Process each driver's lap
        for driver_id, driver_state in self.state["drivers"].items():
            # Capture the degradation from the PREVIOUS lap to be used by LapTimeAgent for THIS lap.
            # This value should have been set at the end of the previous lap's processing for this driver.
            degradation_affecting_current_lap_time = driver_state[
                "prev_lap_degradation_s"
            ]

            # 4. Vehicle Dynamics Agent: Simulate speeds for the lap
            dynamics_input = {
                "circuit_id": self.circuit_name,
                "driver_id": driver_id,
                "grip_level": driver_state["grip_level"],
                "compound": driver_state["current_compound"],
                "lap": self.current_lap,
                "total_laps": self.race_laps,
                "current_SpeedST": driver_state["SpeedST"],
                "current_SpeedI1": driver_state["SpeedI1"],
                "current_SpeedI2": driver_state["SpeedI2"],
                "current_SpeedFL": driver_state["SpeedFL"],
            }
            vehicle_dynamics_output = self.agents["vehicle_dynamics"].process(
                dynamics_input
            )
            # Update driver state with new speeds immediately for subsequent agents
            driver_state["SpeedST"] = vehicle_dynamics_output.get(
                "SpeedST", driver_state["SpeedST"]
            )
            driver_state["SpeedI1"] = vehicle_dynamics_output.get(
                "SpeedI1", driver_state["SpeedI1"]
            )
            driver_state["SpeedI2"] = vehicle_dynamics_output.get(
                "SpeedI2", driver_state["SpeedI2"]
            )
            driver_state["SpeedFL"] = vehicle_dynamics_output.get(
                "SpeedFL", driver_state["SpeedFL"]
            )
            driver_state["SpeedST_Diff"] = vehicle_dynamics_output.get(
                "SpeedST_Diff", 0.0
            )
            driver_state["SpeedI1_Diff"] = vehicle_dynamics_output.get(
                "SpeedI1_Diff", 0.0
            )
            driver_state["SpeedI2_Diff"] = vehicle_dynamics_output.get(
                "SpeedI2_Diff", 0.0
            )
            driver_state["SpeedFL_Diff"] = vehicle_dynamics_output.get(
                "SpeedFL_Diff", 0.0
            )

            # 3. Tire Manager Agent: Update tire state
            # logger.debug(
            #     f"SimLoop: Weather state BEFORE TireManagerAgent for lap {self.current_lap}, driver {driver_id}: {self.state['weather']}"
            # )
            laps_remaining_in_race = self.race_laps - self.current_lap
            tire_manager_input = {
                "circuit_id": self.circuit_name,
                "driver_id": driver_id,
                "team_id": driver_state.get("team_id_code", "Unknown_Team"),
                "year": self.state.get("simulation_year", 2024),
                "weather": self.state["weather"],
                "SpeedST": driver_state["SpeedST"],
                "SpeedI1": driver_state["SpeedI1"],
                "SpeedI2": driver_state["SpeedI2"],
                "laps_remaining": laps_remaining_in_race,
                "current_lap": self.current_lap,
                "characteristics": driver_state.get(
                    "characteristics", {}
                ),  # Add driver characteristics
                **driver_state,
                "strategy": {
                    "total_laps": self.config.get("race_laps", 50),
                    "available_slick_compounds": self.config.get(
                        "tyre_compounds", {}
                    ).get("slicks", ["SOFT", "MEDIUM", "HARD"]),
                    "weather_forecast": self.state["weather"],
                },
            }

            # logger.debug(
            #     f"SimLoop: Passing to TireManagerAgent for driver {driver_id}, lap {self.current_lap}: weather={self.state['weather']}"
            # )

            tire_manager_output = self.agents["tire_manager"].process(
                tire_manager_input
            )
            # Update driver_state with fresh tire info (wear, grip, pit_recommendation etc.)
            driver_state.update(tire_manager_output)

            # 2. Strategy Agent: Decide on pitting
            strategy_input = {
                "lap": self.state["lap"],
                "total_laps": self.race_laps,
                "weather": self.state["weather"],
                "track_state": self.state["track_state"],
                "circuit_id": self.circuit_name,
                "driver_id": driver_id,
                "driver_state": driver_state,
                "pit_time_penalty": self.pit_time_penalty,
                "driver_characteristics": driver_state.get("characteristics", {}),
                "track_positions": track_positions,  # Add track positions
            }
            strategy_result = self.agents["strategy"].process(strategy_input)
            pit_decision = strategy_result.get("pit_decision", False)
            new_compound_decision = strategy_result.get("new_compound")

            if pit_decision:
                logger.info(
                    f"StrategyAgent decided to PIT for {driver_id} on lap {self.current_lap}. Re-evaluating TireManager state for pit."
                )
                pit_tire_manager_input = {
                    **tire_manager_input,
                    "is_pit_lap": True,
                    "new_compound": new_compound_decision,
                }
                tire_manager_output_after_pit_decision = self.agents[
                    "tire_manager"
                ].process(pit_tire_manager_input)
                driver_state.update(tire_manager_output_after_pit_decision)
                # Ensure prev_lap_degradation_s for next lap reflects the most recent TMA output
                # This will be captured before LapTimeAgent and then formally set for next lap.

            # Capture the actual degradation calculated for THIS lap, to be stored for NEXT lap's LTA.
            actual_degradation_this_lap = driver_state[
                "estimated_degradation_s_this_lap"
            ]

            # 5. Lap Time Agent: Calculate lap time using updated state
            lap_time_input = {
                "circuit_id": self.circuit_name,
                "driver_id": driver_state["driver_id_code"],
                "driver_name": driver_state["driver_name"],
                "team_name": driver_state["team_name"],
                "compound": driver_state["current_compound"],
                "tire_age": driver_state["tire_age"],
                "weather": self.state["weather"],
                "degradation_per_lap_s": degradation_affecting_current_lap_time,  # USE CAPTURED VALUE
                "tire_wear_percentage": driver_state["tire_wear"],
                "grip_level": driver_state["grip_level"],
                "SpeedST": driver_state["SpeedST"],
                "SpeedI1": driver_state["SpeedI1"],
                "SpeedI2": driver_state["SpeedI2"],
                "SpeedFL": driver_state["SpeedFL"],
                "SpeedST_Diff": driver_state["SpeedST_Diff"],
                "SpeedI1_Diff": driver_state["SpeedI1_Diff"],
                "SpeedI2_Diff": driver_state["SpeedI2_Diff"],
                "SpeedFL_Diff": driver_state["SpeedFL_Diff"],
                "WetTrack": 1 if self.state["weather"].get("rainfall", 0) > 0 else 0,
                "TrackCondition": 0,
                "WeatherStability": 1,
                "WindSpeed_Avg": self.state["weather"].get("wind_speed", 5),
                "TempDelta": abs(
                    self.state["weather"].get("track_temp", 25)
                    - self.state["weather"].get("air_temp", 20)
                ),
            }
            lap_time_result = self.agents["lap_time"].process(lap_time_input)

            # Finalize lap time and update state
            predicted_laptime = lap_time_result.get("predicted_laptime", 90.0)
            if pit_decision:
                predicted_laptime += self.pit_time_penalty
                driver_state["pit_stops"].append(self.current_lap)

            driver_state["lap_times"].append(predicted_laptime)
            driver_state["total_race_time"] += predicted_laptime

            # Update driver_state with the degradation from THIS lap, for the NEXT lap's LapTimeAgent
            driver_state["prev_lap_degradation_s"] = actual_degradation_this_lap

            # Calculate gap effects (dirty air, DRS, fuel load)
            gap_effects_input = {
                "gap_to_ahead": driver_state["gap_to_ahead"],
                "current_lap": self.current_lap,
                "base_lap_time": predicted_laptime,  # Use initial predicted time
                "track_characteristics": {
                    "dirty_air_sensitivity": 1.0,  # Can be customized per track
                    "drs_effectiveness": 1.0,  # Can be customized per track
                    "drs_zones": 2,  # Can be customized per track
                },
                "driver_characteristics": driver_state["characteristics"],
            }
            gap_effects = self.agents["gap_effects"].process(gap_effects_input)

            # Apply gap effects to lap time
            predicted_laptime += gap_effects["lap_time_delta"]

            # Store data for history with gap effects
            lap_data_for_history["drivers"][driver_id] = {
                **driver_state,
                **lap_time_result,
                **gap_effects,  # Add gap effects data
                "lap_time": predicted_laptime,
                "pit_this_lap": pit_decision,
                "current_compound": driver_state["current_compound"],
                "tire_age": driver_state["tire_age"],
                "tire_wear": driver_state["tire_wear"],
                "grip_level": driver_state["grip_level"],
                "total_race_time": driver_state["total_race_time"],
                "position": driver_state["position"],
                "SpeedST": driver_state["SpeedST"],
                "SpeedI1": driver_state["SpeedI1"],
                "SpeedI2": driver_state["SpeedI2"],
                "SpeedFL": driver_state["SpeedFL"],
                "SpeedST_Diff": driver_state["SpeedST_Diff"],
                "SpeedI1_Diff": driver_state["SpeedI1_Diff"],
                "SpeedI2_Diff": driver_state["SpeedI2_Diff"],
                "SpeedFL_Diff": driver_state["SpeedFL_Diff"],
            }

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
