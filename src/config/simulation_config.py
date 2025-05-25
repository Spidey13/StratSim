"""
Simulation configuration for F1 race strategy simulation.
"""

from typing import Dict, Any
import json
import os
from pathlib import Path

# Update model paths to match actual files
MODEL_PATHS = {
    "lap_time": "laptime_xgboost_pipeline_tuned_v2_driver_team.joblib",
    "tire_wear": "tire_wear_xgboost_pipeline_tuned_v3.joblib",
}

# Default race configuration
DEFAULT_RACE_CONFIG = {
    "circuit": "monza",
    "race_laps": 50,
    "year": 2024,
    "weather_condition": "Dry",
    "pit_time_penalty": 20.0,  # seconds
}

# Default weather configuration
DEFAULT_WEATHER_CONFIG = {
    "condition": "Dry",
    "rainfall": 0,
    "air_temp": 25,
    "track_temp": 35,
    "humidity": 40,
    "wind_speed": 5,
}


def load_driver_data() -> Dict[str, Any]:
    """
    Load driver data from JSON files.
    Returns a dictionary containing driver information and characteristics.
    """
    base_path = Path(__file__).parent.parent.parent / "data" / "categories"

    # Load drivers.json
    with open(base_path / "drivers.json", "r") as f:
        drivers = json.load(f)

    # Load driver characteristics
    with open(base_path / "driver_characteristics.json", "r") as f:
        characteristics = json.load(f)

    return {"drivers": drivers, "characteristics": characteristics}


def create_driver_config(
    driver_id: str, position: int = 1, compound: str = "MEDIUM"
) -> Dict[str, Any]:
    """
    Create a driver configuration dynamically based on driver ID.
    """
    driver_data = load_driver_data()
    driver_chars = driver_data["characteristics"].get(driver_id, {})
    driver_info = driver_data["drivers"].get(driver_id, {})

    # Get the most recent team for the driver
    teams = driver_info.get("teams", [])
    current_team = teams[-1] if teams else "Unknown Team"

    return {
        "driver_name": driver_chars.get("name", driver_id),
        "team_name": current_team,
        "starting_position": position,
        "starting_compound": compound,
        "characteristics": driver_chars.get(
            "characteristics",
            {
                "aggression": 1.0,
                "consistency": 1.0,
                "risk_tolerance": 1.0,
                "tire_management": 1.0,
                "wet_weather": 1.0,
                "qualifying_pace": 1.0,
                "race_pace": 1.0,
                "overtaking": 1.0,
            },
        ),
        # Default speed metrics that can be customized
        "SpeedST": 280.0,
        "SpeedI1": 200.0,
        "SpeedI2": 150.0,
        "SpeedFL": 260.0,
    }


def get_default_config(
    selected_drivers: Dict[str, Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get the default simulation configuration.
    Args:
        selected_drivers: Dictionary of driver_id -> {position, compound} for configuring multiple drivers
    """
    # If no drivers specified, use VER as default
    if not selected_drivers:
        selected_drivers = {"VER": {"position": 1, "compound": "MEDIUM"}}

    # Create driver configurations
    driver_configs = {}
    for driver_id, config in selected_drivers.items():
        position = config.get("position", 1)
        compound = config.get("compound", "MEDIUM")
        driver_configs[driver_id] = create_driver_config(driver_id, position, compound)

    return {
        **DEFAULT_RACE_CONFIG,
        "driver_info": driver_configs,
        "weather": DEFAULT_WEATHER_CONFIG,
    }
