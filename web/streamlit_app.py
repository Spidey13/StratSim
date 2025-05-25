"""
Streamlit demo application for simulating race scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
import json
import logging

# --- Corrected Path Adjustment for project root ---
# Given the structure:
# project_root/
# ├── src/
# │   └── orchestrator/
# ├── data/
# └── web/
#     └── streamlit_app.py  <-- This file

# Get the directory where the current script (streamlit_app.py) is located
current_script_dir = os.path.dirname(__file__)
# Go up one level from 'web/' to reach the project root
project_root_path = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.insert(0, project_root_path)  # Add project root to sys.path

# Now, imports from src will work correctly
# Ensure this path is correct based on your actual project structure
# If simulation_loop.py is directly in src/, it would be 'from src.simulation_loop import RaceSimulator'
# If it's in src/orchestrator/, then it's 'from src.orchestrator.simulation_loop import RaceSimulator'
from src.orchestrator.simulation_loop import RaceSimulator

# Configure logger
logger = logging.getLogger(__name__)


# --- Data Loading Functions ---
@st.cache_data
def load_json_data(file_path):
    """Loads JSON data from a given file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(
            f"Warning: Data file not found at {file_path}. Please ensure your 'data' directory is correctly structured."
        )
        return {}
    except json.JSONDecodeError:
        st.error(
            f"Error: Could not decode JSON from {file_path}. Please check the file's content."
        )
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred loading {file_path}: {e}")
        return {}


@st.cache_data
def load_all_app_data():
    """Loads all necessary data for the Streamlit app from JSON files."""
    # Use the correctly determined project_root_path
    base_data_path = os.path.join(project_root_path, "data", "categories")

    drivers_data = load_json_data(os.path.join(base_data_path, "drivers.json"))
    events_data = load_json_data(os.path.join(base_data_path, "events.json"))
    teams_data = load_json_data(os.path.join(base_data_path, "teams.json"))

    # Create circuit data from events (in F1, each event corresponds to one circuit)
    circuit_data = {
        "circuits": {},
        "weather_conditions": [
            "Dry",
            "Light Rain",
            "Heavy Rain",
            "Variable",
        ],  # Explicit list
    }

    if isinstance(events_data, dict):
        for event_name, event_info in events_data.items():
            years = event_info.get("years", []) if isinstance(event_info, dict) else []
            most_recent_year = (
                max(years) if years else 2023
            )  # Default year if none found

            circuit_data["circuits"][event_name] = {
                "icon": "🏎️",
                "description": f"F1 circuit for the {event_name} ({most_recent_year})",
                "length": 5.0,  # Default length in km
                "laps": 50
                if "Monaco" not in event_name
                else 78,  # Monaco has more laps
                "years": years,
            }

    # Define tire data structure
    tire_data = {
        "compounds": {
            "SOFT": {
                "icon": "🔴",
                "color": "#FF0000",
                "description": "Soft compound - fastest but shortest life",
            },
            "MEDIUM": {
                "icon": "🟡",
                "color": "#FFFF00",
                "description": "Medium compound - balanced performance",
            },
            "HARD": {
                "icon": "⚪",
                "color": "#FFFFFF",
                "description": "Hard compound - slower but longest life",
            },
            "INTERMEDIATE": {
                "icon": "🟢",
                "color": "#00FF00",
                "description": "Intermediate - for damp conditions",
            },
            "WET": {
                "icon": "🔵",
                "color": "#0000FF",
                "description": "Wet - for rainy conditions",
            },
        }
    }

    # Return all loaded and processed data
    return {
        "CIRCUIT_DATA": circuit_data,
        "TIRE_DATA": tire_data,
        "DRIVERS_DATA": drivers_data,
        "EVENTS_DATA": events_data,
        "TEAMS_DATA": teams_data,
    }


# Plotly theme configuration
PLOTLY_THEME = {
    "template": "plotly_dark",
    "layout": {
        "paper_bgcolor": "#1A1A1A",  # Darker background for the entire plot area
        "plot_bgcolor": "#222222",  # Slightly lighter dark for the plot itself
        "font": {"color": "white", "family": "Lato, sans-serif"},  # Use Lato font
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.1)",
            "tickfont": {"color": "#CCCCCC"},
            "title_font": {"color": "#FFFFFF"},
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.1)",
            "tickfont": {"color": "#CCCCCC"},
            "title_font": {"color": "#FFFFFF"},
        },
        "legend": {
            "font": {"color": "white"},
            "bgcolor": "rgba(0,0,0,0.3)",
            "bordercolor": "rgba(255,255,255,0.1)",
        },
        "hoverlabel": {
            "bgcolor": "#333333",
            "font_color": "white",
            "bordercolor": "#FF2801",
        },
    },
}

# Page configuration
st.set_page_config(
    page_title="F1 Race Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Visual Appeal
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
    }

    .main-header {
        font-size: 3.5em;
        color: #FF2801; /* F1 Red */
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    .sub-header {
        font-size: 1.8em;
        color: #D3D3D3; /* Lighter grey for contrast */
        text-align: center;
        margin-bottom: 2em;
    }
    .section-header {
        font-size: 2em;
        color: #FF2801;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        border-bottom: 2px solid rgba(255, 40, 1, 0.5);
        padding-bottom: 0.3em;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.08);
        padding: 1.2em;
        border-radius: 0.7em;
        margin-bottom: 1.2em;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .stButton>button {
        background-color: #FF2801; /* F1 Red button */
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #E00000; /* Darker red on hover */
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5em;
        padding: 1em;
        margin-bottom: 1em;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric > div:first-child { /* Label */
        color: #D3D3D3;
        font-size: 0.9em;
    }
    .stMetric > div:nth-child(2) { /* Value */
        color: #FF2801;
        font-size: 1.8em;
        font-weight: bold;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5em;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def create_simulation_config(
    circuit_name,
    weather_condition,
    race_laps,
    available_compounds,
    selected_drivers_config,  # Dictionary of driver_id -> {position, compound}
    selected_team,  # For backward compatibility
    selected_driver_id,  # For backward compatibility
    drivers_data=None,  # Add drivers data parameter
    use_live_weather=False,  # Add parameter for live weather
):
    """Create a configuration dictionary for the race simulation."""
    logger.debug(f"Creating simulation config with live weather: {use_live_weather}")
    logger.debug(
        f"Circuit: {circuit_name}, Initial weather condition: {weather_condition}"
    )

    # Load driver characteristics
    driver_chars_path = os.path.join(
        project_root_path, "data", "categories", "driver_characteristics.json"
    )
    with open(driver_chars_path, "r") as f:
        driver_chars = json.load(f)

    # Create driver configurations
    driver_info = {}
    for driver_id, config in selected_drivers_config.items():
        # Get driver characteristics or use defaults if not found
        driver_characteristics = driver_chars.get(driver_id, {}).get(
            "characteristics", {}
        )
        if not driver_characteristics:
            driver_characteristics = {
                "aggression": 1.0,
                "consistency": 1.0,
                "risk_tolerance": 1.0,
                "tire_management": 1.0,
                "wet_weather": 1.0,
                "qualifying_pace": 1.0,
                "race_pace": 1.0,
                "overtaking": 1.0,
            }

        # Get driver's team from the config instead of looking it up
        current_team = config.get(
            "team_name", selected_team
        )  # Use the team selected in UI

        # Create driver configuration
        driver_info[driver_id] = {
            "driver_name": driver_chars.get(driver_id, {}).get("name", driver_id),
            "team_name": current_team,
            "starting_position": config["position"],
            "starting_compound": config["compound"],
            "characteristics": driver_characteristics,
            # Default speed metrics that can be customized per driver if needed
            "SpeedST": 280.0,
            "SpeedI1": 200.0,
            "SpeedI2": 150.0,
            "SpeedFL": 260.0,
        }

    # Set weather condition details
    weather_details = {
        "condition": weather_condition,
        "rainfall": 0.0 if weather_condition == "Dry" else 0.5,
        "track_temp": 30.0,  # Default track temperature
        "air_temp": 25.0,  # Default air temperature
        "humidity": 50.0,  # Default humidity
        "wind_speed": 5.0,  # Default wind speed
        "forecast": [],  # Empty forecast for now
    }
    logger.debug(f"Initial weather details: {weather_details}")

    config = {
        "circuit": circuit_name,
        "weather_condition": weather_condition,  # Add this for initialization
        "weather": weather_details,  # Add detailed weather state
        "race_laps": race_laps,
        "available_compounds": available_compounds,
        "tyre_compounds": {
            "slicks": ["SOFT", "MEDIUM", "HARD"],
            "wets": ["INTERMEDIATE", "WET"],
        },
        "driver_info": driver_info,  # Add driver info in the expected format
        "pit_time_penalty": 20.0,  # Default pit stop time penalty
        "year": 2024,  # Current simulation year
    }

    # Add weather API key if live weather is selected
    if use_live_weather:
        logger.debug("Adding weather API key for live weather data")
        config["weather_api_key"] = (
            "f931e470304ac685eca162a8a2ba25c1"  # Use key from config
        )

    logger.debug(f"Final simulation config (excluding driver details): {config}")
    return config


def main():
    # Configure logging for more detailed output
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Load all data at the start
    app_data = load_all_app_data()
    CIRCUIT_DATA = app_data["CIRCUIT_DATA"]
    TIRE_DATA = app_data["TIRE_DATA"]
    DRIVERS_DATA = app_data["DRIVERS_DATA"]
    EVENTS_DATA = app_data["EVENTS_DATA"]
    TEAMS_DATA = app_data["TEAMS_DATA"]

    # Load driver characteristics for names
    driver_chars_path = os.path.join(
        project_root_path, "data", "categories", "driver_characteristics.json"
    )
    with open(driver_chars_path, "r") as f:
        DRIVER_CHARS = json.load(f)

    # Extract data for easier use
    CIRCUITS = list(CIRCUIT_DATA["circuits"].keys())
    WEATHER_CONDITIONS = CIRCUIT_DATA["weather_conditions"]
    TIRE_COMPOUNDS = list(TIRE_DATA["compounds"].keys())
    TIRE_INFO = TIRE_DATA["compounds"]

    # Page header
    st.markdown(
        '<div class="main-header">🏎️ F1 Race Strategy Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI-Powered Race Strategy Simulation</div>',
        unsafe_allow_html=True,
    )

    # Sidebar for inputs
    with st.sidebar:
        st.markdown(
            '<div class="section-header">Simulation Setup</div>', unsafe_allow_html=True
        )

        # Event Selection
        selected_event = st.selectbox(
            "Select Event:",
            CIRCUITS,
            key="selected_event",
            help="Choose the F1 event for the simulation.",
        )

        # Use the selected event as the circuit automatically
        selected_circuit = selected_event

        # Weather mode selection
        weather_mode = st.selectbox(
            "Weather Mode:",
            ["Preset Weather", "Live Weather"],
            key="weather_mode",
            help="Choose between preset weather conditions or real-time weather data for the selected circuit.",
        )
        logger.debug(f"Selected weather mode: {weather_mode}")

        if weather_mode == "Preset Weather":
            # Show the existing weather condition selector
            weather_condition_input = st.selectbox(
                "Weather Condition:",
                WEATHER_CONDITIONS,
                index=WEATHER_CONDITIONS.index("Dry")
                if "Dry" in WEATHER_CONDITIONS
                else 0,
                key="weather_condition",
                help="Select the initial weather condition for the race.",
            )
            logger.debug(
                f"Selected preset weather condition: {weather_condition_input}"
            )
        else:
            # For live weather, show info message
            st.info(
                "Live weather data will be fetched from the circuit's location at race start."
            )
            weather_condition_input = "Dry"  # Default fallback if API fails
            logger.debug(
                "Live weather mode selected, using 'Dry' as fallback condition"
            )

        # Race length
        race_laps_input = st.number_input(
            "Number of Laps:",
            min_value=1,
            max_value=200,
            value=CIRCUIT_DATA["circuits"].get(selected_circuit, {}).get("laps", 50),
            step=1,
            key="race_laps",
            help="Set the total number of laps for the race.",
        )

        # --- Determine options for Starting Compound Dropdown (based on weather) ---
        options_for_starting_compound_dropdown = []
        if weather_condition_input == "Dry":
            options_for_starting_compound_dropdown = [
                c for c in TIRE_COMPOUNDS if c in ["SOFT", "MEDIUM", "HARD"]
            ]
        elif "Rain" in weather_condition_input:  # Covers "Light Rain", "Heavy Rain"
            options_for_starting_compound_dropdown = [
                c for c in TIRE_COMPOUNDS if c in ["INTERMEDIATE", "WET"]
            ]
        else:  # Variable weather or other unhandled conditions
            options_for_starting_compound_dropdown = TIRE_COMPOUNDS[
                :
            ]  # Offer all for variable

        # --- Determine Compounds Available for Strategy During the Simulation (based on weather) ---
        compounds_for_strategy_during_sim = []
        if weather_condition_input == "Dry":
            compounds_for_strategy_during_sim = [
                c for c in TIRE_COMPOUNDS if c in ["SOFT", "MEDIUM", "HARD"]
            ]
        elif "Rain" in weather_condition_input:  # Covers "Light Rain", "Heavy Rain"
            compounds_for_strategy_during_sim = [
                c for c in TIRE_COMPOUNDS if c in ["INTERMEDIATE", "WET"]
            ]
        else:  # Variable weather or other unhandled conditions
            # For variable conditions, all compounds might be needed
            compounds_for_strategy_during_sim = TIRE_COMPOUNDS[:]

        # Fallback if no compounds are available
        if not compounds_for_strategy_during_sim and TIRE_COMPOUNDS:
            compounds_for_strategy_during_sim = TIRE_COMPOUNDS[
                :
            ]  # Use all available compounds
            st.warning(
                f"Could not determine strategically available compounds for {weather_condition_input}; "
                f"defaulting to all known compounds: {TIRE_COMPOUNDS}"
            )
        elif not TIRE_COMPOUNDS:  # If TIRE_COMPOUNDS itself is empty
            compounds_for_strategy_during_sim = ["MEDIUM"]  # Ultimate fallback
            st.error(
                "TIRE_COMPOUNDS list is empty. Cannot determine strategically available compounds."
            )

        # Fallback if the above logic results in an empty list but TIRE_COMPOUNDS is not empty
        if not options_for_starting_compound_dropdown and TIRE_COMPOUNDS:
            options_for_starting_compound_dropdown = [
                TIRE_COMPOUNDS[0]
            ]  # Default to first available
            st.warning(
                f"Could not determine specific starting compounds for {weather_condition_input}; "
                f"dropdown defaulted to: {options_for_starting_compound_dropdown}"
            )
        elif not TIRE_COMPOUNDS:  # If TIRE_COMPOUNDS itself is empty
            options_for_starting_compound_dropdown = ["MEDIUM"]  # Ultimate fallback
            st.error(
                "TIRE_COMPOUNDS list is empty. Cannot populate starting compound dropdown."
            )

        # Determine a sensible default choice based on weather and available options
        default_starting_compound_choice = ""
        if options_for_starting_compound_dropdown:
            if weather_condition_input == "Dry":
                if "MEDIUM" in options_for_starting_compound_dropdown:
                    default_starting_compound_choice = "MEDIUM"
                else:  # If MEDIUM is not an option
                    default_starting_compound_choice = (
                        options_for_starting_compound_dropdown[0]
                    )
            elif "Rain" in weather_condition_input:
                if "INTERMEDIATE" in options_for_starting_compound_dropdown:
                    default_starting_compound_choice = "INTERMEDIATE"
                elif "WET" in options_for_starting_compound_dropdown:
                    default_starting_compound_choice = "WET"
                else:
                    default_starting_compound_choice = (
                        options_for_starting_compound_dropdown[0]
                    )
            else:  # Variable weather or other
                default_starting_compound_choice = (
                    options_for_starting_compound_dropdown[0]
                )
        else:  # Absolute fallback
            default_starting_compound_choice = "MEDIUM"

        # Add multi-driver selection
        st.sidebar.subheader("Driver Selection")

        # Allow selecting multiple drivers
        num_drivers = st.sidebar.number_input(
            "Number of Drivers", min_value=1, max_value=20, value=1
        )

        selected_drivers_config = {}

        for i in range(num_drivers):
            st.sidebar.markdown(f"**Driver {i + 1}**")

            # Team Selection
            available_teams = sorted(TEAMS_DATA.keys())
            selected_team = st.sidebar.selectbox(
                f"Select Team for Driver {i + 1}:",
                available_teams,
                key=f"selected_team_{i}",
            )

            # Driver Selection based on team
            if selected_team:
                available_drivers = TEAMS_DATA[selected_team]["drivers"]
                driver_names = {}
                for driver_id in available_drivers:
                    if driver_id in DRIVER_CHARS:
                        driver_names[driver_id] = (
                            f"{DRIVER_CHARS[driver_id]['name']} ({driver_id})"
                        )
                    else:
                        driver_names[driver_id] = driver_id

                driver_options = sorted(
                    [(id, name) for id, name in driver_names.items()],
                    key=lambda x: x[1],
                )

                selected_driver_id = st.sidebar.selectbox(
                    f"Select Driver {i + 1}:",
                    [id for id, _ in driver_options],
                    format_func=lambda x: driver_names[x],
                    key=f"selected_driver_{i}",
                )

                # Starting position
                position = st.sidebar.number_input(
                    f"Starting Position for {driver_names[selected_driver_id]}:",
                    min_value=1,
                    max_value=20,
                    value=i + 1,
                    key=f"position_{i}",
                )

                # Starting compound
                starting_compound = st.sidebar.selectbox(
                    f"Starting Compound for {driver_names[selected_driver_id]}:",
                    options_for_starting_compound_dropdown,
                    index=options_for_starting_compound_dropdown.index(
                        default_starting_compound_choice
                    ),
                    key=f"compound_{i}",
                )

                # Add to configuration
                selected_drivers_config[selected_driver_id] = {
                    "position": position,
                    "compound": starting_compound,
                    "team_name": selected_team,  # Store the selected team
                }

            st.sidebar.markdown("---")  # Visual separator between drivers

        # Run simulation button
        simulate_button = st.button(
            "Run Simulation", type="primary", use_container_width=True
        )

    # --- Main Content Area ---
    if simulate_button:
        if not selected_drivers_config:
            st.error("Please select at least one driver before running the simulation.")
            st.stop()

        with st.spinner("Running simulation..."):
            try:
                # Create simulation configuration
                logger.debug(
                    f"Creating simulation config for circuit: {selected_circuit}"
                )
                logger.debug(
                    f"Weather mode: {weather_mode}, Initial condition: {weather_condition_input}"
                )

                simulation_config = create_simulation_config(
                    circuit_name=selected_circuit,
                    weather_condition=weather_condition_input,
                    race_laps=race_laps_input,
                    available_compounds=compounds_for_strategy_during_sim,
                    selected_drivers_config=selected_drivers_config,
                    selected_team=None,  # Not needed with new multi-driver setup
                    selected_driver_id=None,  # Not needed with new multi-driver setup
                    drivers_data=DRIVERS_DATA,  # Pass drivers data
                    use_live_weather=(
                        weather_mode == "Live Weather"
                    ),  # Pass live weather flag
                )

                # Log the configuration for debugging
                logger.debug("Simulation configuration created successfully")
                logger.debug(
                    f"Weather configuration: {simulation_config.get('weather', {})}"
                )
                logger.debug(f"Using live weather: {weather_mode == 'Live Weather'}")
                if weather_mode == "Live Weather":
                    logger.debug(
                        f"API key present: {'weather_api_key' in simulation_config}"
                    )

                simulator = RaceSimulator(simulation_config)
                logger.debug("Starting simulation run")
                race_history = simulator.run_simulation()
                results = simulator.get_results()  # Store simulation results

                # Log weather results
                logger.debug("Simulation completed, checking weather data")
                if results and "weather_summary" in results:
                    logger.debug(
                        f"Final weather conditions: {results['weather_summary']}"
                    )

                if not race_history:
                    st.error(
                        "Simulation did not produce any race history. Please check the configuration and try again."
                    )
                    logger.error("No race history produced by simulation")
                    st.stop()

                data = simulator.export_to_dataframe()
                if data.empty:
                    st.error(
                        "The simulation produced an empty dataset. Please check the configuration and try again."
                    )
                    logger.error("Empty DataFrame produced by simulation")
                    st.stop()

                if "Driver" not in data.columns:
                    st.error(
                        f"Error: The 'Driver' column was not found in the simulation results. Available columns: {data.columns.tolist()}"
                    )
                    logger.error(
                        f"Missing Driver column. Available columns: {data.columns.tolist()}"
                    )
                    st.stop()

                # Create plots for all drivers in single graphs
                st.markdown("### Race Data Comparison")

                # Create subplots for all metrics
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.07,
                    subplot_titles=(
                        "Lap Times",
                        "Tire Wear",
                        "Per-Lap Degradation",
                        "Grip Level",
                    ),
                )

                # Define a color palette for different drivers
                f1_colors = [
                    "#FF2801",  # F1 Red
                    "#00FF00",  # Green
                    "#0600EF",  # Blue
                    "#FFD700",  # Gold
                    "#FF69B4",  # Pink
                    "#7F00FF",  # Purple
                    "#00FFFF",  # Cyan
                    "#FF8C00",  # Orange
                    "#4B0082",  # Indigo
                    "#00FF7F",  # Spring Green
                    "#1E90FF",  # Dodger Blue
                    "#FF1493",  # Deep Pink
                    "#FFD700",  # Gold
                    "#32CD32",  # Lime Green
                    "#8A2BE2",  # Blue Violet
                    "#FF4500",  # Orange Red
                    "#00CED1",  # Dark Turquoise
                    "#FF69B4",  # Hot Pink
                    "#32CD32",  # Lime Green
                ]

                # Create driver colors dictionary dynamically
                driver_colors = {
                    driver_id: f1_colors[i % len(f1_colors)]
                    for i, driver_id in enumerate(selected_drivers_config.keys())
                }

                # Plot data for each driver
                for driver_id in selected_drivers_config.keys():
                    driver_data = data[data["Driver"] == driver_id].copy()
                    if driver_data.empty:
                        st.warning(f"No data found for driver {driver_id}")
                        continue

                    driver_name = driver_data["DriverName"].iloc[0]
                    driver_color = driver_colors[driver_id]

                    # Lap Times
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data["Lap"],
                            y=driver_data["LapTime"],
                            mode="lines+markers",
                            name=f"{driver_name} - Lap Time",
                            line=dict(color=driver_color, width=2),
                            marker=dict(size=6),
                            customdata=driver_data["Compound"],
                            hovertemplate=f"<b>Driver:</b> {driver_name}<br><b>Lap:</b> %{{x}}<br><b>Lap Time:</b> %{{y:.2f}}s<br><b>Compound:</b> %{{customdata}}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )

                    # Tire Wear
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data["Lap"],
                            y=driver_data["TireWearPercentage"],
                            mode="lines+markers",
                            name=f"{driver_name} - Tire Wear",
                            line=dict(color=driver_color, width=2),
                            marker=dict(size=6),
                            customdata=driver_data["Compound"],
                            hovertemplate=f"<b>Driver:</b> {driver_name}<br><b>Lap:</b> %{{x}}<br><b>Tire Wear:</b> %{{y:.1f}}%<br><b>Compound:</b> %{{customdata}}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                    # Per-Lap Degradation
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data["Lap"],
                            y=driver_data["EstimatedDegradationPerLap_s"],
                            mode="lines+markers",
                            name=f"{driver_name} - Degradation",
                            line=dict(color=driver_color, width=2),
                            marker=dict(size=6),
                            customdata=driver_data["Compound"],
                            hovertemplate=f"<b>Driver:</b> {driver_name}<br><b>Lap:</b> %{{x}}<br><b>Degradation:</b> %{{y:.3f}}s<br><b>Compound:</b> %{{customdata}}<extra></extra>",
                        ),
                        row=3,
                        col=1,
                    )

                    # Grip Level
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data["Lap"],
                            y=driver_data["GripLevel"],
                            mode="lines+markers",
                            name=f"{driver_name} - Grip",
                            line=dict(color=driver_color, width=2),
                            marker=dict(size=6),
                            customdata=driver_data["Compound"],
                            hovertemplate=f"<b>Driver:</b> {driver_name}<br><b>Lap:</b> %{{x}}<br><b>Grip:</b> %{{y:.3f}}<br><b>Compound:</b> %{{customdata}}<extra></extra>",
                        ),
                        row=4,
                        col=1,
                    )

                # Update layout
                fig.update_layout(
                    height=1200,
                    showlegend=True,
                    template="plotly_dark",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                # Update y-axes labels
                fig.update_yaxes(title_text="Lap Time (s)", row=1, col=1)
                fig.update_yaxes(title_text="Tire Wear (%)", row=2, col=1)
                fig.update_yaxes(title_text="Degradation (s)", row=3, col=1)
                fig.update_yaxes(title_text="Grip Level", row=4, col=1)

                # Update x-axis label
                fig.update_xaxes(title_text="Lap Number", row=4, col=1)

                # Display the combined plot
                st.plotly_chart(fig, use_container_width=True)

                # Display data tables for all drivers
                st.markdown("### Detailed Race Data")

                # Create a combined dataframe with all drivers' data
                display_columns = [
                    "Lap",
                    "DriverName",
                    "TeamName",
                    "LapTime",
                    "Compound",
                    "TireAge",
                    "TireWearPercentage",
                    "EstimatedDegradationPerLap_s",
                    "GripLevel",
                    "Position",
                    "PitStop",
                    "TotalTime",
                    "WeatherCondition",
                    "Rainfall",
                    "TrackTemp",
                    "AirTemp",
                ]

                # Adjust position values in the dataframe to be 0-based
                if "Position" in data.columns:
                    data["Position"] = data["Position"].apply(
                        lambda x: x - 1 if pd.notna(x) else x
                    )

                # Add compound icons to the data
                if "Compound" in data.columns and TIRE_INFO:
                    data["Compound"] = data["Compound"].apply(
                        lambda x: f"{TIRE_INFO.get(x, {}).get('icon', '')} {x}"
                        if pd.notna(x)
                        else x
                    )

                # Display the combined dataframe
                st.dataframe(data[display_columns])

            except Exception as e:
                st.error(f"An error occurred during simulation: {str(e)}")
                logger.exception("Simulation error")
                st.stop()

            st.success(
                f"Simulation completed for {', '.join(selected_drivers_config.keys())}!"
            )

            # --- Define Tabs for additional analysis ---
            tab1, tab2 = st.tabs(
                [
                    "📈 Strategy Analysis",
                    "🌦️ Weather Impact",
                ]
            )

            with tab1:
                st.markdown(
                    '<div class="section-header">Strategy Analysis</div>',
                    unsafe_allow_html=True,
                )

                # Get final results for each driver
                for driver_id, driver_data in results.get("results", {}).items():
                    if not driver_data:
                        continue

                    # Calculate metrics
                    lap_times_list = driver_data.get("lap_times", [])
                    num_laps_completed = len(lap_times_list)
                    total_sim_time_seconds = driver_data.get("total_race_time", 0)
                    avg_lap_time_seconds = (
                        total_sim_time_seconds / num_laps_completed
                        if num_laps_completed > 0
                        else 0
                    )

                    # Create an expander for each driver's detailed strategy
                    with st.expander(
                        f"📊 Strategy Analysis - {driver_data.get('driver_name', driver_id)}",
                        expanded=True,
                    ):
                        # Create three columns for key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        # Calculate position change by comparing first and last lap positions
                        starting_position = driver_data.get("grid_position", 0)
                        final_position = "N/A"
                        position_change = 0
                        if results["history"]:
                            last_lap = results["history"][-1]
                            if driver_id in last_lap["drivers"]:
                                final_position = last_lap["drivers"][driver_id].get(
                                    "position", 0
                                )
                                position_change = starting_position - final_position + 1

                        with col1:
                            st.metric(
                                "Total Race Time",
                                f"{total_sim_time_seconds / 60:.2f} min",
                                help="Total race duration including pit stops",
                            )
                        with col2:
                            st.metric(
                                "Average Lap Time",
                                f"{avg_lap_time_seconds:.2f} s",
                                help="Mean lap time across the race",
                            )
                        with col3:
                            st.metric(
                                "Position Change",
                                f"{position_change:+d}",
                                help="Positions gained/lost from grid position",
                            )
                        with col4:
                            st.metric(
                                "Pit Stops",
                                len(driver_data.get("pit_stops", [])),
                                help="Number of pit stops made",
                            )

                        # Create a pit stop summary table
                        if driver_data.get("pit_stops"):
                            st.markdown("##### 🔧 Pit Stop Details")
                            pit_data = []
                            # Get the starting compound from the first lap of race history
                            prev_compound = None
                            if results["history"]:
                                first_lap = results["history"][0]
                                if driver_id in first_lap["drivers"]:
                                    prev_compound = first_lap["drivers"][driver_id].get(
                                        "current_compound"
                                    )
                            if (
                                not prev_compound
                            ):  # Fallback to starting_compound if history is empty
                                prev_compound = driver_data.get("starting_compound")

                            for pit_lap in driver_data.get("pit_stops", []):
                                # Find the compound after this pit stop
                                for lap_entry in results["history"]:
                                    if lap_entry["lap"] == pit_lap + 1:
                                        new_compound = lap_entry["drivers"][driver_id][
                                            "current_compound"
                                        ]
                                        break

                                # Get position and gap at pit entry
                                pit_entry_data = next(
                                    (
                                        lap_entry["drivers"][driver_id]
                                        for lap_entry in results["history"]
                                        if lap_entry["lap"] == pit_lap
                                    ),
                                    {},
                                )

                                pit_data.append(
                                    {
                                        "Lap": pit_lap,
                                        "Position": pit_entry_data.get(
                                            "position", "N/A"
                                        ),
                                        "Gap to Leader": f"{pit_entry_data.get('gap_to_leader', 0):.1f}s",
                                        "Old Compound": f"{TIRE_INFO.get(prev_compound, {}).get('icon', '')} {prev_compound}",
                                        "New Compound": f"{TIRE_INFO.get(new_compound, {}).get('icon', '')} {new_compound}",
                                        "Tire Age": f"{pit_entry_data.get('tire_age', 0)} laps",
                                        "Tire Wear": f"{pit_entry_data.get('tire_wear_percentage', 0):.1f}%",
                                    }
                                )
                                prev_compound = new_compound

                            pit_df = pd.DataFrame(pit_data)
                            st.dataframe(pit_df, hide_index=True)

                        # Show stint analysis
                        st.markdown("##### 📈 Stint Analysis")
                        stint_data = []
                        current_stint = 1
                        stint_start = 0

                        # Get the starting compound from the first lap of race history
                        prev_compound = None
                        if results["history"]:
                            first_lap = results["history"][0]
                            if driver_id in first_lap["drivers"]:
                                prev_compound = first_lap["drivers"][driver_id].get(
                                    "current_compound"
                                )
                        if (
                            not prev_compound
                        ):  # Fallback to starting_compound if history is empty
                            prev_compound = driver_data.get("starting_compound")

                        for lap_num, lap_entry in enumerate(results["history"], 1):
                            driver_lap = lap_entry["drivers"].get(driver_id, {})
                            if driver_lap.get("pit_this_lap", False) or lap_num == len(
                                results["history"]
                            ):
                                # Stint completed
                                stint_data.append(
                                    {
                                        "Stint": current_stint,
                                        "Compound": f"{TIRE_INFO.get(prev_compound, {}).get('icon', '')} {prev_compound}",
                                        "Length": f"{lap_num - stint_start} laps",
                                        "Avg Lap Time": f"{np.mean(lap_times_list[stint_start:lap_num]):.2f}s",
                                        "Deg/Lap": f"{driver_lap.get('estimated_degradation_per_lap_s', 0):.3f}s",
                                    }
                                )
                                current_stint += 1
                                stint_start = lap_num
                                if driver_lap.get("current_compound"):
                                    prev_compound = driver_lap["current_compound"]

                        stint_df = pd.DataFrame(stint_data)
                        st.dataframe(stint_df, hide_index=True)

                        # Add race summary
                        st.markdown("##### 🏁 Race Summary")

                        # Get final position from the last lap of race history
                        final_position = "N/A"
                        if results["history"]:
                            last_lap = results["history"][-1]
                            if driver_id in last_lap["drivers"]:
                                final_position = last_lap["drivers"][driver_id].get(
                                    "position", "N/A"
                                )

                        st.markdown(f"""
                        - Started P{driver_data.get("grid_position", "N/A")} → Finished P{final_position - 1}
                        - Best Lap: {min(lap_times_list):.3f}s (Lap {lap_times_list.index(min(lap_times_list)) + 1})
                        - Average Gap to Leader: {np.mean([lap["gap_to_leader"] for lap in (lap_entry["drivers"].get(driver_id, {}) for lap_entry in results["history"]) if "gap_to_leader" in lap]):.1f}s
                        - Laps Led: {sum(1 for lap in results["history"] if lap["drivers"].get(driver_id, {}).get("position", 0) == 1)}
                        """)

            with tab2:
                st.markdown(
                    '<div class="section-header">Weather Impact on Simulation</div>',
                    unsafe_allow_html=True,
                )

                # Weather impact
                weather_summary = results.get("weather_summary", {})

                # Create weather metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Initial Track Temp",
                        f"{weather_summary.get('initial_track_temp', 'N/A')}°C",
                    )
                with col2:
                    st.metric(
                        "Initial Air Temp",
                        f"{weather_summary.get('initial_air_temp', 'N/A')}°C",
                    )
                with col3:
                    st.metric(
                        "Initial Humidity",
                        f"{weather_summary.get('initial_humidity', 'N/A')}%",
                    )

                st.markdown(
                    '<div class="section-header">Final Weather Conditions</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="info-box">
                        <h4>Final Lap Conditions</h4>
                        <p>Condition: {weather_summary.get("condition", "N/A")}</p>
                        <p>Rainfall: {weather_summary.get("rainfall", "N/A")}mm</p>
                        <p>Wind Speed: {weather_summary.get("wind_speed", "N/A")} km/h</p>
                        <p>Track Temp: {weather_summary.get("track_temp", "N/A")}°C</p>
                        <p>Air Temp: {weather_summary.get("air_temp", "N/A")}°C</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Display weather forecast using expander
                forecast = weather_summary.get("forecast", [])
                with st.expander("View Full Weather Forecast"):
                    if forecast:
                        forecast_df = pd.DataFrame(forecast)
                        st.dataframe(forecast_df)
                    else:
                        st.info(
                            "No detailed weather forecast available for this simulation."
                        )

    else:
        st.info(
            "Configure simulation parameters in the sidebar and click 'Run Simulation' to start."
        )


if __name__ == "__main__":
    main()
