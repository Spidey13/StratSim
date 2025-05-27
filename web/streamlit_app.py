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
# This script is in web/, so project_root is one level up.
current_script_dir = os.path.dirname(__file__)
project_root_path = os.path.abspath(os.path.join(current_script_dir, ".."))
if project_root_path not in sys.path:  # Ensure it's added only once
    sys.path.insert(0, project_root_path)

from src.orchestrator.simulation_loop import RaceSimulator

# Configure logger
logger = logging.getLogger(__name__)


# --- Data Loading and Config Functions (from the former setup page) ---
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
    base_data_path = os.path.join(project_root_path, "data", "categories")
    drivers_data = load_json_data(os.path.join(base_data_path, "drivers.json"))
    events_data = load_json_data(os.path.join(base_data_path, "events.json"))
    teams_data = load_json_data(os.path.join(base_data_path, "teams.json"))

    circuit_data = {
        "circuits": {},
        "weather_conditions": ["Dry", "Light Rain", "Heavy Rain", "Variable"],
    }
    if isinstance(events_data, dict):
        for event_name, event_info in events_data.items():
            years = event_info.get("years", []) if isinstance(event_info, dict) else []
            most_recent_year = max(years) if years else 2023
            circuit_data["circuits"][event_name] = {
                "icon": "🏎️",
                "description": f"F1 circuit for the {event_name} ({most_recent_year})",
                "length": 5.0,
                "laps": 50 if "Monaco" not in event_name else 78,
                "years": years,
            }
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
    return {
        "CIRCUIT_DATA": circuit_data,
        "TIRE_DATA": tire_data,
        "DRIVERS_DATA": drivers_data,
        "EVENTS_DATA": events_data,
        "TEAMS_DATA": teams_data,
    }


def create_simulation_config(
    circuit_name,
    weather_condition,
    race_laps,
    available_compounds,
    selected_drivers_config,
    selected_team,
    selected_driver_id,
    drivers_data=None,
    use_live_weather=False,
):
    logger.debug(f"Creating simulation config with live weather: {use_live_weather}")
    logger.debug(
        f"Circuit: {circuit_name}, Initial weather condition: {weather_condition}"
    )
    driver_chars_path = os.path.join(
        project_root_path, "data", "categories", "driver_characteristics.json"
    )
    with open(driver_chars_path, "r") as f:
        driver_chars = json.load(f)
    driver_info = {}
    for driver_id_key, config_val in selected_drivers_config.items():
        driver_characteristics = driver_chars.get(driver_id_key, {}).get(
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
        current_team = config_val.get("team_name", selected_team)
        driver_info[driver_id_key] = {
            "driver_name": driver_chars.get(driver_id_key, {}).get(
                "name", driver_id_key
            ),
            "team_name": current_team,
            "grid_position": config_val["position"],
            "starting_compound": config_val["compound"],
            "characteristics": driver_characteristics,
            "SpeedST": 280.0,
            "SpeedI1": 200.0,
            "SpeedI2": 150.0,
            "SpeedFL": 260.0,
        }
    weather_details = {
        "condition": weather_condition,
        "rainfall": 0.0 if weather_condition == "Dry" else 0.5,
        "track_temp": 30.0,
        "air_temp": 25.0,
        "humidity": 50.0,
        "wind_speed": 5.0,
        "forecast": [],
    }
    logger.debug(f"Initial weather details: {weather_details}")
    final_config = {
        "circuit": circuit_name,
        "weather_condition": weather_condition,
        "weather": weather_details,
        "race_laps": race_laps,
        "available_compounds": available_compounds,
        "tyre_compounds": {
            "slicks": ["SOFT", "MEDIUM", "HARD"],
            "wets": ["INTERMEDIATE", "WET"],
        },
        "drivers": driver_info,
        "pit_time_penalty": 20.0,
        "year": 2024,
    }
    if use_live_weather:
        logger.debug("Adding weather API key for live weather data")
        final_config["weather_api_key"] = "f931e470304ac685eca162a8a2ba25c1"
    logger.debug(f"Final simulation config (excluding driver details): {final_config}")
    return final_config


# Plotly theme configuration (remains the same)
PLOTLY_THEME = {
    "template": "plotly_dark",
    "layout": {
        "paper_bgcolor": "#1A1B1E",
        "plot_bgcolor": "#2C2D31",
        "font": {"color": "white", "family": "Lato, sans-serif"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.1)",
            "gridwidth": 0.5,
            "tickfont": {"color": "#E0E0E0"},
            "title_font": {"color": "#FFFFFF"},
            "showspikes": True,
            "spikecolor": "rgba(255,255,255,0.3)",
            "spikesnap": "cursor",
            "spikemode": "across",
            "spikethickness": 1,
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.1)",
            "gridwidth": 0.5,
            "tickfont": {"color": "#E0E0E0"},
            "title_font": {"color": "#FFFFFF"},
            "showspikes": True,
            "spikecolor": "rgba(255,255,255,0.3)",
            "spikesnap": "cursor",
            "spikemode": "across",
            "spikethickness": 1,
        },
        "legend": {
            "font": {"color": "white"},
            "bgcolor": "rgba(44,45,49,0.8)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "borderwidth": 1,
            "x": 0,
            "y": 1.1,
            "orientation": "h",
        },
        "hoverlabel": {
            "bgcolor": "#2C2D31",
            "font_color": "white",
            "bordercolor": "#FF3B30",
            "font_size": 12,
        },
        "transition": {"duration": 500, "easing": "cubic-in-out"},
    },
}

# Page configuration - Remove initial_sidebar_state
st.set_page_config(
    page_title="F1 Race Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
)

# Custom CSS for Visual Appeal (remains the same)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap');
    html, body, [class*="css"] {font-family: 'Lato', sans-serif; background-color: #1A1B1E;}
    .main-header {font-size: 3.5em; color: #FF3B30; text-align: center; margin-bottom: 0.5em; text-shadow: 2px 2px 5px rgba(0,0,0,0.5);}
    .sub-header {font-size: 1.8em; color: #E0E0E0; text-align: center; margin-bottom: 2em;}
    .section-header {font-size: 2em; color: #FF3B30; margin-top: 1.5em; margin-bottom: 0.8em; border-bottom: 2px solid rgba(255, 59, 48, 0.5); padding-bottom: 0.3em;}
    .info-box {background-color: #2C2D31; padding: 1.2em; border-radius: 0.7em; margin-bottom: 1.2em; border: 1px solid rgba(255, 255, 255, 0.1);}
    .stButton>button {background-color: #FF3B30; color: white; border-radius: 0.5rem; border: none; padding: 0.75rem 1.5rem; font-size: 1.1em; transition: background-color 0.3s ease;}
    .stButton>button:hover {background-color: #E0352C;}
    .stMetric {background-color: #2C2D31; border-radius: 0.5em; padding: 1em; margin-bottom: 1em; border: 1px solid rgba(255, 255, 255, 0.1);}
    .stMetric > div:first-child {color: #E0E0E0; font-size: 0.9em;}
    .stMetric > div:nth-child(2) {color: #FF3B30; font-size: 1.8em; font-weight: bold;}
    .stExpander {background-color: #2C2D31; border-radius: 0.5em; border: 1px solid rgba(255, 255, 255, 0.1);}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Function to render the Home View ---
def render_home_view():
    st.markdown(
        '<div class="main-header">🏎️ F1 Race Strategy Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI-Powered Race Strategy Simulation</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        ## Welcome to the F1 Race Strategy Simulator!
        This tool leverages AI to help you explore and understand the impact of various strategic
        decisions in Formula 1 racing.
        **With this simulator, you can:**
        *   Configure detailed race scenarios: select tracks, define weather conditions,
            set up driver and team parameters, and manage tire compounds.
        *   Run simulations to see how these factors play out over a race.
        *   Analyze results through interactive charts and data summaries.
        """,
        unsafe_allow_html=False,
    )
    st.markdown("---")
    if st.button(
        "🚀 Start Configuring Your Simulation", use_container_width=True, type="primary"
    ):
        st.session_state.current_view = "setup"
        st.rerun()  # Rerun to reflect view change


# --- Function to render the Setup and Results View ---
def render_setup_and_results_view():
    if st.button("🏠 Back to Home", key="back_to_home_setup"):
        st.session_state.current_view = "home"
        st.rerun()

    st.title("⚙️ Simulation Setup & Results")
    st.markdown(
        "Configure the parameters for your F1 race simulation below, then click 'Run Simulation'."
    )

    # Load data (this will be cached by Streamlit thanks to @st.cache_data)
    app_data = load_all_app_data()
    CIRCUIT_DATA = app_data["CIRCUIT_DATA"]
    TIRE_DATA = app_data["TIRE_DATA"]
    DRIVERS_DATA = app_data["DRIVERS_DATA"]
    TEAMS_DATA = app_data["TEAMS_DATA"]
    driver_chars_path = os.path.join(
        project_root_path, "data", "categories", "driver_characteristics.json"
    )
    with open(driver_chars_path, "r") as f:
        DRIVER_CHARS = json.load(f)

    CIRCUITS = list(CIRCUIT_DATA["circuits"].keys())
    WEATHER_CONDITIONS = CIRCUIT_DATA["weather_conditions"]
    TIRE_COMPOUNDS = list(TIRE_DATA["compounds"].keys())
    TIRE_INFO = TIRE_DATA["compounds"]

    with st.expander("🗓️ Event & Race Settings", expanded=True):
        st.markdown("### Event & Race Settings")
        selected_event = st.selectbox(
            "Select Event:",
            CIRCUITS,
            key="selected_event_main",
            help="Choose the F1 event for the simulation.",
        )
        selected_circuit = selected_event
        weather_mode = st.selectbox(
            "Weather Mode:",
            ["Preset Weather", "Live Weather"],
            key="weather_mode_main",
            help="Choose between preset weather conditions or real-time weather data for the selected circuit.",
        )
        if weather_mode == "Preset Weather":
            weather_condition_input = st.selectbox(
                "Weather Condition:",
                WEATHER_CONDITIONS,
                index=WEATHER_CONDITIONS.index("Dry")
                if "Dry" in WEATHER_CONDITIONS
                else 0,
                key="weather_condition_main",
                help="Select the initial weather condition for the race.",
            )
        else:
            st.info(
                "Live weather data will be fetched from the circuit's location at race start."
            )
            weather_condition_input = "Dry"
        race_laps_input = st.number_input(
            "Number of Laps:",
            min_value=1,
            max_value=200,
            value=CIRCUIT_DATA["circuits"].get(selected_circuit, {}).get("laps", 50),
            step=1,
            key="race_laps_main",
            help="Set the total number of laps for the race.",
        )

    # Tire compound logic
    options_for_starting_compound_dropdown = (
        [
            c
            for c in TIRE_COMPOUNDS
            if c
            in (
                ["SOFT", "MEDIUM", "HARD"]
                if weather_condition_input == "Dry"
                else ["INTERMEDIATE", "WET"]
            )
        ]
        if weather_condition_input == "Dry" or "Rain" in weather_condition_input
        else TIRE_COMPOUNDS[:]
    )
    compounds_for_strategy_during_sim = (
        [
            c
            for c in TIRE_COMPOUNDS
            if c
            in (
                ["SOFT", "MEDIUM", "HARD"]
                if weather_condition_input == "Dry"
                else ["INTERMEDIATE", "WET"]
            )
        ]
        if weather_condition_input == "Dry" or "Rain" in weather_condition_input
        else TIRE_COMPOUNDS[:]
    )

    # Fallbacks
    if not compounds_for_strategy_during_sim and TIRE_COMPOUNDS:
        compounds_for_strategy_during_sim = TIRE_COMPOUNDS[:]
    if not options_for_starting_compound_dropdown and TIRE_COMPOUNDS:
        options_for_starting_compound_dropdown = [TIRE_COMPOUNDS[0]]

    default_starting_compound_choice = ""
    if options_for_starting_compound_dropdown:
        if weather_condition_input == "Dry":
            default_starting_compound_choice = (
                "MEDIUM"
                if "MEDIUM" in options_for_starting_compound_dropdown
                else options_for_starting_compound_dropdown[0]
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
        else:
            default_starting_compound_choice = options_for_starting_compound_dropdown[0]
    else:
        default_starting_compound_choice = "MEDIUM"

    selected_drivers_config = {}
    with st.expander("👨‍🚀 Driver Configuration", expanded=True):
        st.markdown("### Driver Configuration")
        num_drivers = st.number_input(
            "Number of Drivers",
            min_value=1,
            max_value=20,
            value=1,
            key="num_drivers_main",
        )
        for i in range(num_drivers):
            st.markdown(f"--- \n#### **Driver {i + 1} Setup**")
            cols = st.columns([2, 2, 1, 2])
            available_teams = sorted(TEAMS_DATA.keys())
            if not available_teams:
                st.error("No teams data available.")
                continue
            selected_team_for_driver = cols[0].selectbox(
                f"Select Team:", available_teams, key=f"selected_team_main_{i}"
            )
            if selected_team_for_driver and TEAMS_DATA[selected_team_for_driver].get(
                "drivers"
            ):
                available_driver_ids_for_team = TEAMS_DATA[selected_team_for_driver][
                    "drivers"
                ]
                driver_display_names_map = {
                    driver_id: f"{DRIVER_CHARS.get(driver_id, {}).get('name', driver_id)} ({driver_id})"
                    for driver_id in available_driver_ids_for_team
                }
                sorted_driver_ids = sorted(
                    available_driver_ids_for_team,
                    key=lambda id: driver_display_names_map[id],
                )
                selected_driver_id_for_config = cols[1].selectbox(
                    f"Select Driver:",
                    options=sorted_driver_ids,
                    format_func=lambda id: driver_display_names_map[id],
                    key=f"selected_driver_main_{i}",
                )
                position = cols[2].number_input(
                    f"Grid Pos:",
                    min_value=1,
                    max_value=40,
                    value=i + 1,
                    key=f"position_main_{i}",
                )
                actual_default_compound = default_starting_compound_choice
                if (
                    actual_default_compound
                    not in options_for_starting_compound_dropdown
                ):
                    actual_default_compound = (
                        options_for_starting_compound_dropdown[0]
                        if options_for_starting_compound_dropdown
                        else "MEDIUM"
                    )
                starting_compound = cols[3].selectbox(
                    f"Start Compound:",
                    options=options_for_starting_compound_dropdown,
                    index=options_for_starting_compound_dropdown.index(
                        actual_default_compound
                    )
                    if actual_default_compound in options_for_starting_compound_dropdown
                    else 0,
                    key=f"compound_main_{i}",
                )
                selected_drivers_config[selected_driver_id_for_config] = {
                    "position": position,
                    "compound": starting_compound,
                    "team_name": selected_team_for_driver,
                }
            else:
                st.warning(
                    f"Selected team '{selected_team_for_driver}' has no drivers listed or team data is missing for Driver {i + 1}."
                )

    st.markdown("---")
    simulate_button = st.button(
        "🚀 Run Simulation",
        type="primary",
        use_container_width=True,
        key="run_sim_main",
    )

    if simulate_button:
        if not selected_drivers_config:
            st.error("Please select at least one driver before running the simulation.")
            st.stop()
        with st.spinner("Running simulation..."):
            try:
                simulation_config = create_simulation_config(
                    circuit_name=selected_circuit,
                    weather_condition=weather_condition_input,
                    race_laps=race_laps_input,
                    available_compounds=compounds_for_strategy_during_sim,
                    selected_drivers_config=selected_drivers_config,
                    selected_team=None,
                    selected_driver_id=None,
                    drivers_data=DRIVERS_DATA,
                    use_live_weather=(weather_mode == "Live Weather"),
                )
                simulator = RaceSimulator(simulation_config)
                race_history = simulator.run_simulation()
                results = simulator.get_results()

                if not race_history:
                    st.error("Simulation did not produce any race history.")
                    st.stop()

                data = simulator.export_to_dataframe()
                if data.empty:
                    st.error("The simulation produced an empty dataset.")
                    st.stop()

                if "Driver" not in data.columns:
                    st.error(
                        f"Error: 'Driver' column not found. Columns: {data.columns.tolist()}"
                    )
                    st.stop()

                st.success(
                    f"Simulation completed for {', '.join(selected_drivers_config.keys())}!"
                )

                # Display results visualization
                st.markdown("### 📊 Race Results & Analysis")

                # Create race data visualization
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        "🏁 Lap Times (seconds)",
                        "🛞 Tire Wear (%)",
                        "📉 Per-Lap Degradation (seconds)",
                        "🔧 Grip Level",
                    ),
                    specs=[
                        [{"secondary_y": False}],
                        [{"secondary_y": False}],
                        [{"secondary_y": False}],
                        [{"secondary_y": False}],
                    ],
                )

                # F1 color palette
                f1_colors = [
                    "#00A3FF",
                    "#FF9500",
                    "#34C759",
                    "#AF52DE",
                    "#FF3B30",
                    "#5856D6",
                    "#FF2D55",
                    "#FFCC00",
                    "#64D2FF",
                    "#FF6B22",
                    "#32D74B",
                    "#BF5AF2",
                    "#FF453A",
                    "#0A84FF",
                    "#FFD60A",
                    "#30D158",
                    "#64D2FF",
                    "#FF375F",
                    "#40C8E0",
                ]

                driver_colors = {
                    driver_id: f1_colors[i % len(f1_colors)]
                    for i, driver_id in enumerate(selected_drivers_config.keys())
                }

                # Plot data for each driver
                for driver_id_key in selected_drivers_config.keys():
                    driver_data_df = data[data["Driver"] == driver_id_key].copy()
                    if driver_data_df.empty:
                        st.warning(
                            f"No data for driver {DRIVER_CHARS.get(driver_id_key, {}).get('name', driver_id_key)}"
                        )
                        continue

                    driver_name_display = driver_data_df["DriverName"].iloc[0]
                    driver_color_hex = driver_colors[driver_id_key]

                    # Lap times
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data_df["Lap"],
                            y=driver_data_df["LapTime"],
                            mode="lines+markers",
                            name=f"{driver_name_display}",
                            line=dict(
                                color=driver_color_hex,
                                width=3,
                                shape="spline",
                                smoothing=0.3,
                            ),
                            marker=dict(
                                size=6,
                                color=driver_color_hex,
                                line=dict(width=2, color="white"),
                                opacity=0.8,
                            ),
                            hovertemplate=(
                                f"<b>{driver_name_display}</b><br>"
                                "Lap: %{x}<br>"
                                "Lap Time: %{y:.3f}s<br>"
                                "Compound: %{customdata[0]}<br>"
                                "Tire Age: %{customdata[1]} laps<br>"
                                "<extra></extra>"
                            ),
                            customdata=list(
                                zip(
                                    driver_data_df["Compound"],
                                    driver_data_df["TireAge"],
                                )
                            ),
                            legendgroup=driver_name_display,
                        ),
                        row=1,
                        col=1,
                    )

                    # Tire wear
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data_df["Lap"],
                            y=driver_data_df["TireWearPercentage"],
                            mode="lines+markers",
                            name=f"{driver_name_display}",
                            line=dict(
                                color=driver_color_hex,
                                width=3,
                                shape="spline",
                                smoothing=0.3,
                            ),
                            marker=dict(
                                size=6,
                                color=driver_color_hex,
                                line=dict(width=2, color="white"),
                                opacity=0.8,
                            ),
                            hovertemplate=(
                                f"<b>{driver_name_display}</b><br>"
                                "Lap: %{x}<br>"
                                "Tire Wear: %{y:.1f}%<br>"
                                "Compound: %{customdata[0]}<br>"
                                "Tire Age: %{customdata[1]} laps<br>"
                                "<extra></extra>"
                            ),
                            customdata=list(
                                zip(
                                    driver_data_df["Compound"],
                                    driver_data_df["TireAge"],
                                )
                            ),
                            legendgroup=driver_name_display,
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

                    # Degradation
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data_df["Lap"],
                            y=driver_data_df["EstimatedDegradationPerLap_s"],
                            mode="lines+markers",
                            name=f"{driver_name_display}",
                            line=dict(
                                color=driver_color_hex,
                                width=3,
                                shape="spline",
                                smoothing=0.3,
                            ),
                            marker=dict(
                                size=6,
                                color=driver_color_hex,
                                line=dict(width=2, color="white"),
                                opacity=0.8,
                            ),
                            hovertemplate=(
                                f"<b>{driver_name_display}</b><br>"
                                "Lap: %{x}<br>"
                                "Degradation: %{y:.3f}s/lap<br>"
                                "Compound: %{customdata[0]}<br>"
                                "Tire Age: %{customdata[1]} laps<br>"
                                "<extra></extra>"
                            ),
                            customdata=list(
                                zip(
                                    driver_data_df["Compound"],
                                    driver_data_df["TireAge"],
                                )
                            ),
                            legendgroup=driver_name_display,
                            showlegend=False,
                        ),
                        row=3,
                        col=1,
                    )

                    # Grip level
                    fig.add_trace(
                        go.Scatter(
                            x=driver_data_df["Lap"],
                            y=driver_data_df["GripLevel"],
                            mode="lines+markers",
                            name=f"{driver_name_display}",
                            line=dict(
                                color=driver_color_hex,
                                width=3,
                                shape="spline",
                                smoothing=0.3,
                            ),
                            marker=dict(
                                size=6,
                                color=driver_color_hex,
                                line=dict(width=2, color="white"),
                                opacity=0.8,
                            ),
                            hovertemplate=(
                                f"<b>{driver_name_display}</b><br>"
                                "Lap: %{x}<br>"
                                "Grip Level: %{y:.2f}<br>"
                                "Compound: %{customdata[0]}<br>"
                                "Tire Age: %{customdata[1]} laps<br>"
                                "<extra></extra>"
                            ),
                            customdata=list(
                                zip(
                                    driver_data_df["Compound"],
                                    driver_data_df["TireAge"],
                                )
                            ),
                            legendgroup=driver_name_display,
                            showlegend=False,
                        ),
                        row=4,
                        col=1,
                    )

                    # Mark pit stops
                    pit_stops_df = driver_data_df[driver_data_df["PitStop"] == True]
                    if not pit_stops_df.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pit_stops_df["Lap"],
                                y=pit_stops_df["LapTime"],
                                mode="markers",
                                name=f"{driver_name_display} - Pit Stops",
                                marker=dict(
                                    symbol="diamond",
                                    size=14,
                                    color=driver_color_hex,
                                    line=dict(width=3, color="white"),
                                    opacity=0.9,
                                ),
                                hovertemplate=(
                                    f"<b>{driver_name_display} - PIT STOP</b><br>"
                                    "Lap: %{x}<br>"
                                    "Lap Time: %{y:.3f}s<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=False,
                                legendgroup=driver_name_display,
                            ),
                            row=1,
                            col=1,
                        )

                # Update layout
                fig.update_layout(
                    height=1400,
                    showlegend=True,
                    template=PLOTLY_THEME["template"],
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        bgcolor="rgba(0,0,0,0.5)",
                        bordercolor="rgba(255,255,255,0.2)",
                        borderwidth=1,
                        font=dict(size=12, color="white"),
                    ),
                    margin=dict(l=60, r=60, t=100, b=60),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                # Update axes styling
                fig.update_xaxes(
                    rangeslider_visible=True,
                    row=4,
                    col=1,
                    gridcolor="rgba(255,255,255,0.1)",
                    gridwidth=1,
                    showspikes=True,
                    spikecolor="rgba(255,255,255,0.5)",
                    spikethickness=1,
                    spikedash="dot",
                )

                # Update y-axes styling for all subplots
                for i in range(1, 5):
                    fig.update_yaxes(
                        row=i,
                        col=1,
                        gridcolor="rgba(255,255,255,0.1)",
                        gridwidth=1,
                        showspikes=True,
                        spikecolor="rgba(255,255,255,0.5)",
                        spikethickness=1,
                        spikedash="dot",
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Individual Driver Strategy Analysis
                st.markdown("### 📈 Individual Driver Strategy Analysis")

                # Get final results for each driver
                for driver_id, driver_data_results in results.get(
                    "results", {}
                ).items():
                    if not driver_data_results:
                        continue

                    # Calculate metrics
                    lap_times_list = driver_data_results.get("lap_times", [])
                    num_laps_completed = len(lap_times_list)
                    total_sim_time_seconds = driver_data_results.get(
                        "total_race_time", 0
                    )
                    avg_lap_time_seconds = (
                        total_sim_time_seconds / num_laps_completed
                        if num_laps_completed > 0
                        else 0
                    )

                    # Create an expander for each driver's detailed strategy
                    with st.expander(
                        f"📊 Strategy Analysis - {driver_data_results.get('driver_name', driver_id)}",
                        expanded=False,
                    ):
                        # Create columns for key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        # Calculate position change
                        starting_position = driver_data_results.get("grid_position", 0)
                        final_position = "N/A"
                        position_change = 0

                        # Get final position from race history
                        if results.get("history"):
                            last_lap_data = results["history"][-1]
                            if driver_id in last_lap_data.get("drivers", {}):
                                final_pos_val = last_lap_data["drivers"][driver_id].get(
                                    "position"
                                )
                                if isinstance(final_pos_val, (int, float)):
                                    final_position = int(final_pos_val)
                                    if isinstance(starting_position, int):
                                        position_change = (
                                            starting_position - final_position
                                        )

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
                                f"{position_change:+d}"
                                if isinstance(final_position, int)
                                and isinstance(starting_position, int)
                                else "N/A",
                                help="Positions gained/lost from grid position",
                            )
                        with col4:
                            st.metric(
                                "Pit Stops",
                                len(driver_data_results.get("pit_stops", [])),
                                help="Number of pit stops made",
                            )

                        # Pit stop details
                        if driver_data_results.get("pit_stops"):
                            st.markdown("##### 🔧 Pit Stop Details")
                            pit_data_list = []
                            prev_compound = None

                            # Get starting compound
                            if results.get("history"):
                                first_lap_data = results["history"][0]
                                if driver_id in first_lap_data.get("drivers", {}):
                                    prev_compound = first_lap_data["drivers"][
                                        driver_id
                                    ].get("current_compound")
                            if not prev_compound:
                                prev_compound = driver_data_results.get(
                                    "starting_compound"
                                )

                            for pit_lap_num in driver_data_results.get("pit_stops", []):
                                # Find new compound after pit stop
                                new_compound_found = "N/A"
                                for lap_entry in results.get("history", []):
                                    if lap_entry.get("lap") == pit_lap_num + 1:
                                        if driver_id in lap_entry.get("drivers", {}):
                                            new_compound_found = lap_entry["drivers"][
                                                driver_id
                                            ].get("current_compound", "N/A")
                                            break

                                # Get pit stop lap data
                                pit_entry_data = {}
                                for lap_entry in results.get("history", []):
                                    if lap_entry.get("lap") == pit_lap_num:
                                        if driver_id in lap_entry.get("drivers", {}):
                                            pit_entry_data = lap_entry["drivers"][
                                                driver_id
                                            ]
                                        break

                                pit_data_list.append(
                                    {
                                        "Lap": pit_lap_num,
                                        "Position": pit_entry_data.get(
                                            "position", "N/A"
                                        ),
                                        "Gap to Leader": f"{pit_entry_data.get('gap_to_leader', 0):.1f}s",
                                        "Old Compound": f"{TIRE_INFO.get(prev_compound, {}).get('icon', '')} {prev_compound if prev_compound else 'N/A'}",
                                        "New Compound": f"{TIRE_INFO.get(new_compound_found, {}).get('icon', '')} {new_compound_found}",
                                        "Tire Age": f"{pit_entry_data.get('tire_age', 0)} laps",
                                        "Tire Wear": f"{pit_entry_data.get('tire_wear', 0):.1f}%",
                                    }
                                )
                                prev_compound = (
                                    new_compound_found
                                    if new_compound_found != "N/A"
                                    else prev_compound
                                )

                            pit_df = pd.DataFrame(pit_data_list)
                            st.dataframe(pit_df, hide_index=True)

                        # Stint analysis
                        st.markdown("##### 📈 Stint Analysis")
                        stint_data_list = []
                        current_stint_num = 1
                        stint_start_lap = 0

                        prev_compound_stint = None
                        if results.get("history"):
                            first_lap_data = results["history"][0]
                            if driver_id in first_lap_data.get("drivers", {}):
                                prev_compound_stint = first_lap_data["drivers"][
                                    driver_id
                                ].get("current_compound")
                        if not prev_compound_stint:
                            prev_compound_stint = driver_data_results.get(
                                "starting_compound"
                            )

                        for lap_idx, lap_entry in enumerate(results.get("history", [])):
                            lap_num_actual = lap_entry.get("lap", lap_idx + 1)
                            driver_lap_data = lap_entry.get("drivers", {}).get(
                                driver_id, {}
                            )

                            if driver_lap_data.get(
                                "pit_this_lap", False
                            ) or lap_num_actual == len(results.get("history", [])):
                                stint_end_lap = lap_num_actual
                                stint_lap_times = lap_times_list[
                                    stint_start_lap:stint_end_lap
                                ]

                                stint_data_list.append(
                                    {
                                        "Stint": current_stint_num,
                                        "Compound": f"{TIRE_INFO.get(prev_compound_stint, {}).get('icon', '')} {prev_compound_stint if prev_compound_stint else 'N/A'}",
                                        "Length": f"{stint_end_lap - stint_start_lap} laps",
                                        "Avg Lap Time": f"{np.mean(stint_lap_times):.2f}s"
                                        if stint_lap_times
                                        else "N/A",
                                        "Deg/Lap": f"{driver_lap_data.get('estimated_degradation_per_lap_s', 0):.3f}s",
                                    }
                                )
                                current_stint_num += 1
                                stint_start_lap = stint_end_lap
                                if driver_lap_data.get("current_compound"):
                                    prev_compound_stint = driver_lap_data[
                                        "current_compound"
                                    ]

                        if stint_data_list:
                            stint_df = pd.DataFrame(stint_data_list)
                            st.dataframe(stint_df, hide_index=True)
                        else:
                            st.info("No stint data to display.")

                        # Race summary
                        st.markdown("##### 🏁 Race Summary")
                        final_position_summary_text = (
                            final_position if isinstance(final_position, int) else "N/A"
                        )

                        best_lap_time = min(lap_times_list) if lap_times_list else 0
                        best_lap_number = (
                            lap_times_list.index(best_lap_time) + 1
                            if lap_times_list
                            else 0
                        )

                        gaps_to_leader = []
                        laps_led_count = 0
                        if results.get("history"):
                            for lap_entry in results["history"]:
                                driver_lap_detail = lap_entry.get("drivers", {}).get(
                                    driver_id, {}
                                )
                                if "gap_to_leader" in driver_lap_detail:
                                    gaps_to_leader.append(
                                        driver_lap_detail["gap_to_leader"]
                                    )
                                if driver_lap_detail.get("position") == 1:
                                    laps_led_count += 1

                        avg_gap_to_leader_str = (
                            f"{np.mean(gaps_to_leader):.1f}s"
                            if gaps_to_leader
                            else "N/A"
                        )

                        st.markdown(f"""
                        - Started P{driver_data_results.get("grid_position", "N/A")} → Finished P{final_position_summary_text}
                        - Best Lap: {best_lap_time:.3f}s (Lap {best_lap_number})
                        - Average Gap to Leader: {avg_gap_to_leader_str}
                        - Laps Led: {laps_led_count}
                        """)

                # Display detailed race data
                st.markdown("### 📋 Detailed Race Data")
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
                final_display_columns = [
                    col for col in display_columns if col in data.columns
                ]

                # Add tire compound icons
                if "Compound" in data.columns and TIRE_INFO:
                    data_display = data.copy()
                    data_display["Compound"] = data_display["Compound"].apply(
                        lambda x: f"{TIRE_INFO.get(x, {}).get('icon', '')} {x}"
                        if pd.notna(x)
                        else x
                    )
                    st.dataframe(
                        data_display[final_display_columns], use_container_width=True
                    )
                else:
                    st.dataframe(data[final_display_columns], use_container_width=True)

                # Weather summary
                st.markdown("### 🌦️ Weather Summary")
                weather_summary = results.get("weather_summary", {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Weather Condition",
                        str(weather_summary.get("condition", "N/A")),
                    )
                    st.metric(
                        "Track Temperature",
                        f"{weather_summary.get('track_temp', 'N/A')} °C",
                    )
                with col2:
                    st.metric(
                        "Air Temperature",
                        f"{weather_summary.get('air_temp', 'N/A')} °C",
                    )
                    st.metric(
                        "Rainfall", f"{weather_summary.get('rainfall', 'N/A')} mm/hr"
                    )
                with col3:
                    st.metric(
                        "Humidity",
                        f"{weather_summary.get('initial_humidity', 'N/A')} %",
                    )
                    st.metric(
                        "Wind Speed", f"{weather_summary.get('wind_speed', 'N/A')} m/s"
                    )

                st.info(
                    "The weather conditions listed below were used for the entire simulation."
                )

                condition = weather_summary.get("condition", "N/A").lower()
                track_temp = weather_summary.get("track_temp", 0.0)

                if "dry" in condition:
                    st.markdown("""
                    **Dry Conditions Impact:**
                    - **Tires:** Focus on slick compounds (Soft, Medium, Hard)
                    - **Grip:** High and predictable grip levels
                    - **Strategy:** Tire degradation management is key
                    """)
                    if track_temp > 35:
                        st.markdown(
                            "- *High track temperature likely increased tire degradation*"
                        )
                    elif track_temp < 20:
                        st.markdown(
                            "- *Low track temperature may have affected tire warm-up*"
                        )
                elif "wet" in condition or "rain" in condition:
                    st.markdown("""
                    **Wet Conditions Impact:**
                    - **Tires:** Intermediate or Wet compounds required
                    - **Grip:** Significantly reduced and variable
                    - **Strategy:** Frequent pit stops for tire changes
                    """)
                else:
                    st.markdown(
                        "**Variable Conditions:** Mixed weather requiring adaptive strategy"
                    )

            except Exception as e:
                st.error(f"An error occurred during simulation: {str(e)}")
                logger.exception("Simulation error in setup_and_results_view")
                st.stop()
    else:
        st.info(
            "Configure simulation parameters above and click 'Run Simulation' to start."
        )


# --- Main Application Logic ---
def main():
    logging.basicConfig(level=logging.DEBUG)

    # Initialize session state for view management if not already done
    if "current_view" not in st.session_state:
        st.session_state.current_view = "home"

    # Display the appropriate view based on session state
    if st.session_state.current_view == "home":
        render_home_view()
    elif st.session_state.current_view == "setup":
        render_setup_and_results_view()


if __name__ == "__main__":
    main()
