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

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator.simulation_loop import RaceSimulator

# Constants
CIRCUITS = [
    "Monaco",
    "Monza",
    "Silverstone",
    "Spa",
    "Singapore",
    "Suzuka",
    "Melbourne",
    "Bahrain",
]
TIRE_COMPOUNDS = ["Soft", "Medium", "Hard", "Intermediate", "Wet"]
WEATHER_CONDITIONS = ["Dry", "Light Rain", "Heavy Rain", "Variable"]
DEFAULT_DRIVER_ID = "DRIVER1"

# Circuit information
CIRCUIT_INFO = {
    "Monaco": {
        "description": "Monte Carlo Street Circuit - Tight and twisty street circuit with minimal overtaking opportunities",
        "length": "3.337 km",
        "laps": 78,
        "icon": "🏎️",
    },
    "Monza": {
        "description": "Temple of Speed - High-speed circuit with long straights and minimal downforce requirements",
        "length": "5.793 km",
        "laps": 53,
        "icon": "🏁",
    },
    "Silverstone": {
        "description": "Home of British Motorsport - Fast and flowing circuit with high-speed corners",
        "length": "5.891 km",
        "laps": 52,
        "icon": "🇬🇧",
    },
    "Spa": {
        "description": "Circuit de Spa-Francorchamps - Iconic circuit with elevation changes and variable weather",
        "length": "7.004 km",
        "laps": 44,
        "icon": "🌧️",
    },
    "Singapore": {
        "description": "Marina Bay Street Circuit - Demanding night race with high humidity",
        "length": "5.063 km",
        "laps": 61,
        "icon": "🌃",
    },
    "Suzuka": {
        "description": "Figure-8 circuit with technical sections and high-speed corners",
        "length": "5.807 km",
        "laps": 53,
        "icon": "🎌",
    },
    "Melbourne": {
        "description": "Albert Park Circuit - Semi-permanent street circuit with technical sections",
        "length": "5.303 km",
        "laps": 58,
        "icon": "🦘",
    },
    "Bahrain": {
        "description": "Bahrain International Circuit - Modern circuit with multiple layout options",
        "length": "5.412 km",
        "laps": 57,
        "icon": "🏜️",
    },
}

# Tire compound information
TIRE_INFO = {
    "Soft": {
        "description": "Fastest compound with high degradation",
        "color": "#FF0000",
        "icon": "🔴",
    },
    "Medium": {
        "description": "Balanced performance and durability",
        "color": "#FFFF00",
        "icon": "🟡",
    },
    "Hard": {
        "description": "Most durable but slowest compound",
        "color": "#FFFFFF",
        "icon": "⚪",
    },
    "Intermediate": {
        "description": "For light rain conditions",
        "color": "#00FF00",
        "icon": "🟢",
    },
    "Wet": {
        "description": "For heavy rain conditions",
        "color": "#0000FF",
        "icon": "🔵",
    },
}

# Plotly theme configuration
PLOTLY_THEME = {
    "template": "plotly_dark",
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "white"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.1)",
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.1)",
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


def create_simulation_config(
    circuit, weather_condition, race_laps, available_compounds, starting_compound
):
    """Create simulation configuration dictionary."""
    return {
        "circuit": circuit,
        "race_laps": race_laps,
        "weather_condition": weather_condition,
        "available_compounds": available_compounds,
        "driver_info": {
            DEFAULT_DRIVER_ID: {
                "starting_compound": starting_compound,
                "starting_position": 1,
            }
        },
    }


def main():
    # Page header with custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 3em;
            color: #FF1801;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .sub-header {
            font-size: 1.5em;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 2em;
        }
        .section-header {
            font-size: 1.8em;
            color: #FF1801;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        .info-box {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1em;
            border-radius: 0.5em;
            margin-bottom: 1em;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

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
            '<div class="section-header">Simulation Parameters</div>',
            unsafe_allow_html=True,
        )

        # Circuit selection with info
        circuit = st.selectbox(
            "Select Circuit:", CIRCUITS, help="Choose the circuit for the simulation"
        )

        # Display circuit information
        if circuit in CIRCUIT_INFO:
            info = CIRCUIT_INFO[circuit]
            st.markdown(
                f"""
                <div class="info-box">
                    <h4>{info["icon"]} {circuit}</h4>
                    <p>{info["description"]}</p>
                    <p>Length: {info["length"]}</p>
                    <p>Race Laps: {info["laps"]}</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        # Weather conditions
        weather_condition = st.selectbox(
            "Weather Condition:",
            WEATHER_CONDITIONS,
            help="Select the weather conditions for the race",
        )

        # Race length
        race_laps = st.slider(
            "Race Length (laps):",
            10,
            70,
            50,
            help="Set the number of laps for the race",
        )

        # Tire compound options with info
        st.markdown(
            '<div class="section-header">Tire Strategy</div>', unsafe_allow_html=True
        )

        available_compounds = st.multiselect(
            "Available Tire Compounds:",
            TIRE_COMPOUNDS,
            default=["Soft", "Medium", "Hard"],
            help="Select which tire compounds are available for the race",
        )

        # Display selected tire compounds info
        for compound in available_compounds:
            if compound in TIRE_INFO:
                info = TIRE_INFO[compound]
                st.markdown(
                    f"""
                    <div class="info-box">
                        <h4>{info["icon"]} {compound}</h4>
                        <p>{info["description"]}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        # Starting tire
        if available_compounds:
            starting_compound = st.selectbox(
                "Starting Tire Compound:",
                available_compounds,
                help="Choose the starting tire compound",
            )

        # Run simulation button
        simulate_button = st.button(
            "Run Simulation", type="primary", use_container_width=True
        )

    # Main content area
    if simulate_button:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(
            ["Race Simulation", "Strategy Analysis", "Weather Impact"]
        )

        with tab1:
            st.markdown(
                '<div class="section-header">Race Simulation</div>',
                unsafe_allow_html=True,
            )

            # Show a progress indicator during simulation
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create simulation configuration
            config = create_simulation_config(
                circuit,
                weather_condition,
                race_laps,
                available_compounds,
                starting_compound,
            )

            # Initialize simulator
            simulator = RaceSimulator(config)

            # Define progress callback
            def update_progress(lap, total_laps, lap_data):
                progress_bar.progress(lap / total_laps)
                status_text.text(f"Simulating lap {lap}/{total_laps}")

            # Run simulation
            race_history = simulator.run_simulation(callback=update_progress)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Display results
            st.success("Simulation completed!")

            # Convert race history to DataFrame
            data = simulator.export_to_dataframe()

            # Create subplots with shared x-axis
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Lap Times During Race", "Tire Wear During Race"),
            )

            # Add lap times trace
            fig.add_trace(
                go.Scatter(
                    x=data["Lap"],
                    y=data["LapTime"],
                    mode="lines+markers",
                    name="Lap Time",
                    line=dict(color="#00ff00", width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )

            # Add tire wear trace
            fig.add_trace(
                go.Scatter(
                    x=data["Lap"],
                    y=data["TireWear"],
                    mode="lines+markers",
                    name="Tire Wear",
                    line=dict(color="#ff9900", width=2),
                    marker=dict(size=6),
                ),
                row=2,
                col=1,
            )

            # Add pit stop markers
            pit_stops = data[data["PitStop"]]["Lap"].tolist()
            for pit in pit_stops:
                # Add vertical line for pit stop
                fig.add_vline(
                    x=pit, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1
                )
                fig.add_vline(
                    x=pit, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1
                )

                # Add pit stop annotation
                fig.add_annotation(
                    x=pit,
                    y=data[data["Lap"] == pit]["LapTime"].iloc[0],
                    text="PIT",
                    showarrow=True,
                    arrowhead=1,
                    row=1,
                    col=1,
                )

            # Update layout
            fig.update_layout(height=800, showlegend=True, **PLOTLY_THEME["layout"])

            # Update y-axes labels
            fig.update_yaxes(title_text="Lap Time (seconds)", row=1, col=1)
            fig.update_yaxes(title_text="Tire Wear (%)", row=2, col=1)
            fig.update_xaxes(title_text="Lap", row=2, col=1)

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Show lap time data
            st.markdown(
                '<div class="section-header">Lap Time Data</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(data)

        with tab2:
            st.markdown(
                '<div class="section-header">Strategy Analysis</div>',
                unsafe_allow_html=True,
            )

            # Get final results
            results = simulator.get_results()
            driver_results = results["results"][DEFAULT_DRIVER_ID]

            # Create metrics in a grid
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Race Time",
                    f"{driver_results['total_time'] / 60:.2f} minutes",
                )
            with col2:
                st.metric(
                    "Average Lap Time", f"{driver_results['avg_lap_time']:.2f} seconds"
                )
            with col3:
                st.metric("Number of Pit Stops", str(driver_results["pit_stops"]))

            # Strategy analysis
            st.markdown(
                '<div class="section-header">Pit Stop Strategy</div>',
                unsafe_allow_html=True,
            )
            pit_stops = driver_results["pit_stop_details"]
            strategy_text = f"**Strategy:** {len(pit_stops)}-stop strategy\n\n"
            strategy_text += "**Pit Stop Details:**\n"
            for pit in pit_stops:
                old_compound = pit["old_compound"]
                new_compound = pit["new_compound"]
                old_info = TIRE_INFO.get(old_compound, {})
                new_info = TIRE_INFO.get(new_compound, {})
                strategy_text += f"- Lap {pit['lap']}: {old_info.get('icon', '')} {old_compound} → {new_info.get('icon', '')} {new_compound}\n"

            st.markdown(strategy_text)

        with tab3:
            st.markdown(
                '<div class="section-header">Weather Impact</div>',
                unsafe_allow_html=True,
            )

            # Weather impact
            weather_summary = results["weather_summary"]

            # Create weather metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Track Temperature", f"{weather_summary['track_temp']}°C")
            with col2:
                st.metric("Air Temperature", f"{weather_summary['air_temp']}°C")
            with col3:
                st.metric("Humidity", f"{weather_summary['humidity']}%")

            # Weather conditions
            st.markdown(
                '<div class="section-header">Weather Conditions</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-box">
                    <h4>Current Conditions</h4>
                    <p>Condition: {weather_summary["condition"]}</p>
                    <p>Rainfall: {weather_summary["rainfall"]}mm</p>
                    <p>Wind Speed: {weather_summary["wind_speed"]} km/h</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        st.info("Set the parameters and click 'Run Simulation' to start")


if __name__ == "__main__":
    main()
