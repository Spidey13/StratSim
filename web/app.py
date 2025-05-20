"""
Streamlit web interface for the F1 Strategy Simulation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.orchestrator.simulation_loop import SimulationOrchestrator
from src.agents.lap_time_agent import LapTimeAgent
from src.agents.tire_wear_agent import TireWearAgent
from src.agents.weather_agent import WeatherAgent
from src.agents.strategy_agent import StrategyAgent

st.set_page_config(page_title="F1 Strategy Simulator", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Race Strategy Simulator")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Initial conditions
st.sidebar.subheader("Initial Conditions")
track_temp = st.sidebar.slider("Track Temperature (°C)", 20, 50, 30)
air_temp = st.sidebar.slider("Air Temperature (°C)", 15, 40, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
starting_tire = st.sidebar.selectbox("Starting Tire", ["Soft", "Medium", "Hard"])
fuel_load = st.sidebar.slider("Fuel Load (kg)", 0, 110, 100)

# Simulation settings
st.sidebar.subheader("Simulation Settings")
max_steps = st.sidebar.number_input("Maximum Laps", 1, 100, 50)
weather_update = st.sidebar.number_input("Weather Update Interval (laps)", 1, 10, 5)

if st.sidebar.button("Run Simulation"):
    # Initialize agents
    lap_time_agent = LapTimeAgent()
    tire_wear_agent = TireWearAgent()
    weather_agent = WeatherAgent()
    strategy_agent = StrategyAgent()

    # Create orchestrator
    orchestrator = SimulationOrchestrator(
        lap_time_agent=lap_time_agent,
        tire_wear_agent=tire_wear_agent,
        weather_agent=weather_agent,
        strategy_agent=strategy_agent,
        max_steps=max_steps,
        weather_update_interval=weather_update,
    )

    # Set initial conditions
    initial_conditions = {
        "track_temperature": track_temp,
        "air_temperature": air_temp,
        "humidity": humidity,
        "current_tire": starting_tire,
        "fuel_load": fuel_load,
        "lap_number": 0,
    }

    # Run simulation
    with st.spinner("Running simulation..."):
        history = orchestrator.run_simulation()

        # Convert history to DataFrame
        df = pd.DataFrame(history)

        # Plot results
        st.subheader("Simulation Results")

        # Lap times
        fig_lap_times = px.line(df, x="lap_number", y="lap_time", title="Lap Times")
        st.plotly_chart(fig_lap_times)

        # Tire wear
        fig_tire_wear = px.line(df, x="lap_number", y="tire_wear", title="Tire Wear")
        st.plotly_chart(fig_tire_wear)

        # Weather conditions
        fig_weather = px.line(
            df,
            x="lap_number",
            y=["track_temperature", "air_temperature"],
            title="Track and Air Temperature",
        )
        st.plotly_chart(fig_weather)

        # Strategy decisions
        st.subheader("Strategy Decisions")
        st.dataframe(df[["lap_number", "current_tire", "fuel_load", "pit_stop"]])
