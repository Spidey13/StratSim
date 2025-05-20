"""
FastAPI endpoints for the F1 Strategy Simulation.
"""

from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.orchestrator.simulation_loop import SimulationOrchestrator
from src.agents.lap_time_agent import LapTimeAgent
from src.agents.tire_wear_agent import TireWearAgent
from src.agents.weather_agent import WeatherAgent
from src.agents.strategy_agent import StrategyAgent

app = FastAPI(
    title="F1 Strategy Simulation API",
    description="API for simulating F1 race strategies using multi-agent systems",
    version="1.0.0",
)


class SimulationRequest(BaseModel):
    initial_conditions: Dict
    max_steps: int = 100
    weather_update_interval: int = 5


class SimulationResponse(BaseModel):
    history: List[Dict]
    final_state: Dict


@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a complete race strategy simulation."""
    try:
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
            max_steps=request.max_steps,
            weather_update_interval=request.weather_update_interval,
        )

        # Run simulation
        orchestrator.initialize_simulation(request.initial_conditions)
        history = orchestrator.run_simulation()

        return SimulationResponse(
            history=history, final_state=history[-1] if history else {}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
