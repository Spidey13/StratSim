"""
Script to run the F1 race strategy simulation.
"""

import sys
from pathlib import Path
import logging
import json

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.orchestrator.simulation_loop import RaceSimulator
from src.config.simulation_config import get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/simulation.log")],
)
logger = logging.getLogger(__name__)


def simulation_callback(lap: int, total_laps: int, lap_data: dict) -> None:
    """Callback function to print simulation progress."""
    logger.info(f"Completed lap {lap}/{total_laps}")

    # Print driver positions and times
    logger.info("Driver positions:")
    for driver_id, driver_data in lap_data["drivers"].items():
        logger.info(
            f"{driver_id}: P{driver_data['position']} - "
            f"Lap: {driver_data['lap_time']:.3f}s - "
            f"Total: {driver_data['total_race_time']:.3f}s - "
            f"Tires: {driver_data['current_compound']} ({driver_data['tire_age']} laps)"
        )


def main():
    """Run the simulation with default configuration."""
    try:
        # Get default configuration
        config = get_default_config()

        # Create simulation directory
        Path("logs").mkdir(exist_ok=True)

        # Initialize simulator
        simulator = RaceSimulator(config)

        # Run simulation
        logger.info("Starting simulation...")
        race_history = simulator.run_simulation(callback=simulation_callback)

        # Save results
        results_file = Path("logs/race_results.json")
        with open(results_file, "w") as f:
            json.dump(race_history, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
