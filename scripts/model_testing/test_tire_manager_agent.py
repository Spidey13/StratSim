"""
Script to test the TireManagerAgent with various scenarios.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.agents.tire_manager_agent import TireManagerAgent


def main():
    """Test the TireManagerAgent with various scenarios."""
    print("Testing the TireManagerAgent...\n")

    # Initialize the agent
    tire_manager = TireManagerAgent()

    # Simulate a race
    race_laps = 50
    strategy = {
        "total_laps": race_laps,
        "mandatory_compounds": ["MEDIUM", "HARD"],
        "weather_forecast": {
            "current": {"rainfall": 0, "track_temp": 35},
            "forecast": [
                {"lap": i, "rainfall": 0 if i < 30 else 0.5}
                for i in range(1, race_laps + 1)
            ],
        },
    }

    # Initialize race results tracking
    results = []

    # Initial compound
    compound = "MEDIUM"
    pending_pit_stop = False
    new_compound = None

    # Simulate the race lap by lap
    for lap in range(1, race_laps + 1):
        print(f"\nLap {lap}/{race_laps}")

        # Get laps remaining
        laps_remaining = race_laps - lap

        # Weather data for this lap
        weather = {
            "track_temp": 35 if lap < 30 else 25,  # Temperature drops later in race
            "air_temp": 25 if lap < 30 else 20,
            "humidity": 50,
            "rainfall": 0 if lap < 30 else 0.5,  # Light rain later in race
        }

        # Check if we should execute a pending pit stop
        is_pit_lap = False
        if pending_pit_stop:
            is_pit_lap = True
            compound = new_compound
            pending_pit_stop = False
            print(f"Executing pit stop for {compound} tires")

        # Process the lap with the agent
        result = tire_manager.process(
            {
                "current_lap": lap,
                "race_lap": lap,
                "circuit_id": "monza",
                "driver_id": "VER",
                "weather": weather,
                "compound": compound if lap == 1 or is_pit_lap else None,
                "is_pit_lap": is_pit_lap,
                "laps_remaining": laps_remaining,
                "strategy": strategy,
            }
        )

        # Print current tire state
        print(
            f"Compound: {result['compound']}, Age: {result['tire_age']}, Wear: {result['tire_wear']:.1f}%"
        )
        print(f"Tire Health: {result['tire_health']}, Grip: {result['grip_level']:.2f}")

        # Check if pit recommendation is provided and we're not already planning to pit
        if result["pit_recommendation"]["should_pit"] and not pending_pit_stop:
            print(f"Pit Recommendation: {result['pit_recommendation']['reason']}")
            print(
                f"Recommended Compound: {result['pit_recommendation']['recommended_compound']}"
            )
            print(f"Urgency: {result['pit_recommendation']['urgency']}")
            print(
                f"Laps until cliff: {result['pit_recommendation']['laps_until_cliff']}"
            )

            # If recommendation is high urgency or we're in the pit window, schedule a pit stop
            if (
                result["pit_recommendation"]["urgency"] == "high"
                or (lap >= 25 and lap < 30)  # Force pit stop during specific window
            ):
                pending_pit_stop = True
                new_compound = result["pit_recommendation"]["recommended_compound"]
                print(f"--> Will pit next lap for {new_compound}")

        # Track results
        results.append(
            {
                "lap": lap,
                "compound": result["compound"],
                "tire_age": result["tire_age"],
                "tire_wear": result["tire_wear"],
                "tire_health": result["tire_health"],
                "grip_level": result["grip_level"],
                "wear_rate": result["wear_rate"],
                "rainfall": weather["rainfall"],
                "track_temp": weather["track_temp"],
                "should_pit": result["pit_recommendation"]["should_pit"],
                "pit_urgency": result["pit_recommendation"]["urgency"],
                "laps_until_cliff": result["pit_recommendation"]["laps_until_cliff"],
                "pit_executed": is_pit_lap,
            }
        )

    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Print summary of stint information
    print("\nStint Information:")
    for i, (compound, age) in enumerate(tire_manager.state["tire_history"], 1):
        print(f"Stint {i}: {compound} - {age} laps")

    if len(tire_manager.state["tire_history"]) > 0:
        current_stint = len(tire_manager.state["tire_history"]) + 1
    else:
        current_stint = 1

    print(
        f"Current Stint {current_stint}: {tire_manager.state['current_compound']} - {tire_manager.state['tire_age']} laps"
    )

    # Print compounds used
    print(f"\nCompounds Used: {tire_manager.state['compounds_used']}")

    # Print pit stop summary
    pit_laps = results_df[results_df["pit_executed"]]["lap"].tolist()
    print(f"\nPit Stops at laps: {pit_laps}")

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "tire_manager_simulation.csv", index=False)
    print(f"\nResults saved to {output_dir / 'tire_manager_simulation.csv'}")


if __name__ == "__main__":
    main()
