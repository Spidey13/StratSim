"""
Script to test the lap time prediction model with various scenarios.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.agents.lap_time_agent import LapTimeAgent


def main():
    """Test the lap time prediction model with various scenarios."""
    print("Testing the lap time prediction model...\n")

    # Initialize the agent
    lap_time_agent = LapTimeAgent()

    # Define test scenarios
    test_scenarios = [
        {
            "name": "Ideal conditions (new soft tires, dry)",
            "inputs": {
                "circuit_id": "monza",
                "event": "Italian Grand Prix",
                "compound": "SOFT",
                "tire_age": 0,
                "weather": {
                    "track_temp": 35,
                    "air_temp": 25,
                    "humidity": 50,
                    "rainfall": 0,
                },
                "driver": "VER",
            },
        },
        {
            "name": "Old hard tires, dry",
            "inputs": {
                "circuit_id": "monza",
                "event": "Italian Grand Prix",
                "compound": "HARD",
                "tire_age": 25,
                "weather": {
                    "track_temp": 35,
                    "air_temp": 25,
                    "humidity": 50,
                    "rainfall": 0,
                },
                "driver": "VER",
            },
        },
        {
            "name": "Medium tires, light rain",
            "inputs": {
                "circuit_id": "silverstone",
                "event": "British Grand Prix",
                "compound": "MEDIUM",
                "tire_age": 10,
                "weather": {
                    "track_temp": 22,
                    "air_temp": 18,
                    "humidity": 75,
                    "rainfall": 1.5,
                },
                "driver": "HAM",
            },
        },
        {
            "name": "Wet tires, heavy rain",
            "inputs": {
                "circuit_id": "spa",
                "event": "Belgian Grand Prix",
                "compound": "WET",
                "tire_age": 5,
                "weather": {
                    "track_temp": 18,
                    "air_temp": 15,
                    "humidity": 95,
                    "rainfall": 4.0,
                },
                "driver": "HAM",
            },
        },
    ]

    # Initialize results list
    results = []

    # Process each scenario
    for i, scenario in enumerate(test_scenarios):
        print(f"Scenario {i + 1}: {scenario['name']}")
        print(f"  Inputs: {json.dumps(scenario['inputs'], indent=2)}")

        # Process inputs
        output = lap_time_agent.process(scenario["inputs"])

        # Print results
        print(f"  Predicted lap time: {output['predicted_laptime']:.3f} seconds")
        print(f"  Confidence: {output['confidence']}")
        print("  Key factors:")
        for factor in output["factors"]:
            print(
                f"    - {factor['name']}: {factor['value']} (Impact: {factor['impact']})"
            )
        print()

        # Add to results
        results.append(
            {
                "scenario": scenario["name"],
                "predicted_laptime": output["predicted_laptime"],
                "confidence": output["confidence"],
                "circuit": scenario["inputs"]["circuit_id"],
                "event": scenario["inputs"]["event"],
                "compound": scenario["inputs"]["compound"],
                "tire_age": scenario["inputs"]["tire_age"],
                "rainfall": scenario["inputs"]["weather"]["rainfall"],
            }
        )

    # Create a DataFrame with results
    results_df = pd.DataFrame(results)

    # Print summary table
    print("\nSummary of Results:")
    print(
        results_df[
            [
                "scenario",
                "circuit",
                "compound",
                "tire_age",
                "rainfall",
                "predicted_laptime",
            ]
        ]
    )

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "lap_time_predictions.csv", index=False)
    print(f"\nResults saved to {output_dir / 'lap_time_predictions.csv'}")


if __name__ == "__main__":
    main()
