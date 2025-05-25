"""
Script to visualize the results from the tire manager simulation.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))


def main():
    """Visualize the tire manager simulation results."""
    input_file = Path(__file__).parent / "tire_manager_simulation.csv"

    if not input_file.exists():
        print(f"Error: Could not find simulation results at {input_file}")
        print("Please run test_tire_manager_agent.py first")
        return

    # Load the simulation data
    df = pd.read_csv(input_file)

    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Plot tire wear over time
    plt.figure(figsize=(12, 8))

    # Mark pit stops with vertical lines
    pit_laps = df[df["pit_executed"]]["lap"].tolist()
    for pit_lap in pit_laps:
        plt.axvline(x=pit_lap, color="r", linestyle="--", alpha=0.7)

    # Create compound labels for the legend
    compounds = df["compound"].unique()
    for compound in compounds:
        compound_data = df[df["compound"] == compound]
        plt.plot(
            compound_data["lap"],
            compound_data["tire_wear"],
            marker="o",
            linewidth=2,
            label=f"{compound} Tire",
        )

    # Add cliff thresholds for each compound
    cliff_thresholds = {
        "SOFT": 65.0,
        "MEDIUM": 75.0,
        "HARD": 85.0,
        "INTERMEDIATE": 60.0,
        "WET": 40.0,
    }

    for compound in compounds:
        if compound in cliff_thresholds:
            plt.axhline(
                y=cliff_thresholds[compound],
                color="red",
                linestyle=":",
                alpha=0.5,
                label=f"{compound} Cliff",
            )

    plt.title("Tire Wear Progression During Race")
    plt.xlabel("Lap")
    plt.ylabel("Tire Wear (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "tire_wear_progression.png")

    # Plot grip level over time
    plt.figure(figsize=(12, 8))

    # Mark pit stops with vertical lines
    for pit_lap in pit_laps:
        plt.axvline(x=pit_lap, color="r", linestyle="--", alpha=0.7)

    # Plot grip for each compound
    for compound in compounds:
        compound_data = df[df["compound"] == compound]
        plt.plot(
            compound_data["lap"],
            compound_data["grip_level"],
            marker="o",
            linewidth=2,
            label=f"{compound} Tire",
        )

    plt.title("Grip Level Progression During Race")
    plt.xlabel("Lap")
    plt.ylabel("Grip Level (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grip_level_progression.png")

    # Create a combined plot showing wear, grip, and weather
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1: Tire Wear
    for compound in compounds:
        compound_data = df[df["compound"] == compound]
        ax1.plot(
            compound_data["lap"],
            compound_data["tire_wear"],
            marker="o",
            linewidth=2,
            label=f"{compound} Tire",
        )

    # Add cliff thresholds
    for compound in compounds:
        if compound in cliff_thresholds:
            ax1.axhline(
                y=cliff_thresholds[compound], color="red", linestyle=":", alpha=0.5
            )

    ax1.set_ylabel("Tire Wear (%)")
    ax1.set_title("Race Simulation Overview")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Grip Level
    for compound in compounds:
        compound_data = df[df["compound"] == compound]
        ax2.plot(
            compound_data["lap"],
            compound_data["grip_level"],
            marker="o",
            linewidth=2,
            label=f"{compound} Tire",
        )

    ax2.set_ylabel("Grip Level (0-1)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Weather Conditions
    ax3.plot(df["lap"], df["track_temp"], color="orange", label="Track Temp (°C)")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df["lap"], df["rainfall"], color="blue", label="Rainfall")
    ax3_twin.set_ylabel("Rainfall", color="blue")
    ax3.set_ylabel("Track Temp (°C)", color="orange")

    # Mark pit stops in all plots
    for pit_lap in pit_laps:
        ax1.axvline(x=pit_lap, color="r", linestyle="--", alpha=0.7)
        ax2.axvline(x=pit_lap, color="r", linestyle="--", alpha=0.7)
        ax3.axvline(x=pit_lap, color="r", linestyle="--", alpha=0.7)

    ax3.set_xlabel("Lap")
    ax3.grid(True, alpha=0.3)

    # Add a shared legend for the last plot
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "race_overview.png")

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
