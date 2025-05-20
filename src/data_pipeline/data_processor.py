"""
F1 data processor module.
This module handles processing of raw F1 data from the collector,
performing feature engineering, cleaning, and preparing for modeling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataProcessor:
    """
    Class for processing F1 data collected via FastF1.
    """

    def __init__(self):
        """Initialize the data processor."""
        pass

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw F1 data by cleaning and feature engineering.

        Args:
            raw_data: Raw data from FastF1

        Returns:
            Processed data with additional features
        """
        if raw_data.empty:
            logger.warning("Empty DataFrame provided. No processing performed.")
            return raw_data

        logger.info("Processing data with shape: %s", str(raw_data.shape))

        # Make a copy to avoid modifying the input
        data = raw_data.copy()

        # Basic cleaning
        data = self._basic_cleaning(data)

        # Extract and convert time-based features
        data = self._process_time_features(data)

        # Create compound-specific features
        data = self._process_compound_features(data)

        # Create track position features
        data = self._process_position_features(data)

        # Create stint-based features
        data = self._process_stint_features(data)

        # Calculate delta times
        data = self._calculate_deltas(data)

        # Create race-specific features
        data = self._process_race_features(data)

        logger.info("Data processing complete. Final shape: %s", str(data.shape))
        return data

    def _basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning.

        Args:
            data: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Copy to avoid modification issues
        cleaned = data.copy()

        # Drop rows with null LapTime (these are often in/out laps)
        lap_time_null = cleaned["LapTime"].isnull()
        logger.info(f"Dropping {lap_time_null.sum()} rows with null LapTime")
        cleaned = cleaned[~lap_time_null]

        # Drop invalidated laps if IsPersonalBest and Time are both NaN
        is_invalid = cleaned["IsPersonalBest"].isnull() & cleaned["Time"].isnull()
        logger.info(f"Dropping {is_invalid.sum()} invalidated laps")
        cleaned = cleaned[~is_invalid]

        # Fill missing PitInTime and PitOutTime with 0 (no pit)
        cleaned["PitInTime"] = cleaned["PitInTime"].fillna(pd.Timedelta(0))
        cleaned["PitOutTime"] = cleaned["PitOutTime"].fillna(pd.Timedelta(0))

        # Convert categorical features
        for col in ["Team", "Driver", "Compound"]:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].astype("category")

        return cleaned

    def _process_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and convert time-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with processed time features
        """
        # Make a copy
        result = data.copy()

        # Convert timedelta to seconds for easier modeling
        timedelta_columns = [
            "LapTime",
            "PitInTime",
            "PitOutTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
        ]

        for col in timedelta_columns:
            if col in result.columns and not result[col].isnull().all():
                try:
                    # Convert to seconds (float)
                    result[f"{col}_seconds"] = result[col].dt.total_seconds()
                except Exception as e:
                    logger.warning(f"Could not convert {col} to seconds: {str(e)}")

        # Calculate percentage of lap time for each sector
        if all(
            col in result.columns
            for col in [
                "Sector1Time_seconds",
                "Sector2Time_seconds",
                "Sector3Time_seconds",
            ]
        ):
            total_time = (
                result["Sector1Time_seconds"]
                + result["Sector2Time_seconds"]
                + result["Sector3Time_seconds"]
            )
            result["Sector1Pct"] = result["Sector1Time_seconds"] / total_time
            result["Sector2Pct"] = result["Sector2Time_seconds"] / total_time
            result["Sector3Pct"] = result["Sector3Time_seconds"] / total_time

        return result

    def _process_compound_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to tire compounds.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with tire compound features
        """
        result = data.copy()

        # Check if Compound column exists
        if "Compound" not in result.columns:
            logger.warning("Compound column not found, skipping compound features")
            return result

        # Create dummy variables for compound (one-hot encoding)
        compounds = pd.get_dummies(result["Compound"], prefix="Compound")
        result = pd.concat([result, compounds], axis=1)

        # Create a compound hardness approximation (softer tires are generally faster but degrade faster)
        compound_hardness = {
            "SOFT": 1,
            "MEDIUM": 2,
            "HARD": 3,
            "INTERMEDIATE": 4,
            "WET": 5,
        }

        try:
            result["CompoundHardness"] = result["Compound"].map(compound_hardness)
            # Fill values that weren't mapped (unknown compounds) with median
            median_hardness = result["CompoundHardness"].median()
            result["CompoundHardness"] = result["CompoundHardness"].fillna(
                median_hardness
            )
        except Exception as e:
            logger.warning(f"Error creating compound hardness feature: {str(e)}")

        return result

    def _process_position_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to track position.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with position features
        """
        result = data.copy()

        # Calculate position changes
        if "Position" in result.columns:
            # Group by Driver, Event, Year
            groups = result.groupby(["Driver", "Event", "Year"])

            # Calculate position change for each group
            position_changes = []
            for _, group in groups:
                # Sort by LapNumber
                sorted_group = group.sort_values("LapNumber")
                # Calculate position change
                sorted_group["PositionChange"] = sorted_group["Position"].diff()
                position_changes.append(sorted_group)

            # Combine back
            if position_changes:
                result = pd.concat(position_changes)

            # Fill NaN values (first lap for each driver)
            result["PositionChange"] = result["PositionChange"].fillna(0)

        return result

    def _process_stint_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to tire stints.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with stint features
        """
        result = data.copy()

        # Group by Driver, Event, Year, and Compound
        groups = result.groupby(["Driver", "Event", "Year"])

        # Identify stints
        stint_data = []
        for _, group in groups:
            # Sort by LapNumber
            sorted_group = group.sort_values("LapNumber")

            # Initialize stint counter
            stint_id = 1
            previous_compound = None
            stint_lap = 0

            # Create stint IDs and lap counters
            sorted_group["StintID"] = 0
            sorted_group["StintLap"] = 0

            for idx, row in sorted_group.iterrows():
                current_compound = row["Compound"]

                # If compound changes or PitOutTime > 0, it's a new stint
                if (
                    previous_compound != current_compound
                    or row["PitOutTime_seconds"] > 0
                ):
                    stint_id += 1
                    stint_lap = 1
                else:
                    stint_lap += 1

                sorted_group.at[idx, "StintID"] = stint_id
                sorted_group.at[idx, "StintLap"] = stint_lap
                previous_compound = current_compound

            stint_data.append(sorted_group)

        # Combine back
        if stint_data:
            result = pd.concat(stint_data)

        return result

    def _calculate_deltas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delta times between laps.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with delta features
        """
        result = data.copy()

        if "LapTime_seconds" not in result.columns:
            logger.warning("LapTime_seconds not found, skipping delta calculations")
            return result

        # Group by Driver, Event, Year
        groups = result.groupby(["Driver", "Event", "Year"])

        # Calculate deltas for each group
        delta_data = []
        for _, group in groups:
            # Sort by LapNumber
            sorted_group = group.sort_values("LapNumber")

            # Calculate lap time delta
            sorted_group["LapTimeDelta"] = sorted_group["LapTime_seconds"].diff()

            # Calculate rolling statistics
            sorted_group["LapTime_3Lap_Mean"] = (
                sorted_group["LapTime_seconds"].rolling(window=3).mean()
            )
            sorted_group["LapTime_3Lap_Std"] = (
                sorted_group["LapTime_seconds"].rolling(window=3).std()
            )

            delta_data.append(sorted_group)

        # Combine back
        if delta_data:
            result = pd.concat(delta_data)

        return result

    def _process_race_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create race-specific features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with race features
        """
        result = data.copy()

        # Calculate race progress percentage
        race_groups = result[result["Session"] == "R"].groupby(["Event", "Year"])
        race_progress_data = []

        for _, group in race_groups:
            max_laps = group["LapNumber"].max()
            group["RaceProgressPct"] = group["LapNumber"] / max_laps
            race_progress_data.append(group)

        # Replace race data with progress data
        if race_progress_data:
            # Remove all race data
            non_race_data = result[result["Session"] != "R"]
            # Add back race data with progress
            race_data = pd.concat(race_progress_data)
            # Combine
            result = pd.concat([non_race_data, race_data])

        # Reset index
        result.reset_index(drop=True, inplace=True)

        return result

    def analyze_data(
        self, data: pd.DataFrame, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze processed data and generate insights.

        Args:
            data: Processed DataFrame
            output_dir: Directory to save visualizations

        Returns:
            Dictionary of analysis results
        """
        if data.empty:
            logger.warning("Empty DataFrame provided. No analysis performed.")
            return {}

        results = {}

        # Set up output directory if provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Basic statistics
        results["basic_stats"] = {
            "total_laps": len(data),
            "unique_drivers": data["Driver"].nunique(),
            "unique_events": data["Event"].nunique(),
            "unique_compounds": data["Compound"].nunique()
            if "Compound" in data.columns
            else 0,
        }

        # Lap time statistics
        if "LapTime_seconds" in data.columns:
            lap_time_stats = data["LapTime_seconds"].describe()
            results["lap_time_stats"] = lap_time_stats.to_dict()

            # Visualize lap time distributions
            if output_dir:
                plt.figure(figsize=(10, 6))
                sns.histplot(data["LapTime_seconds"], kde=True)
                plt.title("Distribution of Lap Times")
                plt.xlabel("Lap Time (seconds)")
                plt.ylabel("Frequency")
                plt.savefig(output_path / "lap_time_distribution.png")
                plt.close()

        # Compound performance analysis
        if "Compound" in data.columns and "LapTime_seconds" in data.columns:
            compound_perf = data.groupby("Compound")["LapTime_seconds"].agg(
                ["mean", "std", "min", "max", "count"]
            )
            results["compound_performance"] = compound_perf.to_dict()

            # Visualize compound performance
            if output_dir:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x="Compound", y="LapTime_seconds", data=data)
                plt.title("Lap Time Distribution by Compound")
                plt.ylabel("Lap Time (seconds)")
                plt.xlabel("Compound")
                plt.savefig(output_path / "compound_performance.png")
                plt.close()

        # Weather impact analysis
        if "TrackTemp_Avg" in data.columns and "LapTime_seconds" in data.columns:
            # Correlation of track temperature with lap times
            track_temp_corr = data["TrackTemp_Avg"].corr(data["LapTime_seconds"])
            results["track_temp_correlation"] = track_temp_corr

            # Visualize weather impact
            if output_dir:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    x="TrackTemp_Avg", y="LapTime_seconds", data=data, alpha=0.6
                )
                plt.title("Lap Time vs Track Temperature")
                plt.xlabel("Track Temperature (°C)")
                plt.ylabel("Lap Time (seconds)")
                plt.savefig(output_path / "track_temp_impact.png")
                plt.close()

        # Stint analysis
        if "StintLap" in data.columns and "LapTime_seconds" in data.columns:
            # Average lap time by stint lap
            stint_perf = (
                data.groupby("StintLap")["LapTime_seconds"].mean().reset_index()
            )
            results["stint_performance"] = {
                "stint_lap": stint_perf["StintLap"].tolist(),
                "avg_lap_time": stint_perf["LapTime_seconds"].tolist(),
            }

            # Visualize stint performance
            if output_dir and len(stint_perf) > 1:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x="StintLap", y="LapTime_seconds", data=stint_perf)
                plt.title("Average Lap Time by Stint Lap")
                plt.xlabel("Lap in Stint")
                plt.ylabel("Avg Lap Time (seconds)")
                plt.savefig(output_path / "stint_performance.png")
                plt.close()

        return results

    def visualize_race_simulation(
        self,
        data: pd.DataFrame,
        event: str,
        year: int,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Create visualizations for race simulation.

        Args:
            data: Processed race data
            event: Event name to filter
            year: Year to filter
            output_path: Path to save the visualization
        """
        # Filter data for the specified race
        race_data = data[
            (data["Event"] == event) & (data["Year"] == year) & (data["Session"] == "R")
        ]

        if race_data.empty:
            logger.warning(f"No race data found for {event} {year}")
            return

        # Set up plot
        plt.figure(figsize=(15, 10))
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        # Top drivers to analyze
        top_drivers = race_data["Driver"].value_counts().head(6).index.tolist()

        colors = {
            "SOFT": "red",
            "MEDIUM": "yellow",
            "HARD": "white",
            "INTERMEDIATE": "green",
            "WET": "blue",
        }

        # Plot 1: Lap times
        for driver in top_drivers:
            driver_data = race_data[race_data["Driver"] == driver]

            # Plot each stint with different color
            for stint_id, stint in driver_data.groupby("StintID"):
                compound = stint["Compound"].iloc[0]
                color = colors.get(compound, "gray")
                axs[0].plot(
                    stint["LapNumber"],
                    stint["LapTime_seconds"],
                    "o-",
                    label=f"{driver} ({compound})" if stint_id == 1 else "",
                    color=color,
                    alpha=0.7,
                )

                # Mark pit stops
                pit_laps = driver_data[driver_data["PitOutTime_seconds"] > 0][
                    "LapNumber"
                ]
                if not pit_laps.empty:
                    axs[0].vlines(
                        pit_laps,
                        race_data["LapTime_seconds"].min(),
                        race_data["LapTime_seconds"].max(),
                        colors="gray",
                        linestyles="dashed",
                        alpha=0.5,
                    )

        axs[0].set_title(f"Lap Times - {event} {year}")
        axs[0].set_ylabel("Lap Time (s)")
        axs[0].legend(loc="upper right")
        axs[0].grid(True, alpha=0.3)

        # Plot 2: Track position
        for driver in top_drivers:
            driver_data = race_data[race_data["Driver"] == driver]
            axs[1].plot(
                driver_data["LapNumber"], driver_data["Position"], "o-", label=driver
            )

        axs[1].set_title("Track Position")
        axs[1].set_ylabel("Position")
        axs[1].invert_yaxis()  # Lower position number is better
        axs[1].legend(loc="upper right")
        axs[1].grid(True, alpha=0.3)

        # Plot 3: Tire age
        for driver in top_drivers:
            driver_data = race_data[race_data["Driver"] == driver]
            axs[2].plot(
                driver_data["LapNumber"], driver_data["StintLap"], "o-", label=driver
            )

        axs[2].set_title("Tire Age")
        axs[2].set_xlabel("Lap Number")
        axs[2].set_ylabel("Laps on Current Tires")
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()
