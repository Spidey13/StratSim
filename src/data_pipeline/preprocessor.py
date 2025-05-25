import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
from datetime import timedelta
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataPreprocessor:
    def __init__(self, categories_dir: Optional[str] = None):
        """
        Initialize the F1 data preprocessor with all columns needed for agentic simulation.

        Args:
            categories_dir: Optional custom directory to save category JSON files.
                            If None, uses PROJECT_ROOT/data/categories/
        """
        # Set up paths for saving category lists
        self.project_root = Path(__file__).resolve().parent.parent.parent

        if categories_dir:
            self.categories_dir = Path(categories_dir)
        else:
            self.categories_dir = self.project_root / "data" / "categories"

        # Define paths for category files
        self.drivers_json_path = self.categories_dir / "drivers.json"
        self.teams_json_path = self.categories_dir / "teams.json"
        self.events_json_path = self.categories_dir / "events.json"

        # Time columns to convert to seconds
        self.time_columns = [
            "Time",
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "Sector1SessionTime",
            "Sector2SessionTime",
            "Sector3SessionTime",
            "LapStartTime",
            "PitOutTime",
            "PitInTime",
        ]

        # Integer columns
        self.int_columns = [
            "Position",
            "LapNumber",
            "DriverNumber",
            "TrackStatus",
            "Year",
            "Rainfall",
            "TrackCondition",
            "WetTrack",
            "TyreLife",
            "Stint",
            "StintNumber",
            "StintLength",
        ]

        # Float columns
        self.float_columns = [
            "SpeedI1",
            "SpeedI2",
            "SpeedFL",
            "SpeedST",
            "Distance",
            "AirTemp_Avg",
            "TrackTemp_Avg",
            "Humidity_Avg",
            "WindSpeed_Avg",
            "AirTemp_Min",
            "AirTemp_Max",
            "TrackTemp_Min",
            "TrackTemp_Max",
            "AirTemp_Std",
            "TrackTemp_Std",
            "Humidity_Std",
            "WindSpeed_Std",
            "Circuit_Lat",
            "Circuit_Long",
            "TempDelta",
            "WeatherStability",
            "GripLevel",
            "PitStopDuration",
            "SectorSum",
            "LapTimeDelta",
            "PositionChange",
            "TrackTempDelta",
            "CompoundHardness",
        ]

        # Boolean columns (including new tire and pit flags)
        self.bool_columns = [
            "IsPersonalBest",
            "FreshTyre",
            "Deleted",
            "FastF1Generated",
            "IsAccurate",
            "Outlap",
            "Inlap",
            "PitInLap",
            "PitOutLap",
            "IsSoft",
            "IsMedium",
            "IsHard",
            "IsIntermediate",
            "IsWet",
            "HasPitIn",
            "HasPitOut",
            "IsOutlap",
            "IsInlap",
            "SafetyCar",
            "VirtualSafetyCar",
            "RedFlag",
            "YellowFlag",
            "IsRain",
        ]

        # Valid categories for categorical columns
        self.valid_categories = {
            "Compound": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"],
            "Session": ["FP1", "FP2", "FP3", "Q", "R"],
            "TrackStatus": ["1", "2", "3", "4", "5", "6", "12", "251", "UNKNOWN"],
            "TrackCondition": ["0", "1", "2", "3"],
        }

        # Columns to drop
        self.columns_to_drop = ["DeletedReason"]

        # Value ranges
        self.value_ranges = {
            "TyreLife": (0, 60),
            "Position": (1, 20),
            "LapNumber": (1, 100),
            "SpeedI1": (0, 400),
            "SpeedI2": (0, 400),
            "SpeedFL": (0, 400),
            "SpeedST": (0, 400),
            "AirTemp_Avg": (-10, 50),
            "TrackTemp_Avg": (-5, 70),
            "Humidity_Avg": (0, 100),
            "WindSpeed_Avg": (0, 100),
            "GripLevel": (0, 10),
            "TrackCondition": (0, 3),
        }

        # Columns to preserve
        self.preserve_columns = [
            "Driver",
            "Team",
            "Event",
            "Session",
            "Year",
            "Compound",
            "LapTime",
            "LapNumber",
            "Stint",
            "TyreLife",
        ]

    def ensure_categories_dir(self) -> None:
        """Ensure the categories directory exists."""
        try:
            self.categories_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Ensured categories directory exists at: {self.categories_dir}"
            )
        except Exception as e:
            logger.error(f"Error creating categories directory: {e}")
            raise

    def save_to_json(
        self, data: Union[List, Dict], output_path: Path, name: str = "items"
    ) -> bool:
        """
        Save data to a JSON file with error handling.

        Args:
            data: The data to save
            output_path: Path where to save the JSON file
            name: Description of what's being saved (for logging)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the data
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)

            logger.info(
                f"Successfully saved {len(data) if isinstance(data, list) else 'N/A'} {name} to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error saving {name} to {output_path}: {e}")
            return False

    def extract_and_save_categories(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Extract and save unique values for Driver, Team, and Event columns.
        Also creates driver-team relationships.

        Args:
            df: DataFrame containing the data

        Returns:
            Dict with results for each category type
        """
        # Make sure categories directory exists
        self.ensure_categories_dir()

        results = {}

        # Extract and save Driver categories with team relationships
        if "Driver" in df.columns and "Team" in df.columns:
            # Create a mapping of drivers to their teams
            driver_teams = {}
            for driver, group in df.groupby("Driver"):
                # Get unique teams for this driver
                teams = sorted(group["Team"].unique().tolist())

                # Get years the driver participated
                if "Year" in df.columns:
                    years = sorted(group["Year"].unique().tolist())
                else:
                    years = []

                # Store driver data with teams
                driver_teams[str(driver)] = {
                    "teams": teams,
                    "years": years,
                }

            # Save the driver-team mapping
            results["drivers"] = self.save_to_json(
                driver_teams, self.drivers_json_path, "drivers with team mapping"
            )
        elif "Driver" in df.columns:
            # Fallback to just saving driver names if Team column doesn't exist
            unique_drivers = sorted(df["Driver"].astype(str).unique().tolist())
            results["drivers"] = self.save_to_json(
                unique_drivers, self.drivers_json_path, "drivers"
            )
            logger.warning(
                "No 'Team' column found - saved drivers without team mapping"
            )
        else:
            logger.warning("No 'Driver' column found in data")
            results["drivers"] = False

        # Extract and save Team categories with driver relationships
        if "Team" in df.columns and "Driver" in df.columns:
            # Create a mapping of teams to their drivers
            team_drivers = {}
            for team, group in df.groupby("Team"):
                # Get unique drivers for this team
                drivers = sorted(group["Driver"].unique().tolist())

                # Get years the team participated
                if "Year" in df.columns:
                    years = sorted(group["Year"].unique().tolist())
                else:
                    years = []

                # Store team data with drivers
                team_drivers[str(team)] = {
                    "drivers": drivers,
                    "years": years,
                }

            # Save the team-driver mapping
            results["teams"] = self.save_to_json(
                team_drivers, self.teams_json_path, "teams with driver mapping"
            )
        elif "Team" in df.columns:
            # Fallback to just saving team names
            unique_teams = sorted(df["Team"].astype(str).unique().tolist())
            results["teams"] = self.save_to_json(
                unique_teams, self.teams_json_path, "teams"
            )
            logger.warning(
                "No 'Driver' column found - saved teams without driver mapping"
            )
        else:
            logger.warning("No 'Team' column found in data")
            results["teams"] = False

        # Extract and save Event categories with additional metadata
        if "Event" in df.columns:
            # Create a mapping of events with additional data
            event_data = {}
            for event, group in df.groupby("Event"):
                # Get the year(s) for this event
                if "Year" in df.columns:
                    years = sorted(group["Year"].unique().tolist())
                else:
                    years = []

                # Get unique drivers participating in this event
                if "Driver" in df.columns:
                    drivers = sorted(group["Driver"].unique().tolist())
                else:
                    drivers = []

                # Store event data
                event_data[str(event)] = {
                    "years": years,
                    "drivers": drivers,
                }

            # Save the event data
            results["events"] = self.save_to_json(
                event_data, self.events_json_path, "events with metadata"
            )
        else:
            logger.warning("No 'Event' column found in data")
            results["events"] = False

        return results

    def clean_time_string(self, time_str: str) -> str:
        """Clean time string format."""
        if pd.isna(time_str):
            return np.nan

        try:
            # Convert to string and handle different formats
            time_str = str(time_str).strip()

            # Remove 'days' part if present
            if "days" in time_str:
                time_str = time_str.split("days")[-1].strip()

            return time_str

        except:
            return np.nan

    def convert_timedelta_to_seconds(self, time_val) -> float:
        """Convert various time representations to seconds (float)."""
        try:
            if pd.isna(time_val):
                return np.nan

            # 1. If already a timedelta, convert directly
            if isinstance(time_val, pd.Timedelta):
                return time_val.total_seconds()

            # 2. If numeric, assume it's already seconds
            if isinstance(time_val, (int, float)):
                return float(time_val)

            # 3. If a string, try parsing with pd.to_timedelta
            if isinstance(time_val, str):
                # Handle potential 'X days HH:MM:SS.ffffff' format first
                if "days" in time_val:
                    try:
                        # Extract days and time parts
                        parts = time_val.split(" days ")
                        if len(parts) == 2:
                            days = float(parts[0])
                            time_part_str = parts[1]
                            time_delta = pd.to_timedelta(time_part_str, errors="coerce")
                            if pd.isna(time_delta):
                                logger.warning(
                                    f"Could not parse time part: {time_part_str}"
                                )
                                return np.nan
                            return (days * 86400) + time_delta.total_seconds()
                        else:
                            logger.warning(f"Unexpected 'days' format: {time_val}")
                            # Fall through to standard parsing
                    except Exception as e:
                        logger.warning(
                            f"Error parsing 'days' format {time_val}: {str(e)}"
                        )
                        # Fall through to standard parsing

                # Standard parsing for HH:MM:SS.fff or similar
                td = pd.to_timedelta(time_val, errors="coerce")
                if pd.isna(td):
                    logger.warning(f"Could not parse time string: {time_val}")
                    return np.nan
                return td.total_seconds()

            # 4. Handle other potential types (like numpy timedeltas)
            if hasattr(time_val, "total_seconds") and callable(time_val.total_seconds):
                try:
                    return time_val.total_seconds()
                except Exception as e:
                    logger.warning(
                        f"Failed to call total_seconds() on {type(time_val)}: {str(e)}"
                    )
                    return np.nan

            # If none of the above, log warning and return NaN
            logger.warning(
                f"Unhandled time format: {time_val} (Type: {type(time_val)})"
            )
            return np.nan

        except Exception as e:
            logger.error(
                f"Unexpected error converting time: {time_val}, Error: {str(e)}"
            )
            return np.nan

    def convert_to_boolean(self, value) -> bool:
        """Convert various values to boolean."""
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ["true", "1", "t", "y", "yes"]
        return False

    def process_speed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process speed trap data."""
        speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]

        # Only process columns that exist and have data
        valid_speed_cols = [
            col for col in speed_cols if col in df.columns and df[col].notna().any()
        ]

        if not valid_speed_cols:
            logger.info("No valid speed data columns found")
            return df

        logger.info(f"Processing speed data for columns: {', '.join(valid_speed_cols)}")

        for col in valid_speed_cols:
            # Fill missing values with sector averages
            df[col] = df.groupby(["Event", "Session", "Driver"])[col].transform(
                lambda x: x.fillna(x.mean())
            )

            # Calculate speed differences
            df[f"{col}_Diff"] = df.groupby(["Event", "Session", "Driver"])[col].diff()

        return df

    def process_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process weather data from FastF1."""
        logger.info("Processing weather data...")

        # List of weather columns to process
        weather_cols = [
            col
            for col in df.columns
            if any(
                term in col
                for term in [
                    "Temp",
                    "Humidity",
                    "Wind",
                    "Rain",
                    "Track",
                    "Grip",
                    "Weather",
                ]
            )
        ]

        if not weather_cols:
            logger.warning("No weather columns found in the dataset")
            return df

        logger.info(f"Found weather columns: {', '.join(weather_cols)}")

        # Fill missing weather data with event averages
        for col in weather_cols:
            if col in df.columns and df[col].isna().any():
                missing_count = df[col].isna().sum()
                logger.info(f"Filling {missing_count} missing values in {col}")

                # First try to fill with session/event average
                df[col] = df.groupby(["Year", "Event", "Session"])[col].transform(
                    lambda x: x.fillna(x.mean())
                )

                # If still missing, fill with event average
                if df[col].isna().any():
                    df[col] = df.groupby(["Year", "Event"])[col].transform(
                        lambda x: x.fillna(x.mean())
                    )

                # If still missing after group averages, use global average or zero
                if df[col].isna().any():
                    if df[col].notna().any():
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        logger.warning(f"No valid data for {col}, filling with zeros")
                        df[col] = df[col].fillna(0)

        # Create derived weather features
        if all(col in df.columns for col in ["AirTemp_Avg", "TrackTemp_Avg"]):
            # Temperature delta can affect tire performance
            logger.info("Creating TempDelta feature")
            df["TempDelta"] = df["TrackTemp_Avg"] - df["AirTemp_Avg"]

        # Ensure Rainfall is properly formatted as binary
        if "Rainfall" in df.columns:
            logger.info("Ensuring Rainfall is formatted as binary")
            df["Rainfall"] = df["Rainfall"].apply(lambda x: 1 if x > 0 else 0)

            # Also create WetTrack flag if it doesn't exist
            if "WetTrack" not in df.columns:
                df["WetTrack"] = df["Rainfall"]

        # Create grip level estimation if it doesn't exist
        if "GripLevel" not in df.columns and all(
            col in df.columns for col in ["TrackTemp_Avg", "Humidity_Avg", "Rainfall"]
        ):
            logger.info("Creating GripLevel feature")
            # Higher temp → better grip, higher humidity → worse grip, rainfall → much worse grip
            df["GripLevel"] = (
                (df["TrackTemp_Avg"] / 100) * 5  # Scale from 0-5 based on track temp
                - (df["Humidity_Avg"] / 100) * 2  # Reduce by 0-2 based on humidity
                - df["Rainfall"] * 4  # Reduce by 4 if wet
            )
            # Clip to reasonable range
            df["GripLevel"] = df["GripLevel"].clip(0, 10)

        # Create weather stability index if it doesn't exist
        if "WeatherStability" not in df.columns:
            weather_std_cols = [col for col in df.columns if col.endswith("_Std")]
            if len(weather_std_cols) > 0:
                logger.info("Creating WeatherStability feature")
                # Normalize standard deviations
                norm_cols = []
                for col in weather_std_cols:
                    if df[col].max() > 0:
                        new_col = f"{col}_Norm"
                        df[new_col] = df[col] / df[col].max()
                        norm_cols.append(new_col)

                if norm_cols:
                    df["WeatherStability"] = df[norm_cols].sum(axis=1)
                    # Clean up temporary columns
                    df = df.drop(columns=norm_cols)

        return df

    def process_pit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process pit stop and stint related data."""
        # Use the '_s' columns for time arithmetic
        if all(col in df.columns for col in ["PitInTime_s", "PitOutTime_s"]):
            # Calculate pit stop duration in seconds
            df["PitStopDuration"] = df["PitOutTime_s"] - df["PitInTime_s"]

            # Mark pit laps
            df["PitLap"] = df["PitInTime_s"].notna() | df["PitOutTime_s"].notna()

            # Calculate stint length - check if Compound column exists
            if "Compound" in df.columns:
                # Calculate stint length using Compound column
                df["StintLength"] = df.groupby(
                    ["Event", "Session", "Driver", "Compound"]
                )["LapNumber"].transform("count")
            else:
                # If no Compound column, calculate stint length without it
                logger.warning(
                    "Compound column not found, calculating stint length without tire information"
                )
                df["StintLength"] = df.groupby(["Event", "Session", "Driver"])[
                    "LapNumber"
                ].transform("count")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset with sophisticated strategies."""
        logger.info("Handling missing values...")
        df = df.copy()

        # Only process speed data if at least one speed column has data
        if any(
            col in df.columns and df[col].notna().any()
            for col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
        ):
            df = self.process_speed_data(df)
        else:
            logger.info("Skipping speed data processing - no valid speed data found")

        # Only process pit data if pit columns have data (use _s columns)
        if all(col in df.columns for col in ["PitInTime_s", "PitOutTime_s"]) and (
            df["PitInTime_s"].notna().any() or df["PitOutTime_s"].notna().any()
        ):
            df = self.process_pit_data(df)
        else:
            logger.info("Skipping pit data processing - no valid pit data found")

        # Process weather data
        df = self.process_weather_data(df)

        # 1. Time-related missing values - use numeric _s columns
        sector_cols_s = ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
        lap_time_s_col = "LapTime_s"

        if (
            all(col in df.columns for col in sector_cols_s)
            and lap_time_s_col in df.columns
        ):
            logger.info("Processing missing time data using _s columns...")

            # Calculate sector sum using numeric seconds
            sector_sum_s = df[sector_cols_s].sum(
                axis=1, skipna=False
            )  # Keep NaNs if any sector is NaN

            # Fill missing LapTime_s using sector_sum_s
            fill_mask = df[lap_time_s_col].isna() & sector_sum_s.notna()
            if fill_mask.any():
                logger.info(
                    f"Filling {fill_mask.sum()} missing LapTime_s values using sector sums."
                )
                df.loc[fill_mask, lap_time_s_col] = sector_sum_s[fill_mask]

            # Fill missing sector times using proportions from similar laps (using _s columns)
            if (
                df[lap_time_s_col].notna().any()
            ):  # Ensure LapTime_s has valid values for proportion calculation
                for sector_s_col in sector_cols_s:
                    if df[sector_s_col].notna().any():
                        # Calculate proportion based on LapTime_s
                        df[f"{sector_s_col}_pct"] = (
                            df[sector_s_col] / df[lap_time_s_col]
                        )
                        df[f"{sector_s_col}_pct"] = df[f"{sector_s_col}_pct"].replace(
                            [np.inf, -np.inf], np.nan
                        )

                        if df[f"{sector_s_col}_pct"].notna().any():
                            # Group by Event and Driver (ensure these are categorical or handled correctly)
                            group_cols = ["Event", "Driver"]
                            if all(c in df.columns for c in group_cols):
                                mean_proportions = df.groupby(
                                    group_cols, observed=False
                                )[f"{sector_s_col}_pct"].transform("mean")

                                # Fill where sector is missing but lap time is not
                                missing_sectors = (
                                    df[sector_s_col].isna() & df[lap_time_s_col].notna()
                                )
                                if (
                                    missing_sectors.any()
                                    and mean_proportions.notna().any()
                                ):
                                    logger.info(
                                        f"Filling {missing_sectors.sum()} missing {sector_s_col} values using proportions."
                                    )
                                    df.loc[missing_sectors, sector_s_col] = (
                                        df.loc[missing_sectors, lap_time_s_col]
                                        * mean_proportions[missing_sectors]
                                    )
                            else:
                                logger.warning(
                                    f"Cannot calculate mean proportions for {sector_s_col} - missing grouping columns."
                                )
                        # Drop temporary percentage column
                        if f"{sector_s_col}_pct" in df.columns:
                            df = df.drop(columns=[f"{sector_s_col}_pct"])
            else:
                logger.warning(
                    "Skipping sector time proportion filling - LapTime_s has no valid values."
                )
        else:
            logger.info(
                "Skipping sector time filling - insufficient sector_s or LapTime_s columns."
            )

        # 2. Position data
        if "Position" in df.columns and df["Position"].notna().any():
            # First, forward fill within each lap
            df["Position"] = df.groupby(["Event", "Session", "LapNumber", "Driver"])[
                "Position"
            ].transform(lambda x: x.ffill())

            # Then backward fill
            df["Position"] = df.groupby(["Event", "Session", "LapNumber", "Driver"])[
                "Position"
            ].transform(lambda x: x.bfill())

            # For any remaining NAs, use the previous lap's position
            df["Position"] = df.groupby(["Event", "Session", "Driver"])[
                "Position"
            ].transform(lambda x: x.ffill())

        # 3. Compound/Tire data
        if "Compound" in df.columns:
            logger.info("Processing compound/tire data")
            # Forward fill tire compound within each stint
            df["Compound"] = df.groupby(["Event", "Session", "Driver"])[
                "Compound"
            ].transform(lambda x: x.ffill())
            df["Compound"] = df["Compound"].fillna("UNKNOWN")

        if "TyreLife" in df.columns and "FreshTyre" in df.columns:
            # Calculate tyre life based on stint length where missing
            df["StintStart"] = (~df["FreshTyre"].astype(bool)).astype(int).cumsum()
            df["TyreLife"] = df.groupby(
                ["Event", "Session", "Driver", "StintStart"]
            ).cumcount()
            df = df.drop(columns=["StintStart"])
        elif "TyreLife" in df.columns and df["TyreLife"].isna().any():
            # If we have TyreLife but no FreshTyre, try to calculate based on Stint
            if "Stint" in df.columns and df["Stint"].notna().any():
                logger.info("Calculating TyreLife from Stint information")
                # Group by Stint and count laps within each stint
                df["TyreLife"] = df.groupby(
                    ["Event", "Session", "Driver", "Stint"]
                ).cumcount()
            else:
                logger.warning(
                    "Cannot calculate TyreLife - missing Stint/FreshTyre information"
                )

        # 4. Boolean columns
        for col in self.bool_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False)

        # Report remaining missing values
        remaining_nas = df.isnull().sum()
        if remaining_nas.any():
            logger.warning("Remaining missing values:")
            for col, count in remaining_nas[remaining_nas > 0].items():
                # Only report if more than 1% of values are missing
                if count / len(df) > 0.01:
                    logger.warning(f"{col}: {count} ({(count / len(df) * 100):.2f}%)")

        return df

    def enforce_value_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce valid ranges for numerical columns."""
        logger.info("Enforcing value ranges...")

        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns:
                # Cap values at min/max
                df[col] = df[col].clip(min_val, max_val)

        return df

    def validate_lap_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean lap times based on FastF1 recommendations and handle potential errors."""
        logger.info("Validating lap times...")
        df = df.copy()

        # Ensure main time columns are timedelta for comparison
        try:
            if "Time" in df.columns and df["Time"].dtype == "object":
                df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")
            if "LapTime" in df.columns and df["LapTime"].dtype == "object":
                df["LapTime"] = pd.to_timedelta(df["LapTime"], errors="coerce")
        except Exception as e:
            logger.warning(
                f"Error converting base Time/LapTime columns to timedelta: {str(e)}"
            )
            # Continue, but lap duplication check might be skipped if conversion failed

        # Check for duplicate lap times (only if conversion worked)
        if all(col in df.columns for col in ["Time", "LapTime"]) and all(
            pd.api.types.is_timedelta64_dtype(df[col]) for col in ["Time", "LapTime"]
        ):
            try:
                df["sub_time"] = df["Time"].diff()
                df["bool_check"] = (
                    df["sub_time"] - df["LapTime"]
                ).abs() < pd.Timedelta(milliseconds=1)
                df["bool_previous_lap"] = df["LapTime"] == df["LapTime"].shift(1)

                # Remove duplicate laps by masking
                df["LapTime"] = df["LapTime"].mask(
                    (df["bool_check"] == False) & (df["bool_previous_lap"] == True),
                    pd.NaT,
                )
                df = df.drop(
                    ["sub_time", "bool_check", "bool_previous_lap"],
                    axis=1,
                    errors="ignore",
                )
            except Exception as e:
                logger.warning(f"Error during lap time duplication check: {str(e)}")
                # Clean up temp columns if they exist
                df = df.drop(
                    ["sub_time", "bool_check", "bool_previous_lap"],
                    axis=1,
                    errors="ignore",
                )
        else:
            logger.warning(
                "Skipping lap time duplication check due to missing or non-timedelta columns."
            )

        # Validate sector times sum matches lap time (using seconds)
        sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
        if all(col in df.columns for col in sector_cols) and "LapTime" in df.columns:
            try:
                # Convert relevant columns to seconds for validation calculation
                # Use existing _s columns if available, otherwise convert originals
                if "LapTime_s" in df.columns:
                    lap_time_s = df["LapTime_s"].copy()
                else:
                    lap_time_s = df["LapTime"].apply(self.convert_timedelta_to_seconds)

                sector_sum_s = pd.Series(0.0, index=df.index)
                valid_sectors = True
                for col in sector_cols:
                    if col + "_s" in df.columns:
                        sector_s = df[col + "_s"].copy()
                    else:
                        sector_s = df[col].apply(self.convert_timedelta_to_seconds)

                    # Ensure numeric before adding
                    sector_s = pd.to_numeric(sector_s, errors="coerce")
                    if sector_s.isna().all():
                        valid_sectors = False
                        logger.warning(
                            f"Sector column {col} could not be converted to numeric seconds for validation."
                        )
                        break
                    sector_sum_s = sector_sum_s.add(
                        sector_s.fillna(0), fill_value=0
                    )  # Add safely

                # Ensure LapTime is numeric for comparison
                lap_time_s = pd.to_numeric(lap_time_s, errors="coerce")

                if valid_sectors and not lap_time_s.isna().all():
                    # Flag inconsistent lap times (difference > 0.1 seconds)
                    # Only compare where both lap time and sector sum are valid numbers
                    valid_comparison = lap_time_s.notna() & sector_sum_s.notna()
                    difference = (lap_time_s - sector_sum_s).abs()
                    inconsistent_mask = valid_comparison & (difference > 0.1)

                    # Store consistency flag if needed (optional)
                    # df["LapTimeConsistent"] = valid_comparison & ~inconsistent_mask

                    # Use sector sum (in original timedelta format if possible) for inconsistent lap times
                    # Convert sector_sum_s back to timedelta for assignment
                    sector_sum_td = pd.to_timedelta(
                        sector_sum_s, unit="s", errors="coerce"
                    )
                    # Assign only where inconsistent and sector_sum_td is valid
                    assign_mask = inconsistent_mask & sector_sum_td.notna()
                    if assign_mask.any():
                        logger.info(
                            f"Correcting {assign_mask.sum()} inconsistent lap times using sector sums."
                        )
                        # Ensure LapTime column is suitable for timedelta assignment
                        if not pd.api.types.is_timedelta64_dtype(df["LapTime"]):
                            df["LapTime"] = pd.to_timedelta(
                                df["LapTime"], errors="coerce"
                            )  # Convert if needed
                        df.loc[assign_mask, "LapTime"] = sector_sum_td[assign_mask]

            except Exception as e:
                logger.error(
                    f"Critical error during sector time validation: {str(e)}",
                    exc_info=True,
                )
        else:
            logger.warning("Skipping sector time validation due to missing columns.")

        return df

    def clean_empty_columns(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove columns that are explicitly not useful for modeling or simulation."""
        logger.info("Cleaning unnecessary columns based on project requirements")

        # Explicitly list columns to remove (not useful for modeling/simulation)
        columns_to_remove = set(
            [
                "DeletedReason",
                "TrackStatusRaw",
                "TrackStatusText",
                # Add more columns here if you know they are not useful for modeling
            ]
        )
        # Only drop columns that are present in the DataFrame
        cols_to_drop = [col for col in columns_to_remove if col in df.columns]

        if cols_to_drop:
            logger.info(f"Dropping columns: {', '.join(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        else:
            logger.info("No unnecessary columns to drop")

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the F1 dataset with advanced feature engineering for agentic simulation."""
        logger.info("Starting data preprocessing...")
        df = df.copy()

        # Drop unnecessary columns
        initial_columns = set(df.columns)
        columns_to_drop = [
            col
            for col in self.columns_to_drop
            if col in df.columns and col not in self.preserve_columns
        ]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors="ignore")
            dropped_columns = initial_columns - set(df.columns)
            logger.info(f"Dropped columns: {', '.join(dropped_columns)}")
        else:
            logger.warning("No columns were dropped.")
        logger.info(f"Remaining columns: {', '.join(df.columns)}")

        # Validate lap times before conversion
        df = self.validate_lap_times(df)

        # DEBUG: Check LapTime column state before final conversion
        if "LapTime" in df.columns:
            logger.info(
                "DEBUG: Checking LapTime column before final conversion loop..."
            )
            logger.info(f"DEBUG: LapTime dtype: {df['LapTime'].dtype}")
            problematic_laptimes = df[
                df["LapTime"].apply(
                    lambda x: not isinstance(x, (pd.Timedelta, type(None), float, int))
                )
                & df["LapTime"].notna()
            ]
            if not problematic_laptimes.empty:
                logger.warning(
                    f"DEBUG: Found {len(problematic_laptimes)} non-timedelta/numeric LapTime values before conversion:"
                )
                logger.warning(problematic_laptimes[["LapTime"]].head())
            else:
                logger.info(
                    "DEBUG: All non-null LapTime values appear to be Timedelta or numeric."
                )
        else:
            logger.warning("DEBUG: LapTime column not found before conversion loop.")
        # END DEBUG

        # Convert time columns to seconds
        for col in [c for c in self.time_columns if c in df.columns]:
            logger.info(f"Converting {col} to seconds...")
            # Ensure the column is suitable for apply (e.g., handle mixed types if necessary)
            try:
                df[col + "_s"] = df[col].apply(self.convert_timedelta_to_seconds)
            except Exception as e:
                logger.error(
                    f"Error applying conversion to column {col}: {str(e)}",
                    exc_info=True,
                )
                # Optionally create the _s column with NaNs to prevent downstream errors
                df[col + "_s"] = np.nan

        # Feature engineering: LapTimeDelta, StintNumber, IsOutlap/Inlap, SectorSum, PositionChange, TrackTempDelta
        if "LapTime_s" in df.columns:
            df["LapTimeDelta"] = df.groupby(["Event", "Session", "Driver"])[
                "LapTime_s"
            ].diff()
        if "Compound" in df.columns:
            df["StintNumber"] = df.groupby(["Event", "Session", "Driver"])[
                "Compound"
            ].transform(lambda x: (x != x.shift(1)).cumsum())
        if "HasPitOut" in df.columns:
            df["IsOutlap"] = df["HasPitOut"].astype(bool)
        if "HasPitIn" in df.columns:
            df["IsInlap"] = df["HasPitIn"].astype(bool)
        sector_cols = [
            c + "_s"
            for c in ["Sector1Time", "Sector2Time", "Sector3Time"]
            if c + "_s" in df.columns
        ]
        if len(sector_cols) == 3:
            df["SectorSum"] = df[sector_cols].sum(axis=1)
        if "Position" in df.columns:
            df["PositionChange"] = df.groupby(["Event", "Session", "Driver"])[
                "Position"
            ].diff()
        if "TrackTemp_Avg" in df.columns and "AirTemp_Avg" in df.columns:
            df["TrackTempDelta"] = df["TrackTemp_Avg"] - df["AirTemp_Avg"]
        if "Rainfall" in df.columns:
            df["IsRain"] = df["Rainfall"] > 0

        # One-hot encode compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
        if "Compound" in df.columns:
            for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
                df[f"Is{comp.title()}"] = df["Compound"].str.upper() == comp

        # Forward/backward fill compound within stint
        if "Compound" in df.columns:
            df["Compound"] = (
                df.groupby(["Driver", "Session", "Event"])["Compound"].ffill().bfill()
            )

        # Categorical encoding for driver/team
        if "Driver" in df.columns:
            df["Driver"] = df["Driver"].astype("category")
        if "Team" in df.columns:
            df["Team"] = df["Team"].astype("category")

        # Safety car and flag indicators
        for flag in ["SafetyCar", "VirtualSafetyCar", "RedFlag", "YellowFlag"]:
            if flag in df.columns:
                df[flag] = df[flag].astype(bool)

        # Handle missing values and type conversions
        df = self.handle_missing_values(df)
        for col in [c for c in self.int_columns if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
        for col in [c for c in self.float_columns if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in [c for c in self.bool_columns if c in df.columns]:
            df[col] = df[col].apply(self.convert_to_boolean)
        for col in [c for c in self.valid_categories.keys() if c in df.columns]:
            df[col] = df[col].astype(str).replace("nan", "UNKNOWN").str.upper()
            df.loc[~df[col].isin(self.valid_categories[col]), col] = "UNKNOWN"

        # Clean up empty columns, but preserve important ones
        df = self.clean_empty_columns(df, threshold=0.95)

        logger.info("Data preprocessing completed")
        logger.info("Final column list:")
        for col in sorted(df.columns):
            logger.info(f"- {col}: {df[col].dtype}")
        return df

    def process_and_save(self, input_path: str, output_path: str) -> None:
        """
        Load data from input_path, preprocess it, and save to output_path.
        Also saves unique categorical values (Driver, Team, Event) to JSON files.
        """
        logger.info(f"Starting preprocessing for {input_path}")
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Ensure categories directory exists
            self.ensure_categories_dir()

            # Load input data
            df = pd.read_csv(input_path, low_memory=False)
            logger.info(f"Initial data loaded with shape: {df.shape}")

            # Preprocess data
            df = self.preprocess_data(df)
            logger.info(f"Data after preprocessing steps, shape: {df.shape}")

            # Extract and save category data
            category_results = self.extract_and_save_categories(df)

            for category, success in category_results.items():
                if success:
                    logger.info(f"Successfully saved {category} to JSON")
                else:
                    logger.warning(f"Failed to save {category} to JSON")

            # Save processed data
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully preprocessed data saved to {output_path}")

        except FileNotFoundError:
            logger.error(f"Input file not found: {input_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Input file is empty: {input_path}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during preprocessing and saving: {e}")
            raise
