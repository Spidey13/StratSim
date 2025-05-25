"""
F1 data collector module using the FastF1 API.
This module handles collection of F1 race data including lap times,
weather conditions, circuit information, tire data, and pit stops.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
import time
from requests.exceptions import ReadTimeout
import fastf1.req
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set FastF1 timeout to 30 seconds
fastf1.req.Cache.TIMEOUT = 30


class F1DataCollector:
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the F1 data collector with a cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Enable FastF1 cache
        fastf1.Cache.enable_cache(self.cache_dir)

    def _retry_with_backoff(self, func, *args, max_retries=3, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (ReadTimeout, KeyboardInterrupt) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) * 5  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)

    def merge_lap_weather_data(
        self, laps_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge lap data with corresponding weather data based on timestamp.

        Args:
            laps_df: DataFrame containing lap data
            weather_df: DataFrame containing weather data

        Returns:
            DataFrame with merged lap and weather data
        """
        if weather_df.empty:
            logger.warning("No weather data available for merging")
            return laps_df

        try:
            # Convert time to datetime for easier comparison
            if "Date" in weather_df.columns:
                weather_df["Date"] = pd.to_datetime(weather_df["Date"])

            # Ensure proper types for time columns
            if "Time" in laps_df.columns:
                if not isinstance(laps_df["Time"].iloc[0], pd.Timedelta):
                    # Convert string time to timedelta if needed
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        laps_df["Time"] = pd.to_timedelta(laps_df["Time"])

            # Create a copy of the laps DataFrame to avoid modification warnings
            merged_df = laps_df.copy()

            # Add weather data columns
            weather_cols = [
                "AirTemp",
                "Humidity",
                "Pressure",
                "Rainfall",
                "TrackTemp",
                "WindDirection",
                "WindSpeed",
            ]

            # Create mapping between weather columns and their average columns (if available)
            avg_col_mapping = {
                "AirTemp": "AirTemp_Avg",
                "Humidity": "Humidity_Avg",
                "Pressure": None,  # No average typically available
                "Rainfall": "Rainfall",
                "TrackTemp": "TrackTemp_Avg",
                "WindDirection": None,  # No average typically available
                "WindSpeed": "WindSpeed_Avg",
            }

            # Initialize weather columns with NaN
            for col in weather_cols:
                if col in weather_df.columns:
                    merged_df[f"Weather_{col}"] = np.nan

            # For each lap, find the closest weather data point
            match_count = 0
            for idx, lap in laps_df.iterrows():
                try:
                    # Find the weather data closest to the lap time
                    if "LapStartTime" in lap:
                        lap_time = lap["LapStartTime"]
                    elif "LapStartDate" in lap:
                        lap_time = lap["LapStartDate"]
                    else:
                        # Skip if no time information available
                        continue

                    # Find closest weather data point
                    if "Date" in weather_df.columns:
                        closest_idx = (weather_df["Date"] - lap_time).abs().idxmin()
                        closest_weather = weather_df.loc[closest_idx]

                        # Add weather data to the lap
                        for col in weather_cols:
                            if col in weather_df.columns:
                                merged_df.at[idx, f"Weather_{col}"] = closest_weather[
                                    col
                                ]
                        match_count += 1
                except Exception as e:
                    logger.warning(
                        f"Error matching weather data for lap {idx}: {str(e)}"
                    )

            logger.info(f"Successfully matched weather data for {match_count} laps")

            # Fill any remaining empty weather values with session averages if available
            empty_weather_count = 0
            for col in weather_cols:
                weather_col = f"Weather_{col}"
                if (
                    weather_col in merged_df.columns
                    and merged_df[weather_col].isna().any()
                ):
                    # Count empty values
                    empty_count = merged_df[weather_col].isna().sum()
                    empty_weather_count += empty_count

                    # Try to fill with corresponding average value
                    avg_col = avg_col_mapping.get(col)
                    if (
                        avg_col
                        and avg_col in merged_df.columns
                        and not merged_df[avg_col].isna().all()
                    ):
                        # Use the average value for this column
                        merged_df[weather_col].fillna(merged_df[avg_col], inplace=True)
                        logger.info(
                            f"Filled {empty_count} empty {weather_col} values with {avg_col}"
                        )
                    elif col in weather_df.columns:
                        # Use the average from weather_df directly
                        col_mean = weather_df[col].mean()
                        merged_df[weather_col].fillna(col_mean, inplace=True)
                        logger.info(
                            f"Filled {empty_count} empty {weather_col} values with mean {col_mean}"
                        )

            if empty_weather_count > 0:
                logger.info(
                    f"Filled a total of {empty_weather_count} empty weather values using fallback methods"
                )

            return merged_df

        except Exception as e:
            logger.error(f"Error merging lap and weather data: {str(e)}")
            return laps_df

    def extract_weather_features(self, session_obj) -> Dict[str, Any]:
        """
        Extract weather features from a session object.

        Args:
            session_obj: FastF1 session object

        Returns:
            Dictionary containing weather features
        """
        try:
            weather_features = {}

            # Check if weather data is available
            if session_obj.weather_data is not None:
                # Calculate average conditions
                weather_features["AirTemp_Avg"] = session_obj.weather_data[
                    "AirTemp"
                ].mean()
                weather_features["TrackTemp_Avg"] = session_obj.weather_data[
                    "TrackTemp"
                ].mean()
                weather_features["Humidity_Avg"] = session_obj.weather_data[
                    "Humidity"
                ].mean()
                weather_features["WindSpeed_Avg"] = session_obj.weather_data[
                    "WindSpeed"
                ].mean()

                # Calculate min/max values
                weather_features["AirTemp_Min"] = session_obj.weather_data[
                    "AirTemp"
                ].min()
                weather_features["AirTemp_Max"] = session_obj.weather_data[
                    "AirTemp"
                ].max()
                weather_features["TrackTemp_Min"] = session_obj.weather_data[
                    "TrackTemp"
                ].min()
                weather_features["TrackTemp_Max"] = session_obj.weather_data[
                    "TrackTemp"
                ].max()

                # Add rainfall indicator (0 or 1)
                weather_features["Rainfall"] = (
                    1 if session_obj.weather_data["Rainfall"].max() > 0 else 0
                )

                # Track condition categorization
                # (e.g., 0=dry, 1=slightly damp, 2=wet, 3=very wet)
                rainfall_max = session_obj.weather_data["Rainfall"].max()
                if rainfall_max == 0:
                    track_condition = 0
                elif rainfall_max < 0.5:
                    track_condition = 1
                elif rainfall_max < 2.0:
                    track_condition = 2
                else:
                    track_condition = 3

                weather_features["TrackCondition"] = track_condition

                # Weather variability (standard deviation)
                weather_features["AirTemp_Std"] = session_obj.weather_data[
                    "AirTemp"
                ].std()
                weather_features["TrackTemp_Std"] = session_obj.weather_data[
                    "TrackTemp"
                ].std()
                weather_features["Humidity_Std"] = session_obj.weather_data[
                    "Humidity"
                ].std()
                weather_features["WindSpeed_Std"] = session_obj.weather_data[
                    "WindSpeed"
                ].std()
            else:
                logger.warning(f"No weather data available for this session")
                weather_features = {
                    "AirTemp_Avg": None,
                    "TrackTemp_Avg": None,
                    "Humidity_Avg": None,
                    "WindSpeed_Avg": None,
                    "AirTemp_Min": None,
                    "AirTemp_Max": None,
                    "TrackTemp_Min": None,
                    "TrackTemp_Max": None,
                    "Rainfall": None,
                    "TrackCondition": None,
                    "AirTemp_Std": None,
                    "TrackTemp_Std": None,
                    "Humidity_Std": None,
                    "WindSpeed_Std": None,
                }

            return weather_features

        except Exception as e:
            logger.error(f"Error extracting weather features: {str(e)}")
            # Return empty dict with None values in case of error
            return {
                "AirTemp_Avg": None,
                "TrackTemp_Avg": None,
                "Humidity_Avg": None,
                "WindSpeed_Avg": None,
                "AirTemp_Min": None,
                "AirTemp_Max": None,
                "TrackTemp_Min": None,
                "TrackTemp_Max": None,
                "Rainfall": None,
                "TrackCondition": None,
                "AirTemp_Std": None,
                "TrackTemp_Std": None,
                "Humidity_Std": None,
                "WindSpeed_Std": None,
            }

    def extract_track_status_data(self, session_obj) -> pd.DataFrame:
        """
        Extract track status data from a session.

        Args:
            session_obj: FastF1 session object

        Returns:
            DataFrame containing track status information
        """
        try:
            if session_obj.track_status is not None:
                track_status = session_obj.track_status

                # Log track status information for debugging
                status_count = len(track_status)
                unique_statuses = (
                    track_status["Status"].unique()
                    if "Status" in track_status.columns and status_count > 0
                    else []
                )

                logger.info(
                    f"Extracted {status_count} track status records with statuses: {unique_statuses}"
                )

                # Define a mapping to convert numerical codes to descriptive text
                status_text_map = {
                    "1": "GREEN",  # Track clear
                    "2": "YELLOW",  # Yellow flag
                    "3": "YELLOW_SECTOR",  # Sector specific yellow (if used)
                    "4": "SC",  # Safety Car
                    "5": "RED",  # Red Flag
                    "6": "VSC",  # Virtual Safety Car
                    "7": "VSC_ENDING",  # VSC ending
                }

                # Convert numeric status codes to text for easier interpretation
                if "Status" in track_status.columns and not track_status.empty:
                    # Make a copy to avoid modifying the original
                    track_status = track_status.copy()

                    # Add a column with the text representation
                    track_status["StatusText"] = track_status["Status"].apply(
                        lambda x: status_text_map.get(
                            str(int(x)) if isinstance(x, (int, float)) else x, x
                        )
                    )

                    logger.info(
                        f"Status code mapping: {dict(zip(track_status['Status'], track_status['StatusText']))}"
                    )

                # Check for empty DataFrame with columns
                if status_count == 0:
                    logger.warning("Track status DataFrame is empty")
                    # For qualifying sessions, try to extract at least the global status
                    if hasattr(session_obj, "name") and session_obj.name in [
                        "Qualifying",
                        "Sprint Qualifying",
                    ]:
                        logger.info(
                            "Creating minimal track status for qualifying session"
                        )
                        # Create minimal track status for qualifying
                        if hasattr(session_obj, "session_status"):
                            status = session_obj.session_status
                        else:
                            status = "1"  # Default to GREEN if unknown

                        # Create a DataFrame with a single global status
                        start_time = (
                            session_obj.laps["LapStartTime"].min()
                            if not session_obj.laps.empty
                            and "LapStartTime" in session_obj.laps.columns
                            else pd.Timedelta(0)
                        )

                        df = pd.DataFrame(
                            {
                                "Time": [start_time],
                                "Status": [status],
                                "StatusText": [
                                    status_text_map.get(
                                        str(int(status))
                                        if isinstance(status, (int, float))
                                        else status,
                                        status,
                                    )
                                ],
                            }
                        )

                        logger.info(f"Created minimal track status: {df.to_dict()}")
                        return df

                return track_status
            else:
                logger.warning("No track status data available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error extracting track status: {str(e)}")
            return pd.DataFrame()

    def extract_race_control_messages(self, session_obj) -> pd.DataFrame:
        """
        Extract race control messages from a session.

        Args:
            session_obj: FastF1 session object

        Returns:
            DataFrame containing race control messages
        """
        try:
            if session_obj.race_control_messages is not None:
                return session_obj.race_control_messages
            else:
                logger.warning("No race control messages available")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error extracting race control messages: {str(e)}")
            return pd.DataFrame()

    def merge_lap_track_status(
        self, laps_df: pd.DataFrame, track_status_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge lap data with track status information.

        Args:
            laps_df: DataFrame containing lap data
            track_status_df: DataFrame containing track status

        Returns:
            DataFrame with track status information for each lap
        """
        if track_status_df.empty:
            logger.warning("No track status data available for merging")
            return laps_df

        try:
            # Make a copy of the laps DataFrame
            merged_df = laps_df.copy()

            # Initialize track status columns
            merged_df["TrackStatus"] = None
            merged_df["TrackStatusRaw"] = None  # Store the raw status code
            merged_df["TrackStatusText"] = None  # Store the text representation

            # Check if StatusText column is available (from our preprocessing)
            has_status_text = "StatusText" in track_status_df.columns

            # Check if we have a single global status (common in qualifying)
            global_status = None
            global_status_text = None
            if len(track_status_df) == 1:
                global_status = track_status_df.iloc[0]["Status"]
                global_status_text = (
                    track_status_df.iloc[0]["StatusText"] if has_status_text else None
                )
                logger.info(
                    f"Found single global track status: {global_status} ({global_status_text if global_status_text else 'no text'})"
                )

            # Log all unique status values for debugging
            unique_status_values = (
                track_status_df["Status"].unique()
                if "Status" in track_status_df.columns
                else []
            )
            logger.info(f"Unique track status values in data: {unique_status_values}")

            # For each lap, determine the track status
            for idx, lap in merged_df.iterrows():
                try:
                    if "LapStartTime" in lap and "Time" in lap:
                        lap_start = lap["LapStartTime"]
                        lap_end = lap_start + lap["Time"]

                        # Find track status changes during the lap
                        relevant_status = track_status_df[
                            (track_status_df["Time"] >= lap_start)
                            & (track_status_df["Time"] <= lap_end)
                        ]

                        if not relevant_status.empty:
                            # Get the most severe status during the lap
                            # According to FastF1 documentation:
                            # '1': Track clear (GREEN)
                            # '2': Yellow flag
                            # '4': Safety Car
                            # '5': Red Flag
                            # '6': Virtual Safety Car
                            # '7': VSC Ending

                            # Priority mapping (higher number = higher priority)
                            status_priority = {
                                # Text representations
                                "SC": 5,
                                "VSC": 4,
                                "VSC_ENDING": 4,  # Still considered VSC
                                "YELLOW": 3,
                                "YELLOW_SECTOR": 3,  # Sector yellow
                                "RED": 2,
                                "GREEN": 1,
                                # Numerical codes (standard FastF1 format)
                                "1": 1,  # GREEN
                                "2": 3,  # YELLOW
                                "3": 3,  # YELLOW SECTOR (if used)
                                "4": 5,  # SC
                                "5": 2,  # RED
                                "6": 4,  # VSC
                                "7": 4,  # VSC Ending
                                # Integers (in case they're not converted to strings)
                                1: 1,
                                2: 3,
                                3: 3,
                                4: 5,
                                5: 2,
                                6: 4,
                                7: 4,
                            }

                            highest_priority = 0
                            highest_status = "GREEN"
                            highest_status_text = "GREEN"

                            for _, status_row in relevant_status.iterrows():
                                current_status = status_row["Status"]

                                # Get the text representation if available
                                if has_status_text:
                                    current_status_text = status_row["StatusText"]
                                else:
                                    # Convert numerical status to text
                                    if isinstance(current_status, (int, float)):
                                        current_status_str = str(int(current_status))
                                        current_status_text = {
                                            "1": "GREEN",
                                            "2": "YELLOW",
                                            "3": "YELLOW_SECTOR",
                                            "4": "SC",
                                            "5": "RED",
                                            "6": "VSC",
                                            "7": "VSC_ENDING",
                                        }.get(current_status_str, current_status_str)
                                    else:
                                        current_status_text = str(current_status)

                                # For priority calculation, standardize the status to string
                                if isinstance(current_status, (int, float)):
                                    current_status_key = int(current_status)
                                else:
                                    current_status_key = str(current_status)

                                current_priority = status_priority.get(
                                    current_status_key, 0
                                )

                                if current_priority > highest_priority:
                                    highest_priority = current_priority
                                    highest_status = current_status
                                    highest_status_text = current_status_text

                            # Store both raw and text representation
                            merged_df.at[idx, "TrackStatus"] = (
                                highest_status  # Original behavior
                            )
                            merged_df.at[idx, "TrackStatusRaw"] = highest_status
                            merged_df.at[idx, "TrackStatusText"] = highest_status_text

                            # Add flag indicators based on priority
                            merged_df.at[idx, "YellowFlag"] = (
                                1 if highest_priority == 3 else 0
                            )
                            merged_df.at[idx, "SafetyCar"] = (
                                1 if highest_priority == 5 else 0
                            )
                            merged_df.at[idx, "VirtualSafetyCar"] = (
                                1 if highest_priority == 4 else 0
                            )
                            merged_df.at[idx, "RedFlag"] = (
                                1 if highest_priority == 2 else 0
                            )

                        elif global_status is not None:
                            # Use global status if available (common in qualifying)
                            merged_df.at[idx, "TrackStatus"] = global_status
                            merged_df.at[idx, "TrackStatusRaw"] = global_status
                            merged_df.at[idx, "TrackStatusText"] = (
                                global_status_text
                                if global_status_text
                                else global_status
                            )

                            # Priority mapping as above
                            status_priority = {
                                # Text representations
                                "SC": 5,
                                "VSC": 4,
                                "VSC_ENDING": 4,
                                "YELLOW": 3,
                                "YELLOW_SECTOR": 3,
                                "RED": 2,
                                "GREEN": 1,
                                # Numerical codes
                                "1": 1,
                                "2": 3,
                                "3": 3,
                                "4": 5,
                                "5": 2,
                                "6": 4,
                                "7": 4,
                                # Integers
                                1: 1,
                                2: 3,
                                3: 3,
                                4: 5,
                                5: 2,
                                6: 4,
                                7: 4,
                            }

                            # For priority calculation, get appropriate key
                            if isinstance(global_status, (int, float)):
                                status_key = int(global_status)
                            else:
                                status_key = str(global_status)

                            priority = status_priority.get(status_key, 1)

                            merged_df.at[idx, "YellowFlag"] = 1 if priority == 3 else 0
                            merged_df.at[idx, "SafetyCar"] = 1 if priority == 5 else 0
                            merged_df.at[idx, "VirtualSafetyCar"] = (
                                1 if priority == 4 else 0
                            )
                            merged_df.at[idx, "RedFlag"] = 1 if priority == 2 else 0
                        else:
                            # Default to green if no status data for this lap
                            merged_df.at[idx, "TrackStatus"] = "GREEN"
                            merged_df.at[idx, "TrackStatusRaw"] = 1
                            merged_df.at[idx, "TrackStatusText"] = "GREEN"
                            merged_df.at[idx, "YellowFlag"] = 0
                            merged_df.at[idx, "SafetyCar"] = 0
                            merged_df.at[idx, "VirtualSafetyCar"] = 0
                            merged_df.at[idx, "RedFlag"] = 0
                except Exception as e:
                    logger.warning(
                        f"Error merging track status for lap {idx}: {str(e)}"
                    )

            # Count track status distribution after processing
            if "TrackStatus" in merged_df.columns:
                status_counts = merged_df["TrackStatus"].value_counts()
                status_text_counts = (
                    merged_df["TrackStatusText"].value_counts()
                    if "TrackStatusText" in merged_df.columns
                    else {}
                )

                logger.info(f"Track status distribution: {status_counts.to_dict()}")
                logger.info(
                    f"Track status text distribution: {status_text_counts.to_dict()}"
                )

                # Log flag distribution
                flag_counts = {
                    "YellowFlag": merged_df["YellowFlag"].sum(),
                    "SafetyCar": merged_df["SafetyCar"].sum(),
                    "VirtualSafetyCar": merged_df["VirtualSafetyCar"].sum(),
                    "RedFlag": merged_df["RedFlag"].sum(),
                }
                logger.info(f"Flag indicators distribution: {flag_counts}")

            return merged_df

        except Exception as e:
            logger.error(f"Error merging lap and track status data: {str(e)}")
            return laps_df

    def get_session_data(self, year: int, event: str, session: str) -> pd.DataFrame:
        """
        Fetch session data for a specific F1 event.

        Args:
            year: The year of the event
            event: The event name (e.g., 'Monaco')
            session: The session type (e.g., 'FP1', 'Q', 'R')

        Returns:
            DataFrame containing the session data
        """
        try:
            logger.info(f"Fetching data for {year} {event} {session}")

            def _load_session():
                # Get the session
                session_obj = fastf1.get_session(year, event, session)

                # Load the session data WITH weather data
                session_obj.load(telemetry=True, weather=True, laps=True, messages=True)

                # Get lap data
                laps = session_obj.laps

                # Add metadata
                laps["Event"] = event
                laps["Year"] = year
                laps["Session"] = session

                # Extract weather features (summary statistics)
                weather_features = self.extract_weather_features(session_obj)

                # Add weather summary features to each lap
                for feature, value in weather_features.items():
                    laps[feature] = value

                # Extract and merge per-lap weather data
                if session_obj.weather_data is not None:
                    weather_data = session_obj.weather_data.copy()
                    laps = self.merge_lap_weather_data(laps, weather_data)

                # Ensure all Weather_* columns are populated with at least the average values
                weather_cols = [
                    "AirTemp",
                    "Humidity",
                    "Pressure",
                    "Rainfall",
                    "TrackTemp",
                    "WindDirection",
                    "WindSpeed",
                ]

                # Map weather columns to their average/summary counterparts
                weather_to_avg_map = {
                    "AirTemp": "AirTemp_Avg",
                    "Humidity": "Humidity_Avg",
                    "Pressure": None,  # No direct average
                    "Rainfall": "Rainfall",  # The rainfall indicator (0 or 1)
                    "TrackTemp": "TrackTemp_Avg",
                    "WindDirection": None,  # No direct average
                    "WindSpeed": "WindSpeed_Avg",
                }

                # Create or fill any missing Weather_* columns with available averages
                for col in weather_cols:
                    weather_col = f"Weather_{col}"

                    # Create the column if it doesn't exist
                    if weather_col not in laps.columns:
                        laps[weather_col] = np.nan
                        logger.info(f"Created missing column {weather_col}")

                    # Check if the column is empty or has NaN values
                    if laps[weather_col].isna().all():
                        avg_col = weather_to_avg_map.get(col)
                        if (
                            avg_col
                            and avg_col in laps.columns
                            and not laps[avg_col].isna().all()
                        ):
                            # Fill with the average value for this column
                            laps[weather_col] = laps[avg_col]
                            logger.info(
                                f"Filled empty {weather_col} column with {avg_col} values"
                            )

                # Extract and merge track status data
                track_status = self.extract_track_status_data(session_obj)

                # Add session context to help with track status processing
                session_info = {
                    "session_type": session,
                    "is_race": session == "R",
                    "is_qualifying": session in ["Q", "SQ"],
                    "is_practice": session in ["FP1", "FP2", "FP3"],
                    "session_name": session_obj.name
                    if hasattr(session_obj, "name")
                    else session,
                }

                # Log session context for debugging
                logger.info(f"Processing track status for {session_info}")

                if not track_status.empty:
                    # Add metadata to track status
                    track_status["Event"] = event
                    track_status["Year"] = year
                    track_status["Session"] = session

                    # Merge track status with laps
                    laps = self.merge_lap_track_status(laps, track_status)
                else:
                    # If no track status data, initialize default values
                    logger.warning(
                        f"No track status data for {event} {session}, using defaults"
                    )
                    laps["TrackStatus"] = "GREEN"
                    laps["YellowFlag"] = 0
                    laps["SafetyCar"] = 0
                    laps["VirtualSafetyCar"] = 0
                    laps["RedFlag"] = 0

                # Extract race control messages
                race_control = self.extract_race_control_messages(session_obj)
                if not race_control.empty:
                    race_control["Event"] = event
                    race_control["Year"] = year
                    race_control["Session"] = session
                    # We don't merge this directly - it's stored separately

                # Get circuit coordinates for external weather API queries if needed
                if (
                    session_obj.get_circuit_info() is not None
                    and session_obj.get_circuit_info().corners is not None
                ):
                    try:
                        first_corner = session_obj.get_circuit_info().corners.iloc[0]
                        laps["Circuit_Lat"] = first_corner["X"]
                        laps["Circuit_Long"] = first_corner["Y"]
                    except Exception as e:
                        logger.warning(
                            f"Could not extract circuit coordinates: {str(e)}"
                        )
                        laps["Circuit_Lat"] = None
                        laps["Circuit_Long"] = None
                else:
                    laps["Circuit_Lat"] = None
                    laps["Circuit_Long"] = None

                # Add tire compound flags for easier filtering
                if "Compound" in laps.columns:
                    try:
                        # Add boolean flags for each tire type
                        laps["IsSoft"] = laps["Compound"] == "SOFT"
                        laps["IsMedium"] = laps["Compound"] == "MEDIUM"
                        laps["IsHard"] = laps["Compound"] == "HARD"
                        laps["IsIntermediate"] = laps["Compound"] == "INTERMEDIATE"
                        laps["IsWet"] = laps["Compound"] == "WET"

                        # Add tire hardness as numeric value (useful for modeling)
                        compound_hardness = {
                            "SOFT": 1,
                            "MEDIUM": 2,
                            "HARD": 3,
                            "INTERMEDIATE": 4,
                            "WET": 5,
                        }
                        laps["CompoundHardness"] = (
                            laps["Compound"].map(compound_hardness).fillna(0)
                        )
                    except Exception as e:
                        logger.warning(f"Error adding tire compound flags: {str(e)}")

                # Add pit stop flags
                if "PitInTime" in laps.columns:
                    try:
                        # Create a boolean flag for pit in/out
                        laps["HasPitIn"] = laps["PitInTime"].notnull() & (
                            laps["PitInTime"] != pd.Timedelta(0)
                        )
                        laps["HasPitOut"] = laps["PitOutTime"].notnull() & (
                            laps["PitOutTime"] != pd.Timedelta(0)
                        )
                    except Exception as e:
                        logger.warning(f"Error adding pit stop flags: {str(e)}")

                return laps

            return self._retry_with_backoff(_load_session)

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def get_season_data(
        self, year: int, sessions: List[str] = ["Q", "R"]
    ) -> pd.DataFrame:
        """
        Fetch data for an entire F1 season.

        Args:
            year: The year of the season
            sessions: Session types to include (default: qualifying and race only)

        Returns:
            DataFrame containing the season data
        """
        try:
            logger.info(f"Fetching data for {year} season")

            def _load_schedule():
                return fastf1.get_event_schedule(year)

            schedule = self._retry_with_backoff(_load_schedule)
            all_data = []

            for _, event in schedule.iterrows():
                try:
                    # Get specified session data for each event
                    for session in sessions:
                        try:
                            session_data = self.get_session_data(
                                year, event["EventName"], session
                            )
                            all_data.append(session_data)
                            logger.info(
                                f"Successfully fetched {year} {event['EventName']} {session} data"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not fetch data for {year} {event['EventName']} {session}: {str(e)}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Could not process event {event['EventName']}: {str(e)}"
                    )

            if not all_data:
                raise ValueError(f"No data could be fetched for the {year} season")

            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(
                f"Combined {len(combined_data)} total lap records for {year} season"
            )
            return combined_data

        except Exception as e:
            logger.error(f"Error fetching season data: {str(e)}")
            raise

    def save_race_control_messages(
        self, year: int, event: str, session: str, output_dir: str
    ) -> str:
        """
        Save race control messages for a specific session to a CSV file.

        Args:
            year: The year of the event
            event: The event name (e.g., 'Monaco')
            session: The session type (e.g., 'FP1', 'Q', 'R')
            output_dir: Directory to save the CSV file

        Returns:
            Path to the saved CSV file, or None if no messages available
        """
        try:
            # Get the session
            session_obj = fastf1.get_session(year, event, session)
            session_obj.load()

            # Extract race control messages
            race_control = self.extract_race_control_messages(session_obj)

            if race_control.empty:
                logger.warning(
                    f"No race control messages available for {year} {event} {session}"
                )
                return None

            # Add metadata
            race_control["Event"] = event
            race_control["Year"] = year
            race_control["Session"] = session

            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            filename = f"race_control_{year}_{event.replace(' ', '_')}_{session}.csv"
            output_file = output_path / filename
            race_control.to_csv(output_file, index=False)

            logger.info(
                f"Saved {len(race_control)} race control messages to {output_file}"
            )
            return str(output_file)

        except Exception as e:
            logger.error(f"Error saving race control messages: {str(e)}")
            return None

    def save_track_status_data(
        self, year: int, event: str, session: str, output_dir: str
    ) -> str:
        """
        Save track status data for a specific session to a CSV file.

        Args:
            year: The year of the event
            event: The event name (e.g., 'Monaco')
            session: The session type (e.g., 'FP1', 'Q', 'R')
            output_dir: Directory to save the CSV file

        Returns:
            Path to the saved CSV file, or None if no track status data available
        """
        try:
            # Get the session
            session_obj = fastf1.get_session(year, event, session)
            session_obj.load()

            # Extract track status data
            track_status = self.extract_track_status_data(session_obj)

            if track_status.empty:
                logger.warning(
                    f"No track status data available for {year} {event} {session}"
                )
                return None

            # Add metadata
            track_status["Event"] = event
            track_status["Year"] = year
            track_status["Session"] = session

            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            filename = f"track_status_{year}_{event.replace(' ', '_')}_{session}.csv"
            output_file = output_path / filename
            track_status.to_csv(output_file, index=False)

            logger.info(
                f"Saved {len(track_status)} track status entries to {output_file}"
            )
            return str(output_file)

        except Exception as e:
            logger.error(f"Error saving track status data: {str(e)}")
            return None

    def merge_race_control_by_event(
        self, year: int, event: str, sessions: List[str], output_dir: str
    ) -> str:
        """
        Merge race control messages from all sessions of an event into a single file.

        Args:
            year: The year of the event
            event: The event name (e.g., 'Monaco')
            sessions: List of session types to merge (e.g., ["FP1", "FP2", "FP3", "Q", "R"])
            output_dir: Directory to save the merged CSV file

        Returns:
            Path to the saved CSV file, or None if no messages available
        """
        try:
            all_messages = []

            # Collect race control messages from each session
            for session_type in sessions:
                try:
                    # Get the session
                    session_obj = fastf1.get_session(year, event, session_type)
                    session_obj.load(messages=True)

                    # Extract race control messages
                    race_control = self.extract_race_control_messages(session_obj)

                    if not race_control.empty:
                        # Add metadata
                        race_control["Event"] = event
                        race_control["Year"] = year
                        race_control["Session"] = session_type
                        all_messages.append(race_control)
                        logger.info(
                            f"Found {len(race_control)} messages for {session_type}"
                        )
                    else:
                        logger.info(
                            f"No race control messages for {event} {session_type}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error extracting messages for {session_type}: {str(e)}"
                    )

            # If no messages found, return None
            if not all_messages:
                logger.warning(f"No race control messages found for {year} {event}")
                return None

            # Combine all messages and sort by time
            combined_messages = pd.concat(all_messages, ignore_index=True)

            # Sort by session order and time if possible
            session_order = {"FP1": 1, "FP2": 2, "FP3": 3, "Q": 4, "SQ": 5, "R": 6}
            combined_messages["SessionOrder"] = combined_messages["Session"].map(
                lambda x: session_order.get(x, 99)
            )

            # Sort first by session order, then by time (if date/timestamp column exists)
            sort_cols = ["SessionOrder"]
            time_cols = [
                col
                for col in combined_messages.columns
                if "Time" in col or "Date" in col
            ]
            if time_cols:
                sort_cols.extend(time_cols)

            combined_messages = combined_messages.sort_values(by=sort_cols)

            # Remove the helper column
            if "SessionOrder" in combined_messages.columns:
                combined_messages = combined_messages.drop(columns=["SessionOrder"])

            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            event_clean = event.replace(" ", "_")
            filename = f"race_control_{year}_{event_clean}_ALL_SESSIONS.csv"
            output_file = output_path / filename
            combined_messages.to_csv(output_file, index=False)

            logger.info(
                f"Saved {len(combined_messages)} combined race control messages to {output_file}"
            )
            return str(output_file)

        except Exception as e:
            logger.error(f"Error merging race control messages for {event}: {str(e)}")
            return None
