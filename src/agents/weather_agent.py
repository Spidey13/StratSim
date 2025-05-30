"""
Weather adaptation agent that processes weather data to adjust strategy.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
import json
import requests
from datetime import datetime, timedelta

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WeatherAgent(BaseAgent):
    """
    Agent responsible for processing weather data and forecasting
    changes that might affect race strategy.
    """

    def __init__(self, name: str = "WeatherAgent", api_key: Optional[str] = None):
        """
        Initialize the weather agent.

        Args:
            name: Agent name
            api_key: Optional OpenWeather API key for live data
        """
        super().__init__(name)
        self.api_key = api_key
        self.initial_weather = None  # Store initial weather data

        # Weather condition mapping
        self.condition_map = {"Dry": 0, "Light Rain": 1, "Heavy Rain": 3, "Variable": 2}

        # Weather history for tracking changes
        self.weather_history = []

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process current weather data and forecast for a lap.

        Args:
            inputs: Dictionary containing:
                - lap: Current lap number
                - weather: Weather data
                - circuit_id: Circuit identifier

        Returns:
            Dictionary containing:
                - weather: Updated weather state for this lap
                - trends: Weather trend information
                - alerts: Any weather alerts or significant changes
        """
        lap = inputs.get("lap", 0)
        current_weather_data_source = inputs.get("weather", {})
        current_weather = current_weather_data_source
        forecast = current_weather_data_source.get("forecast", [])

        # Store weather history
        if current_weather:
            self.weather_history.append({"lap": lap, "weather": current_weather})

        # Get real-time weather only once at the start of simulation
        if (
            self.api_key
            and "circuit_id" in inputs
            and lap == 1
            and not self.initial_weather
        ):
            logger.info(f"Initializing live weather data for {inputs['circuit_id']}")
            try:
                self.initial_weather = self._get_live_weather(inputs["circuit_id"])
                if self.initial_weather:
                    logger.info("Successfully initialized live weather data")
                    updated_weather = self._merge_weather_data(
                        current_weather, self.initial_weather
                    )
                else:
                    logger.warning(
                        "Failed to get live weather, using default conditions"
                    )
                    updated_weather = current_weather
            except Exception as e:
                logger.error(f"Failed to get initial live weather: {str(e)}")
                updated_weather = current_weather
        else:
            # Use stored weather data for subsequent laps
            if self.initial_weather and lap > 1:
                updated_weather = self._merge_weather_data(
                    current_weather, self.initial_weather
                )
            else:
                updated_weather = self._get_forecast_for_lap(
                    lap, current_weather, forecast
                )

        # Calculate weather trends
        trends = self._calculate_trends()

        # Generate weather alerts
        alerts = self._generate_alerts(updated_weather, trends)

        return {"weather": updated_weather, "trends": trends, "alerts": alerts}

    def _get_live_weather(self, circuit_id: str) -> Dict[str, Any]:
        """
        Get live weather data from API for a circuit.

        Args:
            circuit_id: Circuit identifier

        Returns:
            Weather data dictionary
        """
        try:
            # Load circuit coordinates from json file
            with open("data/categories/event_location.json", "r") as f:
                circuit_coords_data = json.load(f)

            # Format circuit_id to match json keys
            if not circuit_id.endswith("Grand Prix"):
                circuit_id = f"{circuit_id} Grand Prix"

            # Get coordinates for the circuit
            if circuit_id in circuit_coords_data:
                lat, lon = circuit_coords_data[circuit_id]
                logger.info(
                    f"Fetching weather for {circuit_id} at coordinates: {lat}, {lon}"
                )

                # Make API call to OpenWeather
                base_url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric",  # Get temperature in Celsius
                }

                response = requests.get(base_url, params=params)
                response.raise_for_status()
                weather_data = response.json()

                # Map weather condition to our format
                weather_main = weather_data.get("weather", [{}])[0].get("main", "Clear")
                if weather_main in ["Thunderstorm", "Heavy Rain"]:
                    condition = "Heavy Rain"
                elif weather_main in ["Rain", "Drizzle"]:
                    condition = "Light Rain"
                else:
                    condition = "Dry"

                # Return in our existing format
                weather_result = {
                    "condition": condition,
                    "rainfall": self.condition_map[condition],
                    "air_temp": weather_data.get("main", {}).get("temp", 25),
                    "track_temp": weather_data.get("main", {}).get("temp", 25) + 10,
                    "humidity": weather_data.get("main", {}).get("humidity", 50),
                    "wind_speed": weather_data.get("wind", {}).get("speed", 5),
                    "timestamp": datetime.now().isoformat(),
                }
                logger.info(
                    f"Initial weather conditions: {condition}, {weather_result['air_temp']}°C"
                )
                return weather_result
            else:
                logger.warning(f"Circuit {circuit_id} not found in coordinates data")

        except Exception as e:
            logger.error(f"Failed to get live weather: {str(e)}")

        # Return default values if anything fails
        default_weather = {
            "condition": "Dry",
            "rainfall": 0,
            "air_temp": 25,
            "track_temp": 35,
            "humidity": 50,
            "wind_speed": 8,
            "timestamp": datetime.now().isoformat(),
        }
        return default_weather

    def _merge_weather_data(
        self, current: Dict[str, Any], live: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge current weather state with live data.

        Args:
            current: Current weather state
            live: Live weather data

        Returns:
            Merged weather data
        """
        result = current.copy()

        # Update with live data
        result.update(
            {
                "condition": live.get("condition", result.get("condition")),
                "rainfall": live.get("rainfall", result.get("rainfall")),
                "air_temp": live.get("air_temp", result.get("air_temp")),
                "track_temp": live.get("track_temp", result.get("track_temp")),
                "humidity": live.get("humidity", result.get("humidity")),
                "wind_speed": live.get("wind_speed", result.get("wind_speed")),
            }
        )

        # Keep the original forecast
        if "forecast" in current:
            result["forecast"] = current["forecast"]

        return result

    def _get_forecast_for_lap(
        self, lap: int, current_weather: Dict[str, Any], forecast: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get weather forecast for a specific lap.

        Args:
            lap: Lap number
            current_weather: Current weather state
            forecast: List of forecast data points

        Returns:
            Weather data for the specified lap
        """
        # If no forecast or empty forecast, return current weather
        if not forecast:
            return current_weather

        # Find forecast for this lap
        lap_forecast = next((f for f in forecast if f.get("lap") == lap), None)

        if not lap_forecast:
            # No specific forecast for this lap, use current
            return current_weather

        # Create updated weather
        updated_weather = current_weather.copy()

        # Update with forecast data
        condition = lap_forecast.get("condition")
        if condition:
            updated_weather["condition"] = condition

            # Update rainfall based on condition
            if condition == "Dry":
                updated_weather["rainfall"] = 0
            elif condition == "Light Rain":
                updated_weather["rainfall"] = 1
            elif condition == "Heavy Rain":
                updated_weather["rainfall"] = 3

            # Adjust temperatures based on condition
            if condition != current_weather.get("condition"):
                if condition == "Dry":
                    updated_weather["air_temp"] = min(
                        30, updated_weather.get("air_temp", 25) + 2
                    )
                    updated_weather["track_temp"] = min(
                        45, updated_weather.get("track_temp", 35) + 3
                    )
                elif condition == "Light Rain":
                    updated_weather["air_temp"] = max(
                        15, updated_weather.get("air_temp", 25) - 3
                    )
                    updated_weather["track_temp"] = max(
                        20, updated_weather.get("track_temp", 35) - 5
                    )
                elif condition == "Heavy Rain":
                    updated_weather["air_temp"] = max(
                        10, updated_weather.get("air_temp", 25) - 5
                    )
                    updated_weather["track_temp"] = max(
                        15, updated_weather.get("track_temp", 35) - 8
                    )

        # Keep the original forecast in the result
        updated_weather["forecast"] = current_weather.get("forecast", forecast)

        return updated_weather

    def _calculate_trends(self) -> Dict[str, Any]:
        """
        Calculate weather trends from history.

        Returns:
            Dictionary of weather trends
        """
        # Need at least 3 data points for meaningful trends
        if len(self.weather_history) < 3:
            return {
                "temperature_trend": "stable",
                "rainfall_trend": "stable",
                "condition_trend": "stable",
            }

        # Get last 5 points (or all if fewer)
        recent_history = self.weather_history[-5:]

        # Extract temp and rainfall values
        temps = [h["weather"].get("air_temp", 0) for h in recent_history]
        rainfalls = [h["weather"].get("rainfall", 0) for h in recent_history]
        conditions = [h["weather"].get("condition", "Dry") for h in recent_history]

        # Calculate trends
        temp_diff = temps[-1] - temps[0]
        rain_diff = rainfalls[-1] - rainfalls[0]

        # Temperature trend
        if temp_diff > 2:
            temp_trend = "rising"
        elif temp_diff < -2:
            temp_trend = "falling"
        else:
            temp_trend = "stable"

        # Rainfall trend
        if rain_diff > 0.5:
            rain_trend = "increasing"
        elif rain_diff < -0.5:
            rain_trend = "decreasing"
        else:
            rain_trend = "stable"

        # Condition trend
        if conditions[0] != conditions[-1]:
            if self.condition_map.get(conditions[-1], 0) > self.condition_map.get(
                conditions[0], 0
            ):
                condition_trend = "worsening"
            else:
                condition_trend = "improving"
        else:
            condition_trend = "stable"

        return {
            "temperature_trend": temp_trend,
            "rainfall_trend": rain_trend,
            "condition_trend": condition_trend,
        }

    def _generate_alerts(
        self, weather: Dict[str, Any], trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate weather alerts based on current weather and trends.

        Args:
            weather: Current weather data
            trends: Weather trends

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Rain starting alert
        if trends["rainfall_trend"] == "increasing" and weather.get("rainfall", 0) > 0:
            alerts.append(
                {
                    "type": "rain_starting",
                    "severity": "high",
                    "message": "Rain intensity increasing. Consider pit strategy.",
                }
            )

        # Rain stopping alert
        if trends["rainfall_trend"] == "decreasing" and weather.get("rainfall", 0) > 0:
            alerts.append(
                {
                    "type": "rain_stopping",
                    "severity": "medium",
                    "message": "Rain intensity decreasing. Track drying expected.",
                }
            )

        # Temperature alerts
        if (
            trends["temperature_trend"] == "rising"
            and weather.get("track_temp", 30) > 40
        ):
            alerts.append(
                {
                    "type": "high_temp",
                    "severity": "medium",
                    "message": "Track temperature rising. Increased tire degradation expected.",
                }
            )

        if (
            trends["temperature_trend"] == "falling"
            and weather.get("track_temp", 30) < 20
        ):
            alerts.append(
                {
                    "type": "low_temp",
                    "severity": "medium",
                    "message": "Track temperature falling. Reduced grip expected.",
                }
            )

        # Check forecasts for incoming weather changes
        upcoming_condition_change = self._check_upcoming_condition_change(weather)
        if upcoming_condition_change:
            alerts.append(upcoming_condition_change)

        return alerts

    def _check_upcoming_condition_change(
        self, weather: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if there are upcoming significant weather changes in forecast.

        Args:
            weather: Current weather data

        Returns:
            Alert dictionary if change detected, None otherwise
        """
        forecast = weather.get("forecast", [])
        if not forecast:
            return None

        current_condition = weather.get("condition", "Dry")

        # Look ahead up to 5 laps
        for i in range(1, 6):
            # Find forecast for current_lap + i
            if len(self.weather_history) > 0:
                current_lap = self.weather_history[-1]["lap"]
                upcoming_forecast = next(
                    (f for f in forecast if f.get("lap") == current_lap + i), None
                )

                if (
                    upcoming_forecast
                    and upcoming_forecast.get("condition") != current_condition
                ):
                    return {
                        "type": "condition_change",
                        "severity": "high",
                        "message": f"Weather changing to {upcoming_forecast.get('condition')} in {i} laps",
                        "lap_delta": i,
                    }

        return None
