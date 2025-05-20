# Enhanced F1 Data Collection

This document describes the enhanced F1 data collection capabilities in our F1 Predictor system, specifically focusing on weather data, track status information, and race control messages.

## Overview

The updated FastF1 collector now includes detailed per-lap weather data instead of just average weather conditions, as well as track status information and race control messages. This enhancement provides a more comprehensive dataset for analysis and prediction.

## New Features

### 1. Per-Lap Weather Data

Instead of using session-average weather conditions, the collector now matches each lap with the weather conditions at the time the lap was completed. This includes:

- Air temperature
- Track temperature
- Humidity
- Wind speed
- Wind direction
- Rainfall
- Pressure

Weather data is prefixed with `Weather_` in the resulting DataFrame.

### 2. Track Status Information

Track status changes during a session (yellow flags, safety cars, red flags) are now included and matched to each lap. The status is represented by a code:

- `1`: Track clear
- `2`: Yellow flag
- `4`: Safety Car
- `5`: Red Flag
- `6`: Virtual Safety Car

This information is stored in the `TrackStatus` column of the DataFrame.

### 3. Race Control Messages

Race control messages (which include announcements about flags, penalties, and other official communications) are now collected and can be saved to separate CSV files. These can provide additional context for model training and analysis.

## How to Use

### Collecting Data with Enhanced Features

Use the updated collection scripts to automatically gather and process this enhanced data:

```bash
# Collect data for a specific season with enhanced weather data
python scripts/data_collection/collect_and_process_data.py --years 2023 --sessions Q R --save-individual

# Collect data for multiple seasons
python scripts/data_collection/collect_and_process_data.py --years 2021 2022 2023 --sessions Q R --save-individual

# Collect data for all available seasons in a range
python scripts/data_collection/collect_all_seasons.py --start-year 2018 --end-year 2023 --save-individual
```

### Testing Weather Data Collection

You can run the test script to see how the enhanced data collection works:

```bash
python scripts/data_collection/test_weather_data.py
```

This will:
1. Collect data for a specific race (2023 Monaco Grand Prix by default)
2. Display weather trends throughout the race
3. Analyze the effect of track status on lap times
4. Show race control messages

### Programmatic Usage

```python
from src.data.fastf1_collector import F1DataCollector

# Initialize the collector
collector = F1DataCollector(cache_dir="f1_cache")

# Get session data with enhanced weather and track status
session_data = collector.get_session_data(2023, "Monaco", "R")

# Access different data components
lap_data = session_data['lap_data']
weather_data = session_data['weather_data']
track_status_data = session_data['track_status_data']
race_control_messages = session_data['race_control_messages']

# Save race control messages to a file
collector.save_race_control_messages(2023, "Monaco", "R", "data/race_control")

# Save track status data to a file
collector.save_track_status_data(2023, "Monaco", "R", "data/track_status")
```

## Data Structure

### Weather Data Columns

Weather data is merged with lap data and prefixed with `Weather_`:

- `Weather_AirTemp`: Air temperature in °C
- `Weather_Humidity`: Relative humidity as a percentage
- `Weather_Pressure`: Atmospheric pressure in millibars
- `Weather_Rainfall`: Rainfall indicator (0 for no rain, 1 for rain)
- `Weather_TrackTemp`: Track temperature in °C
- `Weather_WindDirection`: Wind direction in degrees
- `Weather_WindSpeed`: Wind speed in km/h

### Track Status Data

Each lap is linked with the track status at the time of the lap:

- `TrackStatus`: Numeric code representing the track status (1 = clear, 2 = yellow, etc.)
- `TrackStatusMessage`: Text description of the track status (when available)

### Race Control Messages

Race control messages are provided as a separate DataFrame with columns:

- `Time`: Timestamp of the message
- `Category`: Category of the message (e.g., "Flag", "Stewards")
- `Message`: The content of the message
- `Flag`: Flag type if applicable
- `Scope`: Scope of the message (e.g., "Track", "Driver")
- `Sector`: Sector number if applicable
- `RacingNumber`: Driver number if applicable

## Implementation Details

The enhanced data collection is implemented through three main methods:

1. `merge_lap_weather_data`: Merges lap data with weather data based on timestamps
2. `merge_lap_track_status`: Merges lap data with track status based on timestamps
3. `get_session_data`: Retrieves and combines all data types for a session

For more detailed implementation information, see the comments in the `fastf1_collector.py` file.

## Limitations

- Weather data is only available for sessions where FastF1 provides this information (generally most recent seasons)
- The granularity of weather data is limited by the frequency of updates provided by the official F1 timing system
- Track status information may not capture very brief status changes that occur and end between data points

## Future Improvements

Potential future enhancements to the data collection system include:

1. Adding DRS activation zone information for each lap
2. Including more detailed tire information and degradation metrics
3. Correlating driver messages with race control messages and track status
4. Adding video/image metadata linking for visual analysis 