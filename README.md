# F1 Race Strategy Simulator

A sophisticated multi-agent simulation platform for Formula 1 race strategy analysis and prediction. This project combines real-world F1 data, machine learning models, and specialized AI agents to simulate complex race scenarios, model tire degradation, predict lap times, adapt to weather changes, and optimize pit stop strategies.

## Project Overview

This system provides a realistic simulation of F1 racing, focusing on the strategic aspects managed by a race engineer. It utilizes historical data and machine learning to power its predictive capabilities and employs a suite of intelligent agents to make dynamic decisions throughout a simulated race.

### Key Features

- **Multi-Agent Simulation**: A core team of specialized agents work in concert:
    - **Tire Manager Agent**: Tracks tire wear, temperature, and grip.
    - **Strategy Agent**: Makes decisions on pit stops and tire compound changes.
    - **Lap Time Agent**: Predicts lap times considering car, driver, and track conditions.
    - **Weather Agent**: Integrates real-time or simulated weather data and its impact.
    - **Vehicle Dynamics Agent**: Models car performance characteristics.
    - **Gap Effects Agent**: Simulates effects like DRS and dirty air.
- **Data-Driven Models**: Leverages historical F1 data (primarily 2023-2024 seasons via FastF1 API) to train machine learning models for:
    - Lap Time Prediction
    - Tire Wear and Degradation
- **Comprehensive Data Pipeline**: Robust system for collecting, processing, and validating F1 data.
- **Interactive Web Interface**: A Streamlit application allows users to configure race scenarios, run simulations, and visualize results in real-time.
- **Modular and Extensible**: Designed with a clear separation of concerns, allowing for easier updates and addition of new features or agents.
- **Experiment Tracking**: Integrated with MLflow for tracking machine learning experiments and model versions.
- **Deployment Ready**: Includes Docker configuration for easy deployment.

## Directory Structure

```
F1/
├── .github/            # GitHub Actions workflows
├── catboost_info/      # CatBoost training artifacts
├── data/               # Data storage
│   ├── cache/          # FastF1 and other data caches
│   ├── categories/     # Reference data (drivers, teams, events, characteristics)
│   ├── processed/      # Processed data files for modeling
│   ├── raw/            # Raw downloaded data
│   └── validation/     # Data for validation purposes
├── docs/               # Project documentation
├── logs/               # Log files
├── mlruns/             # MLflow experiment tracking data
├── models/             # Saved trained machine learning models (e.g., .joblib files)
├── notebooks/          # Jupyter notebooks for EDA, prototyping, and analysis
├── reports/            # Generated reports (e.g., simulation results, analysis summaries)
├── scripts/            # Utility and automation scripts
├── src/                # Core source code
│   ├── agents/         # Agent implementations
│   │   ├── base_agent.py
│   │   ├── gap_effects_agent.py
│   │   ├── grip_model_agent.py
│   │   ├── lap_time_agent.py
│   │   ├── strategy_agent.py
│   │   ├── tire_manager_agent.py
│   │   ├── tire_temperature_agent.py
│   │   ├── tire_wear_agent.py
│   │   └── weather_agent.py
│   ├── api/            # FastAPI backend (if applicable for broader API exposure)
│   ├── config/         # Configuration files (settings.py, simulation_config.py)
│   ├── data_pipeline/  # Data collection, processing, and validation modules
│   ├── models/         # Interfaces or wrappers for ML models used in simulation
│   ├── orchestrator/   # Simulation coordination and main loop
│   │   └── simulation_loop.py
│   └── utils/          # Utility functions and helper classes
├── web/                # Web interface (Streamlit application)
│   ├── pages/          # Additional pages for the Streamlit app
│   ├── streamlit_app.py # Main Streamlit application
│   └── app.py          # Potentially a supporting FastAPI app or utility for web
├── .gitignore          # Specifies intentionally untracked files
├── Dockerfile          # Docker configuration for containerization
├── docker-compose.yml  # Docker Compose for multi-container setups (e.g., app + MLflow)
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker (recommended for containerized usage)
- An OpenWeather API key (if using live weather features, though the current `WeatherAgent` seems to primarily use pre-set or simulated conditions)

### Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd F1 # Or your chosen project directory name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    The system might use environment variables for certain configurations (e.g., API keys). Refer to `src/config/settings.py` and potentially a `.env.example` file if provided. If an OpenWeather API key is needed for `WeatherAgent`, it would typically be set as an environment variable.
    Create a `.env` file in the root directory if needed:
    ```
    # Example:
    # OPENWEATHER_API_KEY=your_openweather_api_key_here
    ```
    (Note: The current `streamlit_app.py` has a hardcoded API key for weather, which is not best practice. This should ideally be moved to an environment variable or config.)


### Docker Setup

1.  **Build the Docker image:**
    ```bash
    docker build -t f1-strategy-simulator .
    ```

2.  **Run the Docker container:**
    The `Dockerfile` exposes port 8501 for Streamlit and 5000 (presumably for MLflow if run within the same compose setup or if the `MLFLOW_TRACKING_URI` is set to a local service).
    ```bash
    # To run Streamlit app
    docker run -p 8501:8501 f1-strategy-simulator
    ```
    If using `docker-compose.yml` (which might include an MLflow service):
    ```bash
    docker-compose up
    ```

## Usage

### Data Collection and Preparation
The system uses the `FastF1` library. Data collection might be part_of initial setup scripts or notebooks found in `scripts/` or `notebooks/`. Processed data is stored in `data/processed/`.

### Training Models
Machine learning models for lap times and tire wear are crucial. Training scripts might be located in `scripts/` or `notebooks/`. Trained models are stored in the `models/` directory (root level).
-   **MLflow:** Experiments are tracked using MLflow. The tracking URI is configured in `src/config/settings.py`. To view the MLflow UI, you might need to run `mlflow ui` in your terminal, ensuring it points to the `mlruns` directory.

### Running a Simulation
The primary way to interact with the simulation is through the Streamlit web interface:
```bash
streamlit run web/streamlit_app.py
```
Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Core Components Deep Dive

### 1. Agents (`src/agents/`)
-   **`BaseAgent`**: Defines the common interface for all agents (`process`, `update_state`, `get_state`, `reset`).
-   **`TireManagerAgent`**: Central to tire strategy. It uses sub-agents:
    -   `TireWearAgent`: Predicts tire wear percentage based on various factors, potentially using a trained ML model.
    -   `GripModelAgent`: Calculates current tire grip based on wear, compound, and conditions.
    -   `TireTemperatureAgent`: Models tire temperature and its effect on performance.
    It determines tire health, optimal operating windows, and informs pit stop decisions.
-   **`StrategyAgent`**: Makes high-level strategic decisions, primarily:
    -   When to pit.
    -   Which tire compound to switch to.
    Considers current race state, tire condition, weather, and remaining laps.
-   **`LapTimeAgent`**: Predicts lap times. This likely uses an ML model considering driver skill, car performance, tire state, and track conditions.
-   **`WeatherAgent`**: Manages weather information. It can fetch live data (though current implementation might use presets) and provides weather forecasts to other agents.
-   **`VehicleDynamicsAgent`**: Models the car's physical performance, likely providing baseline lap times adjusted by other factors.
-   **`GapEffectsAgent`**: Simulates the impact of racing in close proximity to other cars (dirty air, slipstream/DRS).

### 2. Orchestrator (`src/orchestrator/simulation_loop.py`)
-   **`RaceSimulator`**: The main engine.
    -   Initializes the race state, drivers, and agents.
    -   Runs the simulation lap by lap.
    -   In each lap:
        1.  Updates weather.
        2.  For each driver:
            -   The `StrategyAgent` makes pit decisions.
            -   The `TireManagerAgent` updates tire state (wear, temp, grip).
            -   The `VehicleDynamicsAgent` and `GapEffectsAgent` contribute to lap time calculation.
            -   The `LapTimeAgent` predicts the final lap time.
        3.  Updates driver positions and overall race state.
    -   Collects race history and provides results.

### 3. Data Pipeline (`src/data_pipeline/`)
-   **`F1DataCollector`**: Uses `fastf1` to fetch historical race data (laps, weather, timing, telemetry).
-   **`DataProcessor`**: Cleans, transforms, and enriches the raw data.
-   **`Preprocessor`**: Prepares data specifically for training ML models.
-   **`Validator`**: Ensures data quality and integrity.

### 4. Configuration (`src/config/`)
-   **`settings.py`**: Defines global constants, file paths (data, models, MLflow URI), API settings, and default simulation parameters.
-   **`simulation_config.py`**: Likely provides functions or classes to generate specific configurations for a simulation run (e.g., selecting circuit, drivers, initial conditions).

## Contributing

Contributions are welcome! Please follow these general guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes, ensuring code is well-formatted and commented where necessary.
4.  Write or update tests for your changes.
5.  Commit your changes (`git commit -m 'Add some feature'`).
6.  Push to your branch (`git push origin feature/your-feature-name`).
7.  Open a Pull Request against the main repository.



## Credits

-   **FastF1 API**: For providing access to comprehensive F1 historical data.
-   **OpenWeather API**: (If actively used) For real-time weather data.
-   The developers and contributors to the various open-source libraries used in this project.

---

