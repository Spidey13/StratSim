# F1 Race Strategy Simulator

A hybrid machine learning and agentic AI system for simulating Formula 1 race strategy decisions. This project combines data-driven models with multiple specialized AI agents to mimic the decision-making of a race strategist.

## Project Overview

This system predicts lap times, models tire wear, adapts to weather changes, and plans optimal pit stop strategies under various conditions. The goal is to create a realistic simulation of race strategy planning that could be used for both educational purposes and as a decision support tool.

### Key Features

- **Multi-Agent Simulation**: Four specialized agents collaborate in a simulation loop:
  - Lap Time Prediction agent: Forecasts lap times based on various factors
  - Tire Wear Modeling agent: Estimates tire degradation and performance
  - Weather Adaptation agent: Processes weather data to adjust strategy
  - Strategy Planning agent: Makes pit stop and tire compound decisions

- **Data-Driven Models**: The system leverages historical F1 data from the FastF1 library (2023-2024 seasons) to train machine learning models for lap time and tire wear prediction. OpenWeather API supplies live weather data for real-time strategy adjustments.

- **Deployment-Ready Engineering**: This project includes Docker setup, MLflow for experiment tracking, and comprehensive documentation.

- **Interactive Demo Interface**: A Streamlit web UI where users can input scenario settings and watch the agents' strategy unfold.

## Directory Structure

```
f1-predictor/
├── data/               # Data storage
│   ├── cache/         # FastF1 cache
│   ├── processed/     # Processed data files
│   └── raw/           # Raw data downloads
├── src/               # Source code
│   ├── agents/        # Agent implementations
│   │   ├── base_agent.py        # Base agent interface
│   │   ├── lap_time_agent.py    # Lap time prediction
│   │   ├── tire_wear_agent.py   # Tire degradation modeling
│   │   ├── weather_agent.py     # Weather integration
│   │   └── strategy_agent.py    # Strategy planning
│   ├── orchestrator/  # Simulation coordination
│   │   ├── simulation_loop.py   # Main simulation loop
│   │   ├── planner.py           # Strategy planning
│   │   └── agent_controller.py  # Agent state management
│   ├── data/          # Data handling modules
│   ├── models/        # Model definitions
│   ├── api/           # API integrations
│   └── utils/         # Utility functions
├── models/            # Saved trained models
├── notebooks/         # Jupyter notebooks for EDA and prototyping
├── scripts/           # Utility scripts
│   ├── data_collection/   # Data collection scripts
│   ├── data_processing/   # Data preprocessing scripts
│   ├── model_training/    # Model training scripts
│   │   ├── train_and_log.py    # Train models with MLflow
│   ├── analysis/          # Analysis scripts
│   └── utils/             # Utility scripts
│       └── run_mlflow_ui.sh    # Launch MLflow dashboard
├── web/               # Web interface
│   └── streamlit_app.py        # Streamlit demo app
├── tests/             # Test files
├── .github/           # GitHub workflows
├── Dockerfile         # Docker configuration
├── docker-compose.yml # (Optional) Multi-service setup
├── requirements.txt   # Python dependencies
└── .env               # Environment variables
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized usage)

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-predictor.git
cd f1-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
WEATHER_API_KEY=your_openweather_api_key_here
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t f1-predictor .
```

2. Run the container:
```bash
docker run -p 8501:8501 -p 5000:5000 f1-predictor
```

## Usage

### Training Models

To train the prediction models and log experiments with MLflow:

```bash
python scripts/model_training/train_and_log.py
```

To view the MLflow experiment tracking UI:

```bash
bash scripts/utils/run_mlflow_ui.sh
```

### Running a Simulation

To run the simulation with the Streamlit UI:

```bash
streamlit run web/streamlit_app.py
```

## Implementation Plan

1. **Phase 1: Data Collection and Processing**
   - Collect and process historical F1 data
   - Engineer features for ML models
   - Set up initial data pipeline

2. **Phase 2: Model Development**
   - Train lap time prediction models
   - Develop tire wear simulation models
   - Implement weather integration
   - Set up MLflow tracking

3. **Phase 3: Agent Implementation**
   - Create modular agents following the agent interfaces
   - Develop simulation orchestration system
   - Implement strategy optimization

4. **Phase 4: UI and Deployment**
   - Build Streamlit visualization dashboard
   - Containerize with Docker
   - Create CI/CD pipeline
   - Document and optimize

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Credits

- [FastF1](https://github.com/theOehrly/Fast-F1) - For F1 telemetry data access
- [OpenWeather API](https://openweathermap.org/) - For weather forecasting data
