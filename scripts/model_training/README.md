# Lap Time Prediction Model Training

This directory contains scripts for training and evaluating the lap time prediction model used in the F1 Race Strategy Simulator.

## Model Overview

The lap time prediction model is a supervised machine learning model that predicts F1 lap times based on various factors:
- Tire compound and wear
- Weather conditions (track temperature, air temperature, humidity, rainfall)
- Circuit characteristics

The model uses XGBoost regression with hyperparameter optimization via RandomizedSearchCV.

## Training the Model

### Step 1: Ensure Data is Ready

Make sure the processed F1 data is available at:
```
data/processed/seasons_2023_2024_data_enhanced.csv
```

### Step 2: Run the Training Script

```bash
python scripts/model_training/train_lap_time_model.py
```

This will:
1. Load and prepare the data
2. Train an XGBoost model with hyperparameter tuning
3. Evaluate the model performance
4. Save the model to `models/laptime_xgboost_pipeline_tuned_v1.joblib`
5. Log the model, parameters, and metrics to MLflow

### Step 3: View Results in MLflow

Start the MLflow UI:

```bash
python scripts/utils/run_mlflow_ui.py
```

Then open your browser to [http://localhost:5000](http://localhost:5000) to view experiments.

## Using the Trained Model

The trained model will be automatically loaded by the `LapTimeAgent` in the simulation.

## Performance Metrics

The model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

## Feature Importance

After training, you can analyze feature importance to understand which factors most influence lap time predictions. 