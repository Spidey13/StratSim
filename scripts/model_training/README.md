# Model Training (`scripts/model_training/`)

This directory contains scripts and notebooks related to training the machine learning models used in the F1 Race Strategy Simulator. The primary models are for lap time prediction and tire wear/degradation.

## Models Overview

The simulation relies on several key machine learning models:

1.  **Lap Time Prediction Model**:
    *   **Purpose**: Predicts F1 lap times based on a multitude of factors including driver, team, circuit, tire compound, tire age/wear, fuel load, and potentially weather conditions.
    *   **Typical Algorithm**: Gradient Boosting Regressors (e.g., XGBoost, LightGBM, CatBoost) are commonly used due to their performance on tabular data.
    *   **Features**: Input features can be extensive, derived from historical F1 data (telemetry, lap timings, session info).
    *   **Output**: Predicted lap time in seconds.

2.  **Tire Wear/Degradation Model**:
    *   **Purpose**: Estimates the rate of tire wear and the impact of wear on performance (e.g., grip loss, increase in lap time degradation).
    *   **Typical Algorithm**: Similar to lap time models, regression techniques are suitable. It might also involve modeling different wear curves for different compounds.
    *   **Features**: Tire compound, laps on tire, circuit abrasiveness, driving style (if available/inferred), track temperature, car setup.
    *   **Output**: Percentage tire wear, grip level, or direct lap time degradation due to wear.

*(Other models, such as a dedicated weather impact model or a more granular grip model, might also be developed here.)*

## Training Process

The general workflow for training these models involves:

1.  **Data Collection & Preprocessing**: Handled by modules in `src/data_pipeline/`. Ensures that clean, feature-rich datasets are available, typically stored in `data/processed/`.
2.  **Feature Engineering**: Creating relevant features from the raw data that are predictive of the target variables (lap time, wear rate).
3.  **Model Selection**: Choosing appropriate algorithms and defining model architectures.
4.  **Hyperparameter Tuning**: Optimizing model hyperparameters using techniques like GridSearchCV or RandomizedSearchCV to achieve the best performance.
5.  **Training**: Fitting the model to the training data.
6.  **Evaluation**: Assessing model performance on a held-out test set using relevant metrics (e.g., RMSE, MAE, R² for regression tasks).
7.  **Saving & Versioning**: Trained models are typically saved as `.joblib` or similar files in the root `models/` directory. MLflow is used for experiment tracking, parameter logging, metric logging, and model versioning.

### Example Training Script Execution (Conceptual)

While specific script names might vary (e.g., `train_lap_time_model.py`, `train_tire_wear_model.py`), the execution pattern is generally:

```bash
# Example for a lap time model
python scripts/model_training/train_lap_time_model.py

# Example for a tire wear model
python scripts/model_training/train_tire_wear_model.py
```

These scripts will typically:
*   Load the processed data.
*   Perform any final feature selection or transformations.
*   Train the specified model with hyperparameter tuning.
*   Evaluate the model.
*   Save the trained model artifact (e.g., to `models/laptime_xgboost_pipeline_tuned_vX.joblib`).
*   Log all relevant information (parameters, metrics, model artifact) to MLflow.

### Viewing Results in MLflow

MLflow is configured to track experiments in the `mlruns/` directory. To view the MLflow UI:

1.  Ensure MLflow is installed (`pip install mlflow`).
2.  Navigate to the project root directory in your terminal.
3.  Run the MLflow UI server:
    ```bash
    mlflow ui
    ```
    This will typically start the server on `http://localhost:5000` (or `http://127.0.0.1:5000`).

Open this URL in your browser to view experiments, compare runs, and inspect model artifacts.

## Trained Model Usage

The trained models saved in the `models/` directory are loaded by their respective agents or components within the simulation (e.g., `LapTimeAgent` loads a lap time model, `TireWearAgent` loads a tire wear model). The paths to these models are usually defined in `src/config/settings.py`.

## Key Considerations

-   **Feature Importance**: Analyzing feature importance after training is crucial to understand which factors most influence predictions and to validate that the model behaves as expected.
-   **Model Robustness**: Models should be robust across different circuits, weather conditions, and seasons if possible.
-   **Retraining Strategy**: A strategy for periodically retraining models with new F1 data should be considered to maintain accuracy as car regulations and team performances evolve.

---
*This README provides a general overview. Refer to specific Python scripts and notebooks within this directory for detailed implementation of model training procedures.* 