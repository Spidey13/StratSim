"""
Script to start the MLflow UI with the correct tracking URI.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config.settings import MLFLOW_TRACKING_URI


def main():
    """Start the MLflow UI with the configured tracking URI."""
    print(f"Starting MLflow UI with tracking URI: {MLFLOW_TRACKING_URI}")

    # Start MLflow UI
    subprocess.run(
        ["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI], check=True
    )


if __name__ == "__main__":
    main()
