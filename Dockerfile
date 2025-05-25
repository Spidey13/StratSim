FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/cache data/raw data/processed models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose ports for Streamlit and MLflow
EXPOSE 8501 5000

# Default command to run the Streamlit app
CMD ["streamlit", "run", "web/streamlit_app.py"] 