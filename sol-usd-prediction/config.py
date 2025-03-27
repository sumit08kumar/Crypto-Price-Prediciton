# config.py
# Configuration settings for the Crypto Price Prediction API

# API settings
API = {
    "HOST": "127.0.0.1",
    "PORT": 8000,  # Matches your curl commands
    "BASE_URL": "http://127.0.0.1:8000",
    "ENDPOINTS": {
        "FETCH_DATA": "/fetch_data",
        "TRAIN": "/train",
        "PREDICT": "/predict",
        "EVALUATE": "/evaluate",
        "SAVE_MODEL": "/save_model",
        "LOAD_MODEL": "/load_model"
    }
}

# Tiingo API settings
TIINGO = {
    "API_KEY": "133108c9aaa985638656b96fe20a1f4ae15c097a",  # Your Tiingo API key
    "SYMBOL": "solusd",
    "START_DATE": "2024-01-01",
    "END_DATE": "dynamic",  # Will be set to current date in code
    "FREQUENCY": "5min"
}

# Model settings
MODEL = {
    "LOOKBACK": 50,          # Window size for feature creation
    "SPLIT_RATIO": 0.8,      # Train/test split proportion
    "OPTUNA_TRIALS": 50,     # Number of hyperparameter tuning trials
    "SAVE_JSON": "xgboost_model.json",
    "SAVE_PKL": "xgboost_model.pkl"
}