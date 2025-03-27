# app.py
from flask import Flask, request, jsonify
import numpy as np
from model import CryptoPricePredictor
from datetime import datetime
import config  # Import config.py

app = Flask(__name__)

# Initialize predictor with config values
predictor = CryptoPricePredictor(
    api_key=config.TIINGO["API_KEY"],
    symbol=config.TIINGO["SYMBOL"],
    start_date=config.TIINGO["START_DATE"],
    end_date=datetime.now().strftime('%Y-%m-%d') if config.TIINGO["END_DATE"] == "dynamic" else config.TIINGO["END_DATE"],
    frequency=config.TIINGO["FREQUENCY"]
)

@app.route('/', methods=['GET'])
def home():
    """Return a welcome message for the API."""
    return jsonify({
        "message": "Welcome to the Crypto Price Prediction API",
        "endpoints": config.API["ENDPOINTS"]
    })

@app.route(config.API["ENDPOINTS"]["FETCH_DATA"], methods=['POST'])
def fetch_data():
    """Fetch historical data from Tiingo API."""
    try:
        predictor.fetch_data()
        if predictor.df is None or predictor.df.empty:
            return jsonify({"error": "No data fetched"}), 500
        return jsonify({
            "message": "Data fetched successfully",
            "data_head": predictor.df.head().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["TRAIN"], methods=['POST'])
def train():
    """Train the model with the fetched data."""
    try:
        if predictor.df is None or predictor.df.empty:
            return jsonify({"error": "Please fetch data first"}), 400
        predictor.preprocess_data(lookback=config.MODEL["LOOKBACK"], split_ratio=config.MODEL["SPLIT_RATIO"])
        predictor.train_model(n_trials=config.MODEL["OPTUNA_TRIALS"])
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to train model: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["PREDICT"], methods=['POST'])
def predict():
    """Make a prediction using the trained model."""
    try:
        if predictor.model is None:
            return jsonify({"error": "Model not trained yet"}), 400
        if predictor.X_test_rf is None:
            return jsonify({"error": "No test data available; train the model first"}), 400
        
        test_features = predictor.X_test_rf[-1].reshape(1, -1)
        prediction = predictor.predict(test_features)
        
        if not isinstance(prediction, np.ndarray):
            return jsonify({"error": f"Unexpected prediction type: {type(prediction)}"}), 500
        
        prediction_2d = prediction.reshape(-1, 1)
        prediction_original = predictor.scaler.inverse_transform(prediction_2d).flatten()[0]
        
        return jsonify({"prediction": float(prediction_original)})
    except Exception as e:
        return jsonify({"error": f"Failed to predict: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["EVALUATE"], methods=['GET'])
def evaluate():
    """Evaluate the model and return metrics without a plot."""
    try:
        if predictor.model is None:
            return jsonify({"error": "Model not trained yet"}), 400
        
        rmse, directional_acc, corr_coeff = predictor.evaluate_model()
        
        return jsonify({
            "rmse": float(rmse),
            "directional_accuracy": float(directional_acc),
            "correlation_coefficient": float(corr_coeff)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to evaluate: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["SAVE_MODEL"], methods=['POST'])
def save_model():
    """Save the trained model."""
    try:
        if predictor.model is None:
            return jsonify({"error": "Model not trained yet"}), 400
        predictor.save_model(
            filename_json=config.MODEL["SAVE_JSON"],
            filename_pkl=config.MODEL["SAVE_PKL"]
        )
        return jsonify({"message": "Model saved successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to save model: {str(e)}"}), 500

@app.route(config.API["ENDPOINTS"]["LOAD_MODEL"], methods=['POST'])
def load_model():
    """Load a previously saved model."""
    try:
        predictor.load_model(filename_json=config.MODEL["SAVE_JSON"])
        return jsonify({"message": "Model loaded successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host=config.API["HOST"], port=config.API["PORT"])