# model.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from datetime import datetime
import optuna
from tiingo import TiingoClient
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import joblib

class CryptoPricePredictor:
    def __init__(self, api_key, symbol='solusd', start_date='2024-01-01', end_date='2025-01-01', frequency='5min'):
        self.api_key = api_key
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.config = {'api_key': api_key, 'session': True}
        self.client = TiingoClient(self.config)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.df = None
        self.X_train_rf = None
        self.X_test_rf = None
        self.y_train = None
        self.y_test = None

    def fetch_data(self):
        """Fetch historical price data from Tiingo API."""
        historical_prices = self.client.get_crypto_price_history(
            tickers=[self.symbol],
            startDate=self.start_date,
            endDate=self.end_date,
            resampleFreq=self.frequency
        )
        self.df = pd.DataFrame(historical_prices[0]['priceData'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        print("Data fetched successfully.")
        return self.df

    def preprocess_data(self, lookback=50, split_ratio=0.8):
        """Preprocess the data for training."""
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['month'] = self.df['date'].dt.month

        split_idx = int(split_ratio * len(self.df))
        self.scaler.fit(self.df[['close']][:split_idx])
        self.df['scaled_close'] = self.scaler.transform(self.df[['close']])

        features = ['scaled_close', 'hour', 'day_of_week', 'week_of_year', 'month']
        X, y = [], []
        for i in range(lookback, len(self.df)):
            X.append(self.df[features].iloc[i-lookback:i].values)
            y.append(self.df['scaled_close'].iloc[i])

        X, y = np.array(X), np.array(y)
        self.X_train, self.X_test = X[:split_idx - lookback], X[split_idx - lookback:]
        self.y_train, self.y_test = y[:split_idx - lookback], y[split_idx - lookback:]
        self.X_train_rf = self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test_rf = self.X_test.reshape(self.X_test.shape[0], -1)

        print("Data preprocessed successfully.")
        return self.X_train_rf, self.X_test_rf, self.y_train, self.y_test

    def objective(self, trial):
        """Objective function for Optuna hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }
        model = xgb.XGBRegressor(objective='reg:squarederror', **params)
        model.fit(self.X_train_rf, self.y_train)
        y_pred = model.predict(self.X_test_rf)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return rmse

    def train_model(self, n_trials=50):
        """Train the XGBoost model with hyperparameter tuning."""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best Parameters: {best_params}")

        self.model = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
        self.model.fit(self.X_train_rf, self.y_train)
        print("Model trained successfully.")
        return self.model

    def evaluate_model(self):
        """Evaluate the trained model and visualize results."""
        y_pred = self.model.predict(self.X_test_rf)
        y_test_original = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        directional_acc = np.mean(np.sign(np.diff(self.y_test)) == np.sign(np.diff(y_pred)))
        corr_coeff, p_value = pearsonr(self.y_test, y_pred)
        conf_interval_lower = corr_coeff - (1.96 * np.sqrt((1 - corr_coeff**2) / (len(self.y_test) - 2)))

        # Training RMSE
        y_train_pred = self.model.predict(self.X_train_rf)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))

        # Print evaluation metrics
        print(f"RMSE (XGBoost): {rmse}")
        print(f"Directional Accuracy: {directional_acc}")
        print(f"Pearson Correlation Coefficient: {corr_coeff}")
        print(f"P-value: {p_value}")
        print(f"Confidence Interval Lower Bound: {conf_interval_lower}")
        print(f"XGBoost RMSE: Train = {rmse_train:.4f}, Test = {rmse:.4f}")

        # Performance verification and overfitting check
        if p_value < 0.05 and conf_interval_lower > 0.52 and corr_coeff > 0.05:
            print("Performance verification passed")
        else:
            print("Performance verification failed")

        if rmse_train < rmse * 0.8:
            print("Model is Overfitting! Consider Regularization.")
        else:
            print("No major Overfitting detected.")

        return rmse, directional_acc, corr_coeff

    def save_model(self, filename_json="xgboost_model.json", filename_pkl="xgboost_model.pkl"):
        """Save the trained model in both JSON and pickle formats."""
        self.model.save_model(filename_json)
        joblib.dump(self.model, filename_pkl)
        print(f"Model saved as {filename_json} and {filename_pkl}")

    def load_model(self, filename_json="xgboost_model.json"):
        """Load the model from a JSON file."""
        self.model = xgb.Booster()
        self.model.load_model(filename_json)
        print("Model loaded successfully.")
        return self.model

    def predict(self, test_features):
        """Make predictions using the loaded model."""
        dmatrix = xgb.DMatrix(test_features)
        prediction = self.model.predict(dmatrix)
        return prediction

if __name__ == "__main__":
    api_key = '133108c9aaa985638656b96fe20a1f4ae15c097a'
    predictor = CryptoPricePredictor(api_key)

    # Fetch and preprocess data
    predictor.fetch_data()
    predictor.preprocess_data()

    # Train and evaluate model
    predictor.train_model()
    predictor.evaluate_model()

    # Save the model
    predictor.save_model()

    # Load and test the model
    predictor.load_model()
    test_features = np.random.rand(1, 250)  # Example test features
    prediction = predictor.predict(test_features)
    print("Test Prediction:", prediction)