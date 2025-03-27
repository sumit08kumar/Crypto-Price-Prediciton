# api_client.py
import requests
import config  # Import config.py

# Base URL from config
base_url = config.API["BASE_URL"]

# Fetch data
print("Fetching data...")
response = requests.post(f"{base_url}{config.API['ENDPOINTS']['FETCH_DATA']}")
print("Fetch Data Response:", response.json())

# Train model
print("\nTraining model...")
response = requests.post(f"{base_url}{config.API['ENDPOINTS']['TRAIN']}")
print("Train Response:", response.json())

# Evaluate
print("\nEvaluating model...")
response = requests.get(f"{base_url}{config.API['ENDPOINTS']['EVALUATE']}")
data = response.json()
print("Evaluate Response:", data)

# Save model
print("\nSaving model...")
response = requests.post(f"{base_url}{config.API['ENDPOINTS']['SAVE_MODEL']}")
print("Save Model Response:", response.json())

# Load model
print("\nLoading model...")
response = requests.post(f"{base_url}{config.API['ENDPOINTS']['LOAD_MODEL']}")
print("Load Model Response:", response.json())

# Predict
print("\nMaking prediction...")
response = requests.post(f"{base_url}{config.API['ENDPOINTS']['PREDICT']}")
prediction_data = response.json()
print("Predict Response:", prediction_data)
if "prediction" in prediction_data:
    print(f"Predicted Value: {prediction_data['prediction']}")

if "rmse" in data:
    print(f"RMSE: {data['rmse']}, Directional Accuracy: {data['directional_accuracy']}, Correlation: {data['correlation_coefficient']}")