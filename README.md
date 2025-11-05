# ML-Predictive-Maintenance-Copilot


Predictive Maintenance with XGBoost & MLflow
🧾 Overview
This project trains an anomaly detection model for predictive maintenance using XGBoost and logs the entire process with MLflow. The model is then served as a REST API for real-time predictions.

🧰 Requirements
Install the required Python packages:
pip install pandas numpy xgboost scikit-learn matplotlib seaborn joblib mlflow requests


🚀 Running the Training Script
Run the training script to:

Load and preprocess the dataset
Train an XGBoost model
Log metrics and artifacts to MLflow
Register the model to MLflow Model Registry
Send a sample prediction to the REST API
python predictive_maintenance.py




🔬 MLflow Tracking UI (Optional)
To view experiment details:
mlflow ui


Open in browser: http://127.0.0.1:5000


🌐 Serving the Model as REST API
After the model is registered, serve it using MLflow:
mlflow models serve -m "models:/Predictive_Maintenance_Model/1" -p 5001 --env-manager local


This will expose the prediction endpoint at:
http://127.0.0.1:5001/invocations




📡 Sending Prediction Requests
Use the following Python snippet to send a sample prediction:
import requests
import json

sample_data = {
    "columns": [
        "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
        "Torque_Nm", "Tool_wear_min", "Type_Encoded", "Failure_Heat_Dissipation_Failure"
    ],
    "data": [[300, 310, 1500, 35, 150, 1, 0]]
}

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(sample_data)
)

print(response.json())  # Output: [0] or [1]




🧪 Prediction Output

0 → Machine is normal
1 → Machine has an anomaly → needs inspection or repair


📁 Artifacts Generated

logs/model_anomaly/ → Trained XGBoost model
logs/scaler_anomaly.pkl → Scaler used for preprocessing
logs/cm_anomaly.png → Confusion matrix visualization
logs/cm_anomaly.json → Confusion matrix data
