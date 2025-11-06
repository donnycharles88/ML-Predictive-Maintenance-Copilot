import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import json
import sklearn
import requests
from datetime import datetime

warnings.filterwarnings("ignore")

# Setup folder
os.makedirs("logs", exist_ok=True)

# -------------------------------
# 1. LOAD & PREPARE DATA
# -------------------------------
dataset_path = 'data/preprocessed/processed_dataset.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di path '{dataset_path}'")

df = pd.read_csv(dataset_path)
df.columns = [col.strip().replace(' ', '_') for col in df.columns]

column_mapping = {
    'Air_temperature_[K]': 'Air_temperature_K',
    'Process_temperature_[K]': 'Process_temperature_K',
    'Rotational_speed_[rpm]': 'Rotational_speed_rpm',
    'Torque_[Nm]': 'Torque_Nm',
    'Tool_wear_[min]': 'Tool_wear_min',
    'Product_ID': 'Product_ID',
    'Type': 'Type',
    'UDI': 'UDI',
    'Target': 'Target',
    'Type_Encoded': 'Type_Encoded',
    'Failure_Heat_Dissipation_Failure': 'Failure_Heat_Dissipation_Failure',
    'Failure_Overstrain_Failure': 'Failure_Overstrain_Failure',
    'Failure_Power_Failure': 'Failure_Power_Failure',
    'Failure_Random_Failures': 'Failure_Random_Failures',
    'Failure_Tool_Wear_Failure': 'Failure_Tool_Wear_Failure',
    'Tool_Wear_Bin': 'Tool_Wear_Bin',
    'RPM_Bin': 'RPM_Bin'
}
df.rename(columns=column_mapping, inplace=True)

required_cols = [
    'UDI', 'Product_ID', 'Air_temperature_K', 'Process_temperature_K',
    'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Target',
    'Type_Encoded', 'Failure_Heat_Dissipation_Failure'
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan setelah mapping: {missing_cols}")

print("✅ Semua kolom required tersedia!")

df = df.sort_values('UDI').reset_index(drop=True)

numeric_features = ['Air_temperature_K', 'Process_temperature_K',
                    'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
categorical_features = ['Type_Encoded']
failure_cols = [col for col in df.columns if col.startswith('Failure_') and col != 'Failure_No_Failure']

X = df[numeric_features + categorical_features + failure_cols]
y = df['Target']

# -------------------------------
# 2. TRAIN & LOG MODEL PIPELINE
# -------------------------------
print("🔍 Training Anomaly Detection Model (XGBoost + Scaler Pipeline)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Predictive Maintenance Experiment")

with mlflow.start_run(run_name="XGBoost_Anomaly_Pipeline") as run:
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    }

    model_anomaly = xgb.XGBClassifier(**params)
    scaler = StandardScaler()

    # Pipeline: scaler + model
    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model_anomaly)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "recall": recall,
        "precision": report['1']['precision'],
        "f1_score": report['1']['f1-score']
    })

    # Log model pipeline ke MLflow
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model_pipeline",
        registered_model_name="Predictive_Maintenance_Model_Pipeline"
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix - Anomaly Detection')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("logs/cm_pipeline.png")
    mlflow.log_artifact("logs/cm_pipeline.png", artifact_path="confusion_matrix")
    plt.close()

    print(f"✅ Model pipeline selesai dilatih dan dilog ke MLflow.")
    print(f"🧪 Accuracy: {accuracy:.4f} | Recall: {recall:.4f}")

# -------------------------------
# 3. INFO SERVING API
# -------------------------------
print("\n💡 Jalankan perintah berikut untuk serving model pipeline sebagai REST API:")
print("   mlflow models serve -m \"models:/Predictive_Maintenance_Model_Pipeline/1\" -p 5001")
print("🚀 Endpoint prediksi: http://127.0.0.1:5001/invocations")

# -------------------------------
# 4. TEST PREDIKSI SAMPLE
# -------------------------------
print("\n📡 Mengirim contoh prediksi (nilai mentah) ke endpoint REST API...")

sample_data = {
    "columns": [
        "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
        "Torque_Nm", "Tool_wear_min", "Type_Encoded",
        "Failure_Heat_Dissipation_Failure", "Failure_Overstrain_Failure",
        "Failure_Power_Failure", "Failure_Random_Failures", "Failure_Tool_Wear_Failure"
    ],
    "data": [[300, 310, 1500, 25, 30, 1, 0, 0, 0, 0, 0]]
}

try:
    response = requests.post(
        "http://127.0.0.1:5001/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(sample_data)
    )
    if response.status_code == 200:
        print("✅ Prediksi berhasil diterima dari REST API:")
        print(response.json())
    else:
        print(f"⚠️ Gagal melakukan prediksi (status {response.status_code}): {response.text}")
except requests.exceptions.ConnectionError:
    print("❌ Tidak dapat terhubung ke REST API. Jalankan dulu perintah serving di atas.")
