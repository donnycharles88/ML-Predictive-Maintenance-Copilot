import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# =======================================================
# 1. LOAD DATA
# =======================================================
dataset_path = "data/predictive_maintenance.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

df = pd.read_csv(dataset_path)
df.columns = [col.strip().replace(' ', '_').replace('[', '').replace(']', '') for col in df.columns]

print("✅ Dataset berhasil dimuat:", df.shape)
print(df.head(3))

# =======================================================
# 2. ENCODING & FEATURE ENGINEERING
# =======================================================
# Encode kolom Type (L, M, H)
le = LabelEncoder()
df["Type_Encoded"] = le.fit_transform(df["Type"])

# Tambahkan one-hot untuk Failure_Type
df = pd.get_dummies(df, columns=["Failure_Type"], prefix="Failure")

# =======================================================
# 3. FEATURE SELECTION
# =======================================================
numeric_features = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min"
]
categorical_features = ["Type_Encoded"]

# Ambil semua kolom yang berawalan "Failure_"
failure_features = [col for col in df.columns if col.startswith("Failure_")]

X = df[numeric_features + categorical_features]
y = df["Target"]

# =======================================================
# 4. SPLIT DATA
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================================================
# 5. CLASS IMBALANCE HANDLING
# =======================================================
# Hitung rasio imbalance untuk scale_pos_weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"📊 Rasio kelas: Normal={neg}, Failure={pos} | scale_pos_weight={scale_pos_weight:.2f}")

# =======================================================
# 6. TRAINING MODEL
# =======================================================
mlflow.set_experiment("Predictive_Maintenance_Optimized")

with mlflow.start_run(run_name="XGBoost_Optimized") as run:
    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "eval_metric": "logloss"
    }

    model = xgb.XGBClassifier(**params)
    scaler = StandardScaler()

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(report)

    # MLflow logging
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "recall": recall,
    })
    mlflow.sklearn.log_model(pipeline, "model_pipeline")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix - Optimized Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/confusion_matrix_optimized.png")
    plt.close()
# =======================================================
# REGISTER MODEL KE MLFLOW MODEL REGISTRY
# =======================================================
from mlflow.tracking import MlflowClient

model_name = "Predictive_Maintenance_Model"
run_id = run.info.run_id  # otomatis dari run aktif
model_uri = f"runs:/{run_id}/model_pipeline"

client = MlflowClient()
try:
    client.create_registered_model(model_name)
except Exception:
    print(f"Model '{model_name}' sudah terdaftar, lanjut ke versi berikutnya.")

client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)
print(f"✅ Model terdaftar di registry dengan nama: {model_name}")

# =======================================================
# 7. TESTING PREDIKSI MANUAL (API SIMULATION)
# =======================================================
print("\n=== 🔍 TESTING PREDIKSI MANUAL ===")

# contoh input (bisa ubah sesuai kebutuhan)
test_input = pd.DataFrame([{
    "Air_temperature_K": 298.5,
    "Process_temperature_K": 308.8,
    "Rotational_speed_rpm": 1500,
    "Torque_Nm": 40.0,
    "Tool_wear_min": 10,
    "Type_Encoded": le.transform(["M"])[0],
    # default nilai 0 utk kolom failure agar input API tetap valid
    # **{col: 0 for col in failure_features}
}])

prediction = pipeline.predict(test_input)[0]
prob = pipeline.predict_proba(test_input)[0][1]

print("Input data:")
print(test_input[numeric_features + categorical_features])
print(f"\n🔧 Prediksi: {'FAILURE' if prediction == 1 else 'NORMAL'} (Probabilitas: {prob:.3f})")

# =======================================================
# 8. INFO SERVING
# =======================================================
print("\n💡 Untuk serving model sebagai API:")
print("   mlflow models serve -m \"models:/Predictive_Maintenance_Model/1\" -p 5001")
print("🚀 Endpoint: http://127.0.0.1:5001/invocations")
