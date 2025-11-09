import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, precision_score, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Optional: try import RandomOverSampler
use_ros = True
try:
    from imblearn.over_sampling import RandomOverSampler
except Exception:
    use_ros = False
    print("imbalanced-learn tidak tersedia: fallback akan gunakan simple oversampling (duplication).")
    
# =========================
# 1. LOAD DATA
# =========================
dataset_path = "data/predictive_maintenance.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

df = pd.read_csv(dataset_path)
df.columns = [c.strip().replace(" ", "_").replace("[", "").replace("]", "") for c in df.columns]

# Drop Type / Type_Encoded as requested earlier
if "Type" in df.columns:
    df = df.drop(columns=["Type"])
if "Type_Encoded" in df.columns:
    df = df.drop(columns=["Type_Encoded"])

numeric_features = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min"
]

X = df[numeric_features].copy()
y = df["Target"].copy()

# =========================
# 2. SPLIT: train / test (test_size kept 0.3 like you set)
# then split train -> train_sub + val (for threshold tuning & early stopping)
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)
# Now: train : val : test  ~ 56% : 14% : 30%

print("Sizes -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

# =========================
# 3. Oversampling pada training set (only)
# =========================
if use_ros:
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    print("RandomOverSampler used. Before:", y_train.value_counts().to_dict(), "After:", np.bincount(y_res).tolist())
else:
    # simple duplication oversampling of minority class
    counts = y_train.value_counts()
    if len(counts) > 1:
        maj_label = counts.idxmax()
        min_label = counts.idxmin()
        n_maj = counts.max()
        df_train = X_train.copy()
        df_train["target__"] = y_train.values
        df_min = df_train[df_train["target__"] == min_label]
        # duplicate minority until balanced
        repeats = int(np.ceil(n_maj / len(df_min)))
        df_min_dup = pd.concat([df_min]*repeats, ignore_index=True)
        df_bal = pd.concat([df_train[df_train["target__"] == maj_label], df_min_dup], ignore_index=True)
        df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)
        y_res = df_bal["target__"].values
        X_res = df_bal.drop(columns=["target__"]).values
        print("Fallback oversampling used. After counts approx balanced.")
    else:
        X_res, y_res = X_train.values, y_train.values
        print("Hanya satu kelas di train? (aneh)")

# =========================
# 4. Scaling
# =========================
scaler = StandardScaler()
# fit scaler on resampled training
X_res_scaled = scaler.fit_transform(X_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. TRAIN XGBoost with early stopping (use eval_set = val)
# =========================
mlflow.set_experiment("Predictive_Maintenance_NoType_ImprovedRecall")

with mlflow.start_run(run_name="XGBoost_ImprovedRecall") as run:
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # if you want keep scale_pos_weight uncomment and compute earlier,
        # but we already balanced by oversampling so set to 1
        # "scale_pos_weight": 1,
        # "use_label_encoder": False,
        # "objective": "binary:logistic",
        "random_state": 42,
        # "verbosity": 0,
        "eval_metric": "logloss"
    }
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    # early stopping: monitor validation logloss / auc; we'll use logloss
    model.fit(
        X_res_scaled, y_res,
        # eval_set=[(X_val_scaled, y_val)],
        # early_stopping_rounds=20,
        verbose=True
    )

    # wrap scaler + model for serving
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    # =========================
    # 6. Evaluate on validation & test
    # =========================
    # probs on val (for threshold tuning)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    # find best threshold on val by maximizing F1 (you can change to maximize recall)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1
    for thr in thresholds:
        preds = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # ALSO compute threshold that gives highest recall while precision >= 0.3 (optional)
    best_thr_recall = 0.5
    best_recall = 0.0
    for thr in thresholds:
        preds = (val_probs >= thr).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        # require modest precision to avoid too many false positives; tweak 0.3 as needed
        if prec >= 0.3 and rec > best_recall:
            best_recall = rec
            best_thr_recall = thr

    print(f"Best threshold (F1 on val): {best_thr:.3f} (F1={best_f1:.3f})")
    print(f"Best threshold (recall with prec>=0.3): {best_thr_recall:.3f} (recall={best_recall:.3f})")

    # choose final threshold: you can pick best_thr (F1) or best_thr_recall if recall prioritized
    chosen_threshold = best_thr_recall if best_recall > 0 else best_thr
    print("Chosen threshold for serving:", chosen_threshold)

    # Evaluate on test with chosen_threshold
    test_preds = (test_probs >= chosen_threshold).astype(int)
    acc_test = accuracy_score(y_test, test_preds)
    recall_test = recall_score(y_test, test_preds)
    prec_test = precision_score(y_test, test_preds)
    f1_test = f1_score(y_test, test_preds)

    mlflow.log_metric("accuracy", acc_test)
    mlflow.log_metric("precision", prec_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("f1", f1_test)
    mlflow.log_metric("threshold", chosen_threshold)
    print("\n=== FINAL EVALUATION ON TEST SET ===")
    print(f"Accuracy: {acc_test:.4f}")
    print(f"Precision: {prec_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1: {f1_test:.4f}")
    print("\nClassification report:\n", classification_report(y_test, test_preds))

    # confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Failure"], yticklabels=["Normal","Failure"])
    plt.title("Confusion Matrix - Tuned Model")
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/cm_tuned.png")
    plt.close()

    # log model + threshold
    mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")
    # store threshold as artifact
    thr_path = "model_threshold.json"
    with open(thr_path, "w") as f:
        json.dump({"threshold": float(chosen_threshold)}, f)
    mlflow.log_artifact(thr_path, artifact_path="model_info")

    print("Model & threshold logged to MLflow.")

# =======================================================
# 7. REGISTER MODEL KE MLFLOW MODEL REGISTRY
# =======================================================
from mlflow.tracking import MlflowClient

model_name = "Predictive_Maintenance_Model"
run_id = run.info.run_id
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
# 8. TESTING PREDIKSI MANUAL (API SIMULATION)
# =======================================================
print("\n=== 🔍 TESTING PREDIKSI MANUAL ===")

test_input = pd.DataFrame([{
    "Air_temperature_K": 298.5,
    "Process_temperature_K": 308.8,
    "Rotational_speed_rpm": 1500,
    "Torque_Nm": 40.0,
    "Tool_wear_min": 10
}])

prediction = pipeline.predict(test_input)[0]
prob = pipeline.predict_proba(test_input)[0][1]

print("Input data:")
print(test_input)
print(f"\n🔧 Prediksi: {'FAILURE' if prediction == 1 else 'NORMAL'} (Probabilitas: {prob:.3f})")

# =======================================================
# 9. INFO SERVING
# =======================================================
print("\n💡 Untuk serving model sebagai API:")
print('   mlflow models serve -m "models:/Predictive_Maintenance_Model/1" -p 5001')
print("🚀 Endpoint: http://127.0.0.1:5001/invocations")
