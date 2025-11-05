# ⚙️ ML-Predictive-Maintenance-Copilot

## 🧾 Overview
This project trains an **anomaly detection model** for **predictive maintenance** using **XGBoost** and logs the entire process with **MLflow**.  
The model is then served as a **REST API** for real-time predictions.

---
## 🧰 Requirements
Install the required Python packages:
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn joblib mlflow requests
```
## 🚀 Running the Training Script
```bash
python predictive_maintenance.py
```
---

## 🔬 MLflow Tracking UI (Optional)

To view experiment details, run:

```bash
mlflow ui
```

Then open in your browser:

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Serving the Model as REST API

After the model is registered, serve it using MLflow:

```bash
mlflow models serve -m "models:/Predictive_Maintenance_Model/1" -p 5001 --env-manager local
```

This will expose the prediction endpoint at:

```
http://127.0.0.1:5001/invocations
```

---
## 🧪 Prediction Output

| Output | Meaning |
|:------:|:---------|
| `0` | Machine is **normal** |
| `1` | Machine has an **anomaly** → needs inspection or repair |

---

