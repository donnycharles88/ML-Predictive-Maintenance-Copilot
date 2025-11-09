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

## 🔬 MLflow Tracking UI 

To view experiment details, run:

```bash
mlflow ui
```

## 🚀 Running the Training Script
```bash
python predictive.py
```
---

Then open in your browser:

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Serving the Model as REST API

After the model is registered, serve it using MLflow:

```bash
mlflow models serve -m "models:/Predictive_Maintenance_Model_Pipeline/1" -p 5001 --env-manager local
```

This will expose the prediction endpoint at:

```
http://127.0.0.1:5001/invocations
```

Maka sekarang kamu bisa kirim payload seperti ini:

```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5001/invocations -ContentType 'application/json' -Body '{"dataframe_split": {"columns": ["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"], "data": [[308, 315, 1200, 75.0, 240]]}}'

```

dan hasilnya seharusnya jadi:
```
predictions
-----------
{1}

```
## 🧪 Prediction Output

| Output | Meaning |
|:------:|:---------|
| `0` | Machine is **normal** |
| `1` | Machine has an **anomaly** → needs inspection or repair |

---

