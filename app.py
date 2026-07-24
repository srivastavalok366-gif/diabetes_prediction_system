"""
app.py
------
Flask web app for the Diabetes Prediction System.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request

MODEL_DIR = "model"

app = Flask(__name__)

model = None
scaler = None
feature_columns = None


def load_artifacts():
    global model, scaler, feature_columns
    model_path = os.path.join(MODEL_DIR, "diabetes_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    columns_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            "Model files not found. Please run 'python train_model.py' "
            "first to train and save the model."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(columns_path)


FIELDS = [
    ("Pregnancies", "Pregnancies", "e.g. 2", 0, 20, "1"),
    ("Glucose", "Glucose (mg/dL)", "e.g. 120", 0, 300, "1"),
    ("BloodPressure", "Blood Pressure (mm Hg)", "e.g. 70", 0, 200, "1"),
    ("SkinThickness", "Skin Thickness (mm)", "e.g. 20", 0, 100, "1"),
    ("Insulin", "Insulin (mu U/mL)", "e.g. 80", 0, 900, "1"),
    ("BMI", "BMI", "e.g. 28.5", 0, 70, "0.1"),
    ("DiabetesPedigreeFunction", "Diabetes Pedigree Function", "e.g. 0.45", 0, 3, "0.001"),
    ("Age", "Age (years)", "e.g. 35", 1, 120, "1"),
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", fields=FIELDS, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []
        submitted = {}
        for key, *_ in FIELDS:
            raw = request.form.get(key, "")
            value = float(raw)
            input_values.append(value)
            submitted[key] = raw

        X = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        result = {
            "prediction": prediction,
            "label": "Diabetic risk detected" if prediction == 1 else "Low diabetes risk",
            "probability": round(probability * 100, 1),
        }

        return render_template("index.html", fields=FIELDS, result=result, submitted=submitted)

    except (ValueError, TypeError):
        error = "Please enter valid numeric values in every field."
        return render_template("index.html", fields=FIELDS, result=None, error=error)


if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # Also load artifacts if imported by a WSGI server
    try:
        load_artifacts()
    except FileNotFoundError:
        pass

