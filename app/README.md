# Diabetes Prediction System

A machine learning web app that predicts diabetes risk from 8 clinical
measurements (Pima Indians Diabetes feature set), built with
**scikit-learn** (model) and **Flask** (web UI).

## Project structure
```
diabetes_prediction_system/
├── app.py                 # Flask web app
├── train_model.py         # Trains and saves the ML model
├── requirements.txt
├── data/
│   └── diabetes.csv       # (optional) put the real Kaggle dataset here
├── model/                 # created automatically after training
│   ├── diabetes_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## Setup in VS Code

1. **Open the folder** in VS Code (`File > Open Folder`).

2. **Create a virtual environment** (Terminal in VS Code, `Ctrl+\``):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

   In VS Code, also select this interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `venv`.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional but recommended) Add the real dataset:**
   - Download `diabetes.csv` from Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
   - Place it at `data/diabetes.csv`
   - If you skip this step, `train_model.py` automatically generates a
     synthetic dataset with the same 8 features so the project still runs.

5. **Train the model:**
   ```bash
   python train_model.py
   ```
   This prints accuracy/precision/recall/ROC-AUC and saves the trained
   model into the `model/` folder.

6. **Run the web app:**
   ```bash
   python app.py
   ```
   Open the link shown in the terminal (usually http://127.0.0.1:5000)
   in your browser. Fill in the 8 fields and click **Run prediction**.

## The 8 input features

| Feature | Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic/family-history risk score |
| Age | Age in years |

## Notes for your project report / viva

- **Algorithm selection:** `train_model.py` trains both Logistic
  Regression and Random Forest with `GridSearchCV` (5-fold cross
  validation) and automatically keeps whichever scores higher on
  ROC-AUC.
- **Preprocessing:** Biologically-impossible zero values in Glucose,
  BloodPressure, SkinThickness, Insulin, and BMI are treated as
  missing and imputed with the column median, then all features are
  standardized with `StandardScaler` before training.
- **Evaluation metrics printed:** Accuracy, Precision, Recall, F1
  score, ROC-AUC, confusion matrix, and full classification report.
- You can swap in other models (SVM, XGBoost, KNN) by adding them to
  the `candidates` dictionary in `train_model.py`.

## Disclaimer
This is an educational project. It is **not** a certified medical
device and should never be used for real clinical decisions.
