# src/train_rf_mlflow.py

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# -------- CONFIG --------
DATA_PATH = "./data/loan_risk_data.csv"   # adjust if needed
ARTIFACT_PATH = "./artifacts"
os.makedirs(ARTIFACT_PATH, exist_ok=True)

# -------- LOAD DATA --------
df = pd.read_csv(DATA_PATH)
X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

# Optional: simple preprocessing for categorical features
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------- MLflow EXPERIMENT --------
mlflow.set_experiment("RF_Loan_Risk_Bayesian")

# -------- Bayesian Search Space --------
search_space = {
    "n_estimators": Integer(50, 500),
    "max_depth": Integer(3, 20),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 5)
}

rf = RandomForestClassifier(class_weight="balanced", random_state=42)

bayes_search = BayesSearchCV(
    rf,
    search_spaces=search_space,
    n_iter=20,                 # number of trials
    cv=StratifiedKFold(n_splits=3),
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=0
)

# -------- START MLflow RUN --------
with mlflow.start_run():

    bayes_search.fit(X_train, y_train)
    best_model = bayes_search.best_estimator_

    # Predict on test
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log best params
    for param, val in bayes_search.best_params_.items():
        mlflow.log_param(param, val)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(best_model, "rf_model")

    # Also save locally in artifacts
    joblib.dump(best_model, os.path.join(ARTIFACT_PATH, "rf_model.pkl"))

    print("✅ Best params:", bayes_search.best_params_)
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
