# src/rf_loan_mlflow.py

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from skopt import BayesSearchCV
from skopt.space import Integer

# ---------------- CONFIG ----------------
DATA_PATH = "../data/loan_risk_data.csv"
ARTIFACT_PATH = "../artifacts"
os.makedirs(ARTIFACT_PATH, exist_ok=True)

mlflow.set_experiment("RF_Loan_Risk_Bayesian")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- MODEL + SEARCH ----------------
search_space = {
    "n_estimators": Integer(50, 300),
    "max_depth": Integer(3, 20),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 5),
}

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

bayes = BayesSearchCV(
    rf,
    search_space,
    n_iter=20,
    scoring="f1_weighted",
    cv=StratifiedKFold(n_splits=3),
    n_jobs=-1,
    random_state=42
)

# ---------------- RUN BAYESIAN SEARCH ----------------
bayes.fit(X_train, y_train)

# ---------------- LOG EACH TRIAL ----------------
results = bayes.cv_results_

for i in range(len(results["params"])):
    with mlflow.start_run(run_name=f"trial_{i+1}"):
        for k, v in results["params"][i].items():
            mlflow.log_param(k, v)

        # BayesSearchCV stores NEGATIVE score
        mlflow.log_metric("f1_weighted", results["mean_test_score"][i])

# ---------------- LOG BEST MODEL ----------------
best_model = bayes.best_estimator_

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

with mlflow.start_run(run_name="BEST_MODEL"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    for k, v in bayes.best_params_.items():
        mlflow.log_param(f"best_{k}", v)

    mlflow.sklearn.log_model(best_model, "rf_model")

joblib.dump(best_model, os.path.join(ARTIFACT_PATH, "rf_model.pkl"))

print("✅ DONE")
print("Best params:", bayes.best_params_)
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
