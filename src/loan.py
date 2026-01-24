import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# load data
df = pd.read_csv("./data/loan_risk_data.csv")

# target
y = df["RiskCategory"]
X = df.drop("RiskCategory", axis=1)

# categorical + numeric columns
cat_cols = ["EmploymentType", "ResidenceType", "PreviousDefault"]
num_cols = [c for c in X.columns if c not in cat_cols]

# preprocessing (one-hot for categorical)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

# pipeline
model = Pipeline([
    ("prep", preprocess),
    ("rf", rf)
])

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
