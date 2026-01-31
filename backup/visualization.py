import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# make sure folder exists
SAVE_DIR = "./logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# load saved model
model = joblib.load("./artifacts/rf_model.pkl")
# load data
df = pd.read_csv("./data/loan_risk_data.csv")
y = df["RiskCategory"]
X = df.drop("RiskCategory", axis=1)

# ---- Confusion Matrix Plot ----
plt.figure()
ConfusionMatrixDisplay.from_estimator(model, X, y)
plt.title("Confusion Matrix - Loan Risk Prediction (Full Data)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ---- Feature Importance Plot ----
cat_cols = ["EmploymentType", "ResidenceType", "PreviousDefault"]
num_cols = [c for c in X.columns if c not in cat_cols]

ohe = model.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_cols)

feature_names = list(cat_names) + num_cols
importances = model.named_steps["rf"].feature_importances_

fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

plt.figure()
plt.barh(fi["Feature"][::-1], fi["Importance"][::-1])
plt.title("Top 15 Feature Importances (RandomForest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "feature_importance.png"), dpi=300)
plt.close()

print("Saved plots in:", SAVE_DIR)
