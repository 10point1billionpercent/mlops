import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Loan Risk Predictor", page_icon="💳", layout="centered")

st.title("💳 Loan Risk Prediction")
st.caption("Single prediction + Batch CSV prediction")

# load model
MODEL_PATH = os.path.join("artifacts", "rf_model.pkl")
model = joblib.load(MODEL_PATH)

# ----------------- helpers -----------------
def nice_result(pred_label: str):
    if pred_label == "Low Risk":
        st.success(f"✅ Predicted Risk: **{pred_label}**")
    elif pred_label == "Medium Risk":
        st.warning(f"⚠️ Predicted Risk: **{pred_label}**")
    else:
        st.error(f"🚨 Predicted Risk: **{pred_label}**")

def proba_dict(df_one_row):
    proba = model.predict_proba(df_one_row)[0]
    class_names = list(model.named_steps["rf"].classes_)
    return {cls: float(p) for cls, p in zip(class_names, proba)}

def add_predictions(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    preds = model.predict(df)
    probs = model.predict_proba(df)
    class_names = list(model.named_steps["rf"].classes_)

    df["PredictedRisk"] = preds

    # add probability columns
    for i, cls in enumerate(class_names):
        df[f"Prob_{cls.replace(' ', '_')}"] = probs[:, i]

    return df

# ----------------- UI Tabs -----------------
tab1, tab2 = st.tabs(["🧍 Single Prediction", "📁 Batch CSV Prediction"])

# ----------------- Single Prediction -----------------
with tab1:
    st.subheader("🧍 Single User Input")

    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income", 0, 1000000, 50000)
    loan_amount = st.number_input("Loan Amount", 0, 1000000, 20000)
    credit_score = st.number_input("Credit Score", 300, 850, 680)
    months_employed = st.number_input("Months Employed", 0, 600, 36)
    num_credit_lines = st.number_input("Num Credit Lines", 0, 50, 3)
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 12.5)
    loan_term = st.number_input("Loan Term (months)", 1, 360, 24)
    dti_ratio = st.number_input("DTI Ratio", 0.0, 1.0, 0.32)

    employment = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
    residence = st.selectbox("Residence Type", ["Rent", "Own", "Mortgage"])
    prev_default = st.selectbox("Previous Default", ["No", "Yes"])

    if st.button("Predict Risk 🚀"):
        one = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
            "EmploymentType": employment,
            "ResidenceType": residence,
            "PreviousDefault": prev_default
        }])

        pred = model.predict(one)[0]
        nice_result(pred)

        probs = proba_dict(one)
        st.write("### 📊 Probabilities (%)")
        st.write({k: round(v * 100, 2) for k, v in probs.items()})

# ----------------- Batch CSV Prediction -----------------
with tab2:
    st.subheader("📁 Upload CSV for Batch Prediction")

    st.info("CSV must contain the same columns used in training (except RiskCategory).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df_batch = pd.read_csv(uploaded)

        st.write("✅ Preview of uploaded data:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Prediction ⚡"):
            try:
                out = add_predictions(df_batch)

                st.success("Done! Predictions added.")
                st.dataframe(out.head())

                # download button
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_bytes,
                    file_name="batch_predictions_output.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error("Something went wrong. Check your CSV columns.")
                st.code(str(e))
