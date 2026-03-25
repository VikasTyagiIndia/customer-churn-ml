# app/app.py

import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Developer Info
# -------------------------------
NAME = "Vikas Tyagi"
UNIVERSITY = "Amity University Jaipur"
COURSE = "MSc Data Science"

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, columns

model, training_cols = load_model()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧾 Customer Details")

st.sidebar.markdown("### 👤 Developer Info")
st.sidebar.markdown(f"""
**Name:** {NAME}  
**University:** {UNIVERSITY}  
**Course:** {COURSE}
""")
st.sidebar.markdown("---")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# -------------------------------
# Main UI
# -------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn using a machine learning model.")

# -------------------------------
# Input DataFrame
# -------------------------------
input_dict = {
    "gender": gender,
    "SeniorCitizen": str(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# Feature Engineering
# -------------------------------
def create_features(df):
    df = df.copy()

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )

    df['avg_charge'] = df['TotalCharges'] / (df['tenure'] + 1)

    return df


def preprocess(df):
    df = create_features(df)
    df = pd.get_dummies(df)
    return df


def align_columns(df):
    for col in training_cols:
        if col not in df:
            df[col] = 0
    return df[training_cols]

# -------------------------------
# Prediction Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Summary")
    st.dataframe(input_df, use_container_width=True)

if st.button("🚀 Predict Churn"):

    processed = preprocess(input_df)
    processed = align_columns(processed)

    prob = model.predict_proba(processed)[0][1]

    with col2:
        st.subheader("📊 Prediction Result")

        # Metric
        st.metric("Churn Probability", f"{prob:.2%}")

        # Progress bar
        st.progress(float(prob))

        # Risk interpretation
        if prob > 0.7:
            st.error("🔴 High Risk of Churn")
            st.markdown("**Action:** Offer retention incentives immediately.")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk of Churn")
            st.markdown("**Action:** Monitor and engage customer.")
        else:
            st.success("🟢 Low Risk of Churn")
            st.markdown("**Action:** Customer likely to stay.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(f"""
👨‍💻 **{NAME}**  
🎓 {COURSE} | {UNIVERSITY}  

Built with ❤️ using Streamlit | Model: XGBoost  
""")
