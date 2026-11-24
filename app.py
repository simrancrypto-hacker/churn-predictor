import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")

@st.cache_resource
def load():
    model = joblib.load("models/churn_model.joblib")
    thr = json.load(open("models/threshold.json"))["threshold"]
    return model, thr

model, threshold = load()

st.title("Customer Churn Predictor")
st.caption("Enter a customer's details to estimate the risk of leaving.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
with col2:
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)

total = tenure * monthly  # estimate of total charges so far

if st.button("Predict"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": int(senior == "Yes"),
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    proba = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{proba:.0%}")
    if proba >= threshold:
        st.error("High risk: flag this customer for a retention offer.")
    else:
        st.success("Lower risk: no retention offer needed.")
    st.caption(f"Decision threshold used: {threshold:.2f}")