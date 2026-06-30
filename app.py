import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("fraud_model.pkl")

st.title("💳 AI-Powered Real-Time Credit Card Fraud Detection System")
st.write("Banking transaction risk analysis dashboard")

st.sidebar.header("Transaction Input")

amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=500.0)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 23)
customer_age = st.sidebar.slider("Customer Age", 18, 100, 35)
distance = st.sidebar.number_input("Customer-Merchant Distance", min_value=0.0, value=0.8)

category = st.sidebar.selectbox(
    "Merchant Category",
    [
        "grocery_pos",
        "shopping_net",
        "misc_net",
        "shopping_pos",
        "gas_transport",
        "misc_pos",
        "kids_pets",
        "entertainment",
        "personal_care",
        "home"
    ]
)

gender = st.sidebar.selectbox("Gender", ["F", "M"])

input_data = pd.DataFrame([{
    "merchant": 0,
    "category": 0,
    "amt": amt,
    "gender": 0 if gender == "F" else 1,
    "city": 0,
    "state": 0,
    "zip": 0,
    "lat": 0.0,
    "long": 0.0,
    "city_pop": 100000,
    "job": 0,
    "unix_time": 0,
    "merch_lat": 0.0,
    "merch_long": 0.0,
    "hour": hour,
    "customer_age": customer_age,
    "weekday": 0,
    "is_weekend": 0,
    "distance": distance
}])

st.subheader("Transaction Summary")
st.dataframe(input_data)

if st.button("Analyze Transaction"):
    prob = model.predict_proba(input_data)[0][1]

    if prob >= 0.70:
        risk = "HIGH"
        action = "Block or Manual Review"
        st.error("🚨 Suspicious transaction detected")
    elif prob >= 0.30:
        risk = "MEDIUM"
        action = "Manual Review"
        st.warning("⚠️ Transaction requires manual review")
    else:
        risk = "LOW"
        action = "Approve"
        st.success("✅ Transaction appears safe")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fraud Probability", f"{prob:.2%}")

    with col2:
        st.metric("Risk Level", risk)

    with col3:
        st.metric("Recommended Action", action)

    st.subheader("Analyst Explanation")

    reasons = []

    if amt > 500:
        reasons.append("High transaction amount increases fraud risk.")
    else:
        reasons.append("Low transaction amount reduces fraud risk.")

    if hour >= 22 or hour <= 3:
        reasons.append("Transaction occurred during late-night hours, which increases risk.")
    else:
        reasons.append("Transaction occurred during normal hours, which reduces risk.")

    if distance > 1:
        reasons.append("Customer and merchant locations appear far apart.")
    else:
        reasons.append("Customer and merchant locations appear close.")

    if category in ["shopping_net", "misc_net", "grocery_pos", "gas_transport"]:
        reasons.append(f"The selected merchant category ({category}) is commonly associated with higher fraud activity in the dataset.")

    for reason in reasons:
        st.write(f"- {reason}")
