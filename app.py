import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0B0F19;
    color: #E5E7EB;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #F9FAFB;
}

[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 20px;
    border-radius: 16px;
}

.stDataFrame {
    background-color: #111827;
    border-radius: 12px;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("fraud_model.pkl")

st.title("💳 AI-Powered Credit Card Fraud Detection System")
st.caption("Enterprise-style banking transaction risk analysis dashboard")

st.sidebar.header("Transaction Input")

amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=500.0)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 23)
customer_age = st.sidebar.slider("Customer Age", 18, 100, 35)
distance = st.sidebar.number_input("Customer-Merchant Distance", min_value=0.0, value=0.8)

category = st.sidebar.selectbox(
    "Merchant Category",
    ["grocery_pos", "shopping_net", "misc_net", "shopping_pos",
     "gas_transport", "misc_pos", "kids_pets", "entertainment",
     "personal_care", "home"]
)

gender = st.sidebar.selectbox("Gender", ["F", "M"])

def build_input(amt, hour, customer_age, distance, gender):
    return pd.DataFrame([{
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

def get_risk(prob):
    if prob >= 0.70:
        return "HIGH", "Block or Manual Review"
    elif prob >= 0.30:
        return "MEDIUM", "Manual Review"
    else:
        return "LOW", "Approve"

def risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#EF4444" if score >= 70 else "#F59E0B" if score >= 30 else "#22C55E"},
            "steps": [
                {"range": [0, 30], "color": "#14532D"},
                {"range": [30, 70], "color": "#713F12"},
                {"range": [70, 100], "color": "#7F1D1D"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0B0F19",
        font={"color": "#E5E7EB"}
    )
    return fig

input_data = build_input(amt, hour, customer_age, distance, gender)

tab1, tab2, tab3 = st.tabs([
    "Single Transaction Analysis",
    "Live Transaction Feed",
    "System Overview"
])

with tab1:
    st.subheader("Transaction Summary")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Analyze Transaction"):
        prob = model.predict_proba(input_data)[0][1]
        risk_score = int(prob * 100)
        risk, action = get_risk(prob)

        if risk == "HIGH":
            st.error("🚨 Suspicious transaction detected")
        elif risk == "MEDIUM":
            st.warning("⚠️ Transaction requires manual review")
        else:
            st.success("✅ Transaction appears safe")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fraud Probability", f"{prob:.2%}")

        with col2:
            st.metric("Risk Level", risk)

        with col3:
            st.metric("Recommended Action", action)

        st.plotly_chart(risk_gauge(risk_score), use_container_width=True)

        st.subheader("AI Analyst Explanation")

        reasons = []

        if amt > 500:
            reasons.append("High transaction amount strongly increases fraud risk.")
        else:
            reasons.append("Low transaction amount reduces fraud risk.")

        if hour >= 22 or hour <= 3:
            reasons.append("Late-night transaction time increases suspicion.")
        else:
            reasons.append("Transaction occurred during normal hours.")

        if distance > 1:
            reasons.append("Customer and merchant locations appear far apart.")
        else:
            reasons.append("Customer and merchant locations appear close.")

        if category in ["shopping_net", "misc_net", "grocery_pos", "gas_transport"]:
            reasons.append(f"The merchant category ({category}) has higher fraud activity in the dataset.")

        for r in reasons:
            st.write(f"- {r}")

        st.subheader("Final Decision")
        st.write(f"Risk Level: **{risk}**")
        st.write(f"Action: **{action}**")

with tab2:
    st.subheader("Live Transaction Feed Simulation")

    np.random.seed(42)
    rows = []

    for i in range(15):
        random_amt = np.random.choice([25, 60, 120, 500, 1200, 3500])
        random_hour = np.random.randint(0, 24)
        random_age = np.random.randint(18, 75)
        random_distance = np.random.uniform(0.1, 2.5)

        tx = build_input(random_amt, random_hour, random_age, random_distance, "F")
        p = model.predict_proba(tx)[0][1]
        r, a = get_risk(p)

        rows.append({
            "Transaction ID": f"TX-{1000+i}",
            "Amount": random_amt,
            "Hour": random_hour,
            "Distance": round(random_distance, 2),
            "Fraud Probability": f"{p:.2%}",
            "Risk": r,
            "Action": a
        })

    feed_df = pd.DataFrame(rows)
    st.dataframe(feed_df, use_container_width=True)

with tab3:
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "XGBoost")

    with col2:
        st.metric("Fraud Recall", "94%")

    with col3:
        st.metric("ROC-AUC", "0.999")

    with col4:
        st.metric("Features", "19")

    st.write("### System Architecture")

    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style="rounded,filled", color="#2563EB", fillcolor="#111827", fontcolor="#E5E7EB"];

        A [label="Transaction Input"];
        B [label="Feature Engineering"];
        C [label="XGBoost Fraud Model"];
        D [label="Fraud Probability"];
        E [label="Risk Level"];
        F [label="Recommended Action"];
        G [label="Analyst Dashboard"];

        A -> B;
        B -> C;
        C -> D;
        D -> E;
        E -> F;
        F -> G;
    }
    """)

    st.write("### Model Pipeline")
    st.write("""
    1. Transaction data is collected.
    2. Features are engineered from amount, time, customer profile, and location.
    3. The XGBoost model estimates fraud probability.
    4. The system assigns a risk level and recommends an action.
    5. Analysts review explanations to support decision-making.
    """)

    st.write("### Key Fraud Signals")
    st.write("""
    - High transaction amount
    - Late-night transaction time
    - Risky merchant category
    - Customer-merchant distance
    - Customer profile and location-related patterns
    """)
