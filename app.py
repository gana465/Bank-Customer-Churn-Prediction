import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------
# Load Model and Scaler
# ----------------------------

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1f4e79;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#666666;
}

.stButton > button {
    width:100%;
    background-color:#1f4e79;
    color:white;
    font-size:18px;
    height:55px;
    border-radius:10px;
}

.metric-box {
    background-color:white;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------

st.markdown(
    '<p class="title">🏦 Customer Churn Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a customer is likely to leave the bank</p>',
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("📊 Project Information")

    st.write("""
    **Model Used:** XGBoost
    
    **Dataset Size:** 10,000 Customers
    
    **Goal:** Predict customer churn using
    banking and demographic information.
    """)

    st.divider()

    st.write("### Churn Risk Factors")

    st.write("""
    - Age
    - Balance
    - Geography
    - Active Membership
    - Number of Products
    """)

# ----------------------------
# Input Form
# ----------------------------

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=70000.0
    )

with col2:

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    is_active = st.selectbox(
        "Is Active Member",
        ["Yes", "No"]
    )

# ----------------------------
# Encoding
# ----------------------------

gender_male = 1 if gender == "Male" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

has_card = 1 if has_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

# ----------------------------
# Prediction Button
# ----------------------------

st.write("")
st.write("")

if st.button("🔍 Predict Churn"):

    data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        is_active,
        estimated_salary,
        geo_germany,
        geo_spain,
        gender_male
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with colB:
        st.metric(
            "Risk Level",
            "High" if probability > 0.5 else "Low"
        )

    st.write("")

    if prediction == 1:

        st.error(
            f"⚠ Customer is likely to Churn\n\n"
            f"Predicted Churn Probability: {probability*100:.2f}%"
        )

    else:

        st.success(
            f"✅ Customer is likely to Stay\n\n"
            f"Predicted Churn Probability: {probability*100:.2f}%"
        )

# ----------------------------
# Footer
# ----------------------------

st.divider()

st.caption(
    "Customer Churn Prediction using Machine Learning and XGBoost"
)