import streamlit as st
import pickle
import pandas as pd
import download_model
import plotly.graph_objects as go

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
loaded_model = pickle.load(open("models/rf_model_new.pkl", "rb"))

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#f1651d,#f57c32,#ffb347);
}

/* Hide Streamlit menu */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* Title */

.main-title{
font-size:52px;
font-weight:800;
text-align:center;
color:white;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:20px;
color:#FFF7E6;
margin-bottom:40px;
}

/* Card */

.card{
background:#FCEEBF;
padding:28px;
border-radius:20px;
box-shadow:0px 10px 25px rgba(0,0,0,.25);
margin-bottom:20px;
}

/* Inputs */

.stNumberInput input{
background:#FFF9EC;
color:black;
border-radius:12px;
border:2px solid #C63E4E;
}

.stSelectbox div[data-baseweb="select"]{
background:#FFF9EC;
border-radius:12px;
border:2px solid #C63E4E;
}

.stSlider{
background:#FFD9A8;
padding:8px;
border-radius:15px;
}

/* Labels */

label{
font-weight:700 !important;
color:#7A1D22 !important;
}

/* Button */

div.stButton > button{
width:100%;
height:65px;
font-size:22px;
font-weight:bold;
border:none;
border-radius:15px;
background:linear-gradient(90deg,#C63E4E,#7A1D22);
color:white;
transition:.3s;
}

div.stButton > button:hover{
transform:scale(1.03);
box-shadow:0px 8px 25px rgba(0,0,0,.35);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------

st.markdown(
"""
<div class='main-title'>
🏦 Loan Default Risk Predictor
</div>

<div class='subtitle'>
Predict the probability that an applicant may default on a loan.
</div>
""",
unsafe_allow_html=True
)

# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:

    st.title("ℹ About")

    st.write("""
This application predicts the probability of loan default using a trained **Random Forest Classifier**.

### Features

- Loan Amount
- Annual Income
- Interest Rate
- DTI
- Loan Grade
- Home Ownership
- Loan Purpose
- FICO Score
- Employment Length

Developed using **Python**, **Scikit-Learn** and **Streamlit**.
""")

# ---------------------------
# INPUT CARD
# ---------------------------

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📋 Applicant Information")

col1,col2=st.columns(2)

with col1:

    loan_amnt=st.number_input(
        "Loan Amount",
        min_value=1000,
        value=10000,
        step=1000
    )

    annual_inc=st.number_input(
        "Annual Income",
        min_value=1000,
        value=50000,
        step=1000
    )

    int_rate=st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        value=12.0
    )

    dti=st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=15.0
    )

with col2:

    grade=st.selectbox(
        "Loan Grade",
        ["A","B","C","D","E","F","G"]
    )

    home_ownership=st.selectbox(
        "Home Ownership",
        ["RENT","OWN","MORTGAGE"]
    )

    purpose=st.selectbox(
        "Loan Purpose",
        [
            "debt_consolidation",
            "home",
            "education",
            "medical",
            "other"
        ]
    )

    fico=st.slider(
        "FICO Score",
        300,
        850,
        650
    )

    emp_length=st.slider(
        "Employment Length",
        0,
        40,
        5
    )

st.markdown("</div>", unsafe_allow_html=True)



# ----------------------------
# PREDICTION BUTTON
# ----------------------------
if st.button("Predict Risk"):

    # create input dataframe
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "int_rate": int_rate,
        "dti": dti,
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "fico_range_low": fico,
        "emp_length": emp_length
    }])

    # probability prediction
    prob = loaded_model.predict_proba(input_data)[:, 1][0]

    # final decision using threshold
    prediction = 1 if prob > 0.48 else 0

    # ----------------------------
    # OUTPUT SECTION
    # ----------------------------

    st.subheader("📊 Result")

    st.metric(
    label="Default Risk Probability",
    value=f"{prob * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠ HIGH RISK: Loan likely to default")
    else:
        st.success("✅ LOW RISK: Loan likely safe")

    # risk interpretation
    st.write("---")
    st.write("### Risk Interpretation")

    if prob > 0.7:
        st.write("🔴 Very High Risk")
    elif prob > 0.4:
        st.write("🟠 Medium Risk")
    else:
        st.write("🟢 Low Risk")