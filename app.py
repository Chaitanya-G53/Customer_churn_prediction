import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Resolve paths relative to this script's own folder, not the process's
# working directory (Streamlit Cloud does not guarantee cwd == app folder).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #cfd8dc;
        font-size: 1.05rem;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 18px 22px;
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 14px;
    }
    .badge-risk {
        display:inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        background: rgba(255, 82, 82, 0.18);
        color: #ff5252;
        border: 1px solid rgba(255,82,82,0.4);
    }
    .badge-safe {
        display:inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        background: rgba(46, 204, 113, 0.18);
        color: #2ecc71;
        border: 1px solid rgba(46,204,113,0.4);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141e30, #243b55);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.4em;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 18px rgba(255,94,98,0.6);
    }
    h1, h2, h3, h4 {
        color: #e3f2fd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(
            f"❌ Could not find `model.pkl` at:\n\n`{path}`\n\n"
            "Make sure `model.pkl` is committed to the same GitHub repo folder as `app.py` "
            "(same directory, exact filename `model.pkl`, not `.gitignore`d, and under GitHub's "
            "100MB file size limit)."
        )
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model()

FEATURES = list(getattr(model, "feature_names_in_", [
    "credit_score", "en_country", "en_gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary"
]))

COUNTRY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.markdown('<div class="main-title">🏦 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by a K-Nearest Neighbors classifier — will this customer stay or leave?</div>',
    unsafe_allow_html=True,
)
st.caption(
    "⚠️ Country and Gender are label-encoded in the model as France=0, Germany=1, Spain=2 "
    "and Female=0, Male=1 — this is the standard mapping for this dataset. If your training "
    "encoding differs, adjust `COUNTRY_MAP` / `GENDER_MAP` in the code."
)

# --------------------------------------------------------------------------------
# SIDEBAR — INPUTS
# --------------------------------------------------------------------------------
st.sidebar.header("🧾 Customer Profile")
st.sidebar.caption("Fill in the customer's details")

credit_score = st.sidebar.slider("💳 Credit Score", 300, 850, 650, 1)
country = st.sidebar.selectbox("🌍 Country", list(COUNTRY_MAP.keys()))
gender = st.sidebar.selectbox("🚻 Gender", list(GENDER_MAP.keys()))
age = st.sidebar.slider("🎂 Age", 18, 92, 35, 1)
tenure = st.sidebar.slider("📅 Tenure (years with bank)", 0, 10, 3, 1)
balance = st.sidebar.number_input("💰 Account Balance", min_value=0.0, max_value=300000.0, value=50000.0, step=500.0)
products_number = st.sidebar.selectbox("📦 Number of Products", [1, 2, 3, 4])
credit_card = st.sidebar.radio("💳 Has Credit Card?", ["Yes", "No"], horizontal=True)
active_member = st.sidebar.radio("⚡ Active Member?", ["Yes", "No"], horizontal=True)
estimated_salary = st.sidebar.number_input("💵 Estimated Salary", min_value=0.0, max_value=250000.0, value=75000.0, step=500.0)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🚀 Predict Churn")
st.sidebar.markdown("---")
st.sidebar.info(
    f"Model: **K-Nearest Neighbors Classifier**\n\n"
    f"Neighbors used (k): **{getattr(model, 'n_neighbors', 5)}**\n\n"
    f"Trained on **{getattr(model, 'n_samples_fit_', 'N/A')}** samples"
)

# --------------------------------------------------------------------------------
# BUILD INPUT ROW (must match feature_names_in_ order)
# --------------------------------------------------------------------------------
raw_values = {
    "credit_score": credit_score,
    "en_country": COUNTRY_MAP[country],
    "en_gender": GENDER_MAP[gender],
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": 1 if credit_card == "Yes" else 0,
    "active_member": 1 if active_member == "Yes" else 0,
    "estimated_salary": estimated_salary,
}
input_df = pd.DataFrame([[raw_values[f] for f in FEATURES]], columns=FEATURES)

display_df = pd.DataFrame({
    "Field": ["Credit Score", "Country", "Gender", "Age", "Tenure (yrs)", "Balance",
              "Products", "Has Credit Card", "Active Member", "Estimated Salary"],
    "Value": [credit_score, country, gender, age, tenure, f"${balance:,.2f}",
              products_number, credit_card, active_member, f"${estimated_salary:,.2f}"],
})

# --------------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------------
col_left, col_right = st.columns([1.05, 1])

with col_left:
    st.markdown("#### 📥 Customer Snapshot")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 📊 Quick Stats")
    c1, c2 = st.columns(2)
    c1.metric("Balance", f"${balance:,.0f}")
    c2.metric("Salary", f"${estimated_salary:,.0f}")
    c3, c4 = st.columns(2)
    c3.metric("Age", age)
    c4.metric("Tenure", f"{tenure} yrs")

with col_right:
    st.markdown("#### 🔮 Prediction")

    if predict_clicked or "last_pred" not in st.session_state:
        pred_class = int(model.predict(input_df.values)[0])
        try:
            proba = model.predict_proba(input_df.values)[0]
            churn_prob = float(proba[list(model.classes_).index(1)])
        except Exception:
            churn_prob = float(pred_class)
        st.session_state["last_pred"] = pred_class
        st.session_state["last_prob"] = churn_prob
    else:
        pred_class = st.session_state["last_pred"]
        churn_prob = st.session_state["last_prob"]

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if pred_class == 1:
        st.markdown('<span class="badge-risk">⚠️ Likely to Churn</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-safe">✅ Likely to Stay</span>', unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='margin-top:14px; margin-bottom:0; color:#ffab91;'>{churn_prob*100:.1f}%</h1>"
        f"<p style='color:#b0bec5;'>Estimated Churn Probability</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**Churn Risk Level**")
    st.progress(min(max(churn_prob, 0.0), 1.0))

    proba_df = pd.DataFrame({
        "Outcome": ["Stay", "Churn"],
        "Probability": [1 - churn_prob, churn_prob],
    }).set_index("Outcome")
    st.bar_chart(proba_df, use_container_width=True, color="#ff5e62")

    if churn_prob >= 0.6:
        st.error("🔴 High risk — consider proactive retention outreach (offers, check-ins, loyalty perks).")
    elif churn_prob >= 0.35:
        st.warning("🟡 Moderate risk — keep an eye on engagement and satisfaction.")
    else:
        st.success("🟢 Low risk — this customer looks stable.")

# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#78909c; font-size:0.85rem;'>"
    "Built with ❤️ using Streamlit &nbsp;|&nbsp; Model: KNeighborsClassifier (scikit-learn)"
    "</p>",
    unsafe_allow_html=True,
)
