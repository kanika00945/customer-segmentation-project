import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Prediction Dashboard",
    layout="wide",
)

def inject_custom_css():
    st.markdown("""
    <style>
    .main {background-color: #0d1117; color: #c9d1d9;}
    [data-testid="stSidebar"] {background-color: #161b22 !important;}
    h1, h2, h3 {color: #58a6ff; border-bottom: 2px solid #21262d; padding-bottom: 5px;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #21262d, #2c313c);
        border-radius: 10px; padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stButton>button {background-color:#58a6ff; color:#0d1117; font-weight:bold;}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

model = joblib.load("kmeans_model.pkl")  
scaler = joblib.load("scaler.pkl")       
cluster_names = joblib.load("cluster_names.pkl")  

with st.sidebar:
    st.title("⚙ Settings Panel")
    st.markdown("---")
    st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    st.number_input("Average Spend", min_value=0, max_value=1000, value=50, step=1)
    st.number_input("Visits per week", min_value=1, max_value=7, value=2, step=1)
    st.number_input("Promotion Interest", min_value=0, max_value=10, value=5, step=1)
    st.markdown("---")
    st.info("Status: *Active* | Last Updated: 5 min ago")

st.title("✨ Customer Prediction Dashboard")
st.caption("Predict customer segments interactively with your inputs.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Customers", value="1.2K", delta="12%", delta_color="normal")
with col2:
    st.metric(label="Top Segment", value="Weekend Buyers")
with col3:
    st.metric(label="Revenue Goal", value="$50K", delta="+$5K")

st.markdown("---")
st.header("📈 Prediction Area")

with st.expander("Input Customer Details for Prediction"):
    age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    avg_spend = st.number_input("Average Spend", min_value=0, max_value=1000, value=50, step=1)
    visits = st.number_input("Visits per week", min_value=1, max_value=7, value=2, step=1)
    promo = st.number_input("Promotion Interest", min_value=0, max_value=10, value=5, step=1)
    
    if st.button("Predict Customer Segment"):
        input_data = np.array([[age, avg_spend, visits, promo]])
        if scaler:
            input_data = scaler.transform(input_data)
        cluster = model.predict(input_data)[0]
        cluster_label = cluster_names[cluster]
        st.success(f"Predicted Customer Segment: *{cluster_label}*")

