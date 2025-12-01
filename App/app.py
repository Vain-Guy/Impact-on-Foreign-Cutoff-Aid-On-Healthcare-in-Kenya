import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os

# PAGE CONFIGURATION
st.set_page_config(
    page_title="U.S. Foreign Aid Funding Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ENHANCED CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Glassmorphism containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 3rem;
        color: white;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        will-change: transform;
    }

    @keyframes rotate {
        0% {transform: rotate(0deg);}
        100% {transform: rotate(360deg);}
    }

    .main-header h1 {
        margin: 0;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #a8b9ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }

    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }

    /* Section headers */
    .section-header {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 3rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .section-header::before {
        content: '';
        width: 5px;
        height: 35px;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Metrics styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 18px;
        padding: 1.8rem 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 45px rgba(102, 126, 234, 0.4);
        background: rgba(255, 255, 255, 0.18);
    }

    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* Inputs (Selectbox, NumberInput, Radio) */
    .stSelectbox, .stNumberInput, .stRadio {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
        color: white !important;
        position: relative;
    }

    .stSelectbox:hover, .stNumberInput:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* Force input text and placeholder visible */
    .stSelectbox div[role="combobox"] > div,
    .stSelectbox span,
    .stNumberInput > div > input,
    .stRadio > div {
        color: white !important;
        font-weight: 500;
    }

    input::placeholder {
        color: rgba(255,255,255,0.5) !important;
    }

    input, select {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 0.8rem !important;
    }

    /* Hide native dropdown arrow */
    .stSelectbox div[role="combobox"] > div > svg {
        display: none !important;
    }

    /* Custom arrow */
    .stSelectbox div[role="combobox"]::after {
        content: '▾';
        color: white;
        position: absolute;
        right: 1rem;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 1.2rem 3rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        margin-top: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 45px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Success box */
    .success-box {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.3) 0%, rgba(56, 239, 125, 0.3) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(56, 239, 125, 0.5);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 50px rgba(56, 239, 125, 0.3);
        animation: slideIn 0.6s ease-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .success-box strong {
        font-size: 2rem;
        display: block;
        margin-top: 1rem;
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Chart container */
    .stAltairChart {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2.5rem;
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
    }

    .footer strong {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.3rem;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 3rem 0;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2.2rem; }
        .glass-container { padding: 1rem; }
        .section-header { font-size: 1.6rem; }
        .stButton > button { font-size: 1.1rem; padding: 1rem 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# LOAD DATA AND MODEL
@st.cache_data
def load_data():
    if not os.path.exists("modeling data.csv"):
        st.error("❌ Missing 'modeling data.csv'")
        st.stop()
    return pd.read_csv("modeling data.csv")

@st.cache_resource
def load_model():
    try:
        return joblib.load("best_xgb_pipeline.pkl")
    except FileNotFoundError:
        st.error("❌ Missing 'best_xgb_pipeline.pkl'")
        return None

df = load_data()
model_pipeline = load_model()

# HEADER
st.markdown("""
<div class="main-header">
    <h1>💼 U.S. Foreign Aid Funding Predictor</h1>
    <p>Advanced machine learning forecasting for strategic resource allocation in Kenya's socio-economic sectors</p>
</div>
""", unsafe_allow_html=True)

# OVERVIEW DASHBOARD
st.markdown('<p class="section-header">📊 Overview Dashboard</p>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("📁 Total Records", f"{len(df):,}")
col2.metric("🏛️ Agencies", f"{df['managing_agency_name'].nunique()}")
col3.metric("🎯 Sectors", f"{df['sector'].nunique()}")
col4.metric("💰 Total Funding", f"${df['constant_dollar_amount'].sum()/1e9:.1f}B")

# INPUTS & PREDICTION
st.markdown('<p class="section-header">🔮 Prediction Engine</p>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        fiscal_year = st.number_input("Fiscal Year", 2000, 2100, 2025)
        managing_agency_name = st.selectbox("Managing Agency", sorted(df['managing_agency_name'].unique()))
    with col2:
        funding_agency_name = st.selectbox("Funding Agency", sorted(df['funding_agency_name'].unique()))
        sector = st.selectbox("Sector", sorted(df['sector'].unique()))
    is_refund = st.radio("Refund Status", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

def compute_features(df, year, agency, fund_agency, sector, refund):
    df_filtered = df[(df['managing_agency_name']==agency) & (df['funding_agency_name']==fund_agency) & (df['sector']==sector)].sort_values('fiscal_year')
    lag_1 = df_filtered.loc[df_filtered['fiscal_year']==year-1,'constant_dollar_amount'].values
    lag_2 = df_filtered.loc[df_filtered['fiscal_year']==year-2,'constant_dollar_amount'].values
    lag_1 = lag_1[0] if len(lag_1)>0 else 0
    lag_2 = lag_2[0] if len(lag_2)>0 else 0
    last_3 = df_filtered[df_filtered['fiscal_year'].isin([year-1,year-2,year-3])]
    rolling_mean_3yr = last_3['constant_dollar_amount'].mean() if not last_3.empty else 0
    rolling_std_3yr = last_3['constant_dollar_amount'].std() if not last_3.empty else 0
    funding_growth_rate = ((lag_1-lag_2)/lag_2) if lag_2!=0 else 0
    transaction_type_name = df_filtered['transaction_type_name'].mode().values[0] if not df_filtered.empty else 'Grant'
    return pd.DataFrame([{"fiscal_year":year,"is_refund":refund,"managing_agency_name":agency,"funding_agency_name":fund_agency,"sector":sector,"lag_1":lag_1,"lag_2":lag_2,"rolling_mean_3yr":rolling_mean_3yr,"rolling_std_3yr":rolling_std_3yr,"funding_growth_rate":funding_growth_rate,"transaction_type_name":transaction_type_name}])

if st.button("🚀 Generate Forecast"):
    if model_pipeline is not None:
        input_data = compute_features(df,fiscal_year,managing_agency_name,funding_agency_name,sector,is_refund)
        pred_log = model_pipeline.predict(input_data)[0]
        pred = max(0, np.expm1(pred_log))
        st.markdown(f"""<div class="success-box">💰 Predicted Allocation for {sector}<br><strong>${pred:,.2f}</strong><br><small>Fiscal Year {fiscal_year}</small></div>""", unsafe_allow_html=True)
    else:
        st.error("Model not loaded — unable to generate forecast.")

# HISTORICAL TRENDS
st.markdown('<p class="section-header">📈 Historical Trends Analysis</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    agency_sel = st.selectbox("Select Agency", sorted(df['managing_agency_name'].unique()), key='trend_agency')
with col2:
    sector_sel = st.selectbox("Select Sector", sorted(df['sector'].unique()), key='trend_sector')

df_plot = df[(df['managing_agency_name']==agency_sel)&(df['sector']==sector_sel)]

if df_plot.empty:
    st.info("📊 No data available for this combination.")
else:
    chart = alt.Chart(df_plot).mark_line(point=True, color='#667eea', strokeWidth=3).encode(
        x=alt.X('fiscal_year:O', title='Fiscal Year'),
        y=alt.Y('constant_dollar_amount:Q', title='Funding Amount (USD)'),
        tooltip=['fiscal_year','constant_dollar_amount']
    ).properties(height=450, title=f"Funding Trends: {agency_sel} - {sector_sel}")
    st.altair_chart(chart, use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown('<div class="footer"><p><strong>Developed by Ahjin Analytics</strong><br>Empowering data-driven decisions for global impact</p></div>', unsafe_allow_html=True)