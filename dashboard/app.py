# dashboard/app.py
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime

# ============================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================
st.set_page_config(
    page_title="AfyaAlert - Malaria Prediction",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# USER AUTHENTICATION
# ============================================

# User credentials
authenticator = stauth.Authenticate(
    {
        'usernames': {
            'health_official': {
                'name': 'Health Official',
                'password': 'malaria2024',
                'email': 'health@example.com'
            },
            'admin': {
                'name': 'Admin',
                'password': 'admin123',
                'email': 'admin@example.com'
            }
        }
    },
    'afyaalert_cookie',
    'random_cookie_key_12345',
    cookie_expiry_days=1
)

# Display login widget
name, authentication_status, username = authenticator.login('Login', 'main')

# Check login status
if authentication_status == False:
    st.error("❌ Username/password is incorrect")
    st.stop()

if authentication_status == None:
    st.warning("🔐 Please enter your username and password to continue")
    st.stop()

# ============================================
# MAIN APP CONTENT (ONLY IF LOGGED IN)
# ============================================
if authentication_status:

    # Add logout button to sidebar
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f"✅ Welcome, {name}!")

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #7f8c8d;
            text-align: center;
            margin-top: 0;
            margin-bottom: 2rem;
        }
        .high-risk { color: #e74c3c; font-weight: bold; }
        .medium-risk { color: #f39c12; font-weight: bold; }
        .low-risk { color: #27ae60; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🦟 AfyaAlert</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Malaria Outbreak Prediction System for Kenya</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png", width=100)
    st.sidebar.title("🔍 Prediction Controls")

    # ============================================
    # LOAD DATA AND MODELS
    # ============================================
    @st.cache_data
    def load_data():
        df = pd.read_csv('data/processed/real_kenya_county_dataset.csv')
        return df

    @st.cache_resource
    def load_model():
        try:
            model = joblib.load('models/xgboost_model.pkl')
            return model, "XGBoost"
        except:
            try:
                model = joblib.load('models/random_forest_model.pkl')
                return model, "Random Forest"
            except:
                return None, None

    @st.cache_data
    def load_feature_importance():
        try:
            fi = pd.read_csv('models/feature_importance_real.csv')
            return fi
        except:
            return None

    # Load everything
    df = load_data()
    model, model_name = load_model()
    feature_importance = load_feature_importance()

    if df is None:
        st.error("❌ Could not load dataset")
        st.stop()

    if model is None:
        st.warning("⚠️ No trained model found")
        st.stop()

    # Get unique counties
    counties = sorted(df['county'].unique())

    # Sidebar controls
    selected_county = st.sidebar.selectbox("📍 Select County", counties)

    # ============================================
    # RISK MAP AND CURRENT ASSESSMENT
    # ============================================
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🗺️ Risk Map")
        # Your map code here (simplified for brevity)
        st.info("Interactive map would show here")

    with col2:
        st.subheader("📊 Current Risk Assessment")
        st.info("Select a county to see risk assessment")

    # ============================================
    # WHAT-IF PREDICTION TOOL
    # ============================================
    st.markdown("---")
    st.subheader("🔮 What-If Prediction Tool")

    st.info("""
    **How it works:** Enter the current month's weather conditions, and the system will predict
    the malaria risk for the **next month**.
    """)

    col_wi1, col_wi2 = st.columns(2)

    with col_wi1:
        st.markdown("### 📍 Location & Time")
        whatif_county = st.selectbox("Select County", options=counties, key="whatif_county")
        whatif_month = st.selectbox(
            "Current Month",
            options=list(range(1, 13)),
            format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
            key="whatif_month"
        )
        whatif_year = st.number_input("Current Year", min_value=2024, max_value=2030, value=2025, key="whatif_year")

    with col_wi2:
        st.markdown("### 🌧️ Weather Input")
        whatif_rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=10.0)
        whatif_temp = st.number_input("Temperature (°C)", min_value=15.0, max_value=35.0, value=24.0, step=0.5)
        whatif_humidity = st.number_input("Humidity (%)", min_value=20.0, max_value=100.0, value=70.0, step=5.0)

    # Calculate next month
    next_month = whatif_month + 1 if whatif_month < 12 else 1
    next_month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][next_month-1]
    next_year = whatif_year + 1 if whatif_month == 12 else whatif_year

    st.markdown(f"### 📅 Predicting for: **{next_month_name} {next_year}**")

    if st.button("🔮 Predict Next Month's Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            # Get historical data for this county
            county_history = df[df['county'] == whatif_county].sort_values(['year', 'month'])

            if len(county_history) > 0:
                # Get historical average for this month
                historical_data = county_history[county_history['month'] == next_month]
                historical_avg = historical_data['incidence_rate'].mean() if len(historical_data) > 0 else county_history['incidence_rate'].mean()

                # Determine season factor
                if next_month in [3, 4, 5]:
                    season = 'Long_Rains'
                    seasonal_factor = 1.8
                elif next_month in [10, 11, 12]:
                    season = 'Short_Rains'
                    seasonal_factor = 1.5
                else:
                    season = 'Dry'
                    seasonal_factor = 0.7

                # Get normal rainfall
                normal_rainfall = historical_data['rainfall_mm'].mean() if len(historical_data) > 0 else 100
                rainfall_factor = max(0.5, min(2.0, whatif_rainfall / normal_rainfall))

                # Get last month cases
                county_history_monthly = county_history.sort_values(['year', 'month'])
                last_cases = county_history_monthly['incidence_rate'].tail(3).values
                lag_factor = last_cases[-1] / county_history['incidence_rate'].mean() if len(last_cases) > 0 else 1.0
                lag_factor = max(0.5, min(1.5, lag_factor))

                # Calculate predicted rate
                predicted_rate = historical_avg * rainfall_factor * seasonal_factor * lag_factor
                predicted_rate = max(10, min(200, predicted_rate))

                # Determine risk level
                if predicted_rate > 70:
                    risk = 'High'
                    icon = '🔴'
                    recommendation = "⚠️ **Immediate action recommended**"
                elif predicted_rate > 40:
                    risk = 'Medium'
                    icon = '🟠'
                    recommendation = "📋 **Prepare for potential outbreak**"
                else:
                    risk = 'Low'
                    icon = '🟢'
                    recommendation = "✅ **Continue monitoring**"

                # Display results
                st.markdown("---")
                st.markdown(f"## {icon} Prediction for {whatif_county} - {next_month_name} {next_year}")
                st.markdown(f"**Risk Level:** {risk}")
                st.markdown(f"**Predicted Incidence Rate:** {predicted_rate:.1f} per 1,000 people")
                st.info(recommendation)

            else:
                st.error(f"No historical data available for {whatif_county}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 1rem;'>"
        "🦟 AfyaAlert - Malaria Prediction System | Data Science Project | © 2026"
        "</div>",
        unsafe_allow_html=True
    )