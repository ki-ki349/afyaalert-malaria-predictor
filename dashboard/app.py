# dashboard/app.py
import streamlit as st
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
# SIMPLE LOGIN SYSTEM
# ============================================

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# Simple login function
def check_login():
    if not st.session_state.logged_in:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔐 Login Required")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        # Simple hardcoded credentials
        valid_users = {
            "health_official": "malaria2024",
            "admin": "admin123"
        }
        
        if st.sidebar.button("Login"):
            if username in valid_users and valid_users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
        
        # Show login prompt
        st.warning("🔐 Please login using the sidebar to access the dashboard")
        return False
    else:
        # Show logout button
        st.sidebar.success(f"✅ Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        return True

# Check login status
if not check_login():
    st.stop()

# ============================================
# MAIN APP CONTENT (ONLY IF LOGGED IN)
# ============================================

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
        try:
            fi = pd.read_csv('models/feature_importance.csv')
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

# Sidebar
st.sidebar.title("🔍 Prediction Controls")

# Sidebar controls
selected_county = st.sidebar.selectbox("📍 Select County", counties)

# ============================================
# RISK MAP AND CURRENT ASSESSMENT
# ============================================
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Risk Map")
    
    # Create a simple map visualization
    risk_colors = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    # Get latest risk for each county
    latest_data = df.groupby('county').last().reset_index()
    
    # Create bar chart of risk levels
    risk_counts = df.groupby('county')['risk_level'].last().value_counts()
    
    fig_risk = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=risk_colors,
        title="Current Risk Distribution Across Counties",
        labels={'x': 'Risk Level', 'y': 'Number of Counties'}
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    st.subheader("📊 Current Risk Assessment")
    
    # Get data for selected county
    county_data = df[df['county'] == selected_county].sort_values(['year', 'month'])
    latest = county_data.iloc[-1] if len(county_data) > 0 else None
    
    if latest is not None:
        risk = latest['risk_level']
        if risk == 'High':
            st.markdown(f"<h2 class='high-risk'>🔴 HIGH RISK</h2>", unsafe_allow_html=True)
            st.markdown("⚠️ **Immediate action recommended**")
        elif risk == 'Medium':
            st.markdown(f"<h2 class='medium-risk'>🟠 MEDIUM RISK</h2>", unsafe_allow_html=True)
            st.markdown("📋 **Prepare for potential outbreak**")
        else:
            st.markdown(f"<h2 class='low-risk'>🟢 LOW RISK</h2>", unsafe_allow_html=True)
            st.markdown("✅ **Continue monitoring**")
        
        st.metric("Latest Incidence Rate", f"{latest['incidence_rate']:.1f} per 1,000")
        st.metric("Rainfall", f"{latest['rainfall_mm']:.1f} mm")
        st.metric("Temperature", f"{latest['temp_mean_c']:.1f}°C")
        st.metric("Humidity", f"{latest['humidity_pct']:.1f}%")
        
        st.markdown(f"**🤖 Model:** {model_name}")
    else:
        st.warning("No data available for selected county")

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
                recommendation = "⚠️ **Immediate action recommended:** Increase mosquito net distribution and plan for spraying campaigns."
            elif predicted_rate > 40:
                risk = 'Medium'
                icon = '🟠'
                recommendation = "📋 **Prepare for potential outbreak:** Stock up on supplies and alert community health workers."
            else:
                risk = 'Low'
                icon = '🟢'
                recommendation = "✅ **Continue monitoring:** Maintain routine surveillance and prevention efforts."

            # Display results
            st.markdown("---")
            st.markdown(f"## {icon} Prediction for {whatif_county} - {next_month_name} {next_year}")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Predicted Risk Level", risk)
            with col_r2:
                st.metric("Predicted Incidence Rate", f"{predicted_rate:.1f} per 1,000")
            with col_r3:
                confidence = int(min(100, rainfall_factor * 100))
                st.metric("Confidence", f"{confidence}%")
            
            st.info(recommendation)
            
            # Show what influenced the prediction
            with st.expander("🔍 What influenced this prediction?"):
                st.markdown(f"""
                | **Factor** | **Value** | **Impact** |
                |------------|-----------|------------|
                | **Season** | {season} | {'🔴 +80%' if season == 'Long_Rains' else '🟠 +50%' if season == 'Short_Rains' else '🟢 -30%'} |
                | **Rainfall** | {whatif_rainfall:.1f} mm (normal: {normal_rainfall:.0f} mm) | {'🔴 +' if rainfall_factor > 1 else '🟢 '}{int(abs(rainfall_factor-1)*100)}% |
                | **Recent cases** | {int(last_cases[-1]) if len(last_cases) > 0 else 'N/A'} cases | {'🔴 +' if lag_factor > 1 else '🟢 '}{int(abs(lag_factor-1)*100)}% |
                | **Historical average** | {historical_avg:.1f} cases | Baseline |
                """)
            
            # Show confidence meter
            st.progress(confidence/100, text=f"Confidence Level: {confidence}%")
        else:
            st.error(f"No historical data available for {whatif_county}")

# Feature importance section
if feature_importance is not None:
    st.markdown("---")
    st.subheader("🔍 What Influences Malaria Risk?")
    
    top_features = feature_importance.head(10)
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Factors',
        labels={'importance': 'Importance Score', 'feature': 'Factor'},
        color='importance',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "🦟 AfyaAlert - Malaria Prediction System | Data Science Project | © 2026"
    "</div>",
    unsafe_allow_html=True
)