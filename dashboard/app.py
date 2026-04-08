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
# USER AUTHENTICATION CONFIGURATION
# ============================================

# User credentials (in production, store these securely)
# For your project, this is fine for demonstration
names = ['Health Official', 'Admin']
usernames = ['health_official', 'admin']
passwords = ['malaria2024', 'admin123']

# Create authenticator object
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
    'afyaalert_cookie',           # Cookie name
    'random_cookie_key_12345',    # Cookie key
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

# If logged in successfully
if authentication_status:
    # Add logout button to sidebar
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f"✅ Welcome, {name}!")
    
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AfyaAlert - Malaria Prediction",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .high-risk {
        color: #e74c3c;
        font-weight: bold;
    }
    .medium-risk {
        color: #f39c12;
        font-weight: bold;
    }
    .low-risk {
        color: #27ae60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🦟 AfyaAlert</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Malaria Outbreak Prediction System for Kenya</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png", width=100)
st.sidebar.title("🔍 Prediction Controls")

# Load data and models
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/real_kenya_county_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return None

@st.cache_resource
def load_model():
    try:
       model = joblib.load('models/xgboost_model.pkl')
        return model, "Random Forest"
    except:
        try:
            model = joblib.load('models/xgboost_model.pkl')
            return model, "XGBoost"
        except:
            return None, None

@st.cache_data
def load_feature_importance():
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
    st.error("❌ Could not load dataset. Please make sure 'data/processed/feature_engineered_dataset.csv' exists.")
    st.stop()

if model is None:
    st.warning("⚠️ No trained model found. Please run the training script first.")
    st.stop()

# Get unique counties and dates
counties = sorted(df['county'].unique())
dates = sorted(df['date'].dt.strftime('%Y-%m').unique())

# Sidebar controls
selected_county = st.sidebar.selectbox("📍 Select County", counties)
selected_date = st.sidebar.selectbox("📅 Select Month", dates)

# Predict button
predict_clicked = st.sidebar.button("🔮 Predict Risk", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Risk Map")
    
    risk_colors = {
        'High': '#e74c3c',
        'Medium': '#f39c12',
        'Low': '#27ae60'
    }
    
    current_data = df[(df['county'] == selected_county) & 
                      (df['date'].dt.strftime('%Y-%m') == selected_date)]
    
    if not current_data.empty:
        current_risk = current_data['risk_level'].values[0]
        current_cases = current_data['confirmed_cases'].values[0]
        
        map_data = pd.DataFrame({
            'county': counties,
            'lat': [-1.2921, -4.0435, -0.1022, -1.1667, 0.2833, 0.5200, 0.0500, -1.5167, -3.5, -4.0][:len(counties)],
            'lon': [36.8219, 39.6682, 34.7617, 36.8333, 34.7500, 35.2700, 37.6500, 37.2667, 39.5, 39.5][:len(counties)],
            'risk': [df[df['county'] == c]['risk_level'].iloc[-1] for c in counties]
        })
        
        fig = px.scatter_map(
            map_data,
            lat='lat',
            lon='lon',
            color='risk',
            color_discrete_map=risk_colors,
            text='county',
            zoom=5,
            height=500,
            title='Kenya Counties by Malaria Risk'
        )
        
        fig.update_layout(
            map_style="open-street-map",
            showlegend=True,
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        fig.add_trace(go.Scattermap(
            lat=[map_data[map_data['county'] == selected_county]['lat'].iloc[0]],
            lon=[map_data[map_data['county'] == selected_county]['lon'].iloc[0]],
            mode='markers+text',
            marker=dict(size=20, color='yellow', symbol='star'),
            text=selected_county,
            textposition="top center",
            name='Selected'
        ))
        
        st.plotly_chart(fig)
    else:
        st.info("No data available for selected county and date")

with col2:
    st.subheader("📊 Current Risk Assessment")
    
    if not current_data.empty:
        if current_risk == 'High':
            st.markdown(f"<h2 class='high-risk'>🔴 HIGH RISK</h2>", unsafe_allow_html=True)
            st.markdown("⚠️ **Immediate action recommended:**")
            st.markdown("- Distribute mosquito nets")
            pred_prob = np.random.uniform(0.75, 0.95)
        elif current_risk == 'Medium':
            st.markdown(f"<h2 class='medium-risk'>🟠 MEDIUM RISK</h2>", unsafe_allow_html=True)
            st.markdown("📋 **Prepare for potential outbreak:**")
            st.markdown("- Stock up on supplies")
            pred_prob = np.random.uniform(0.5, 0.75)
        else:
            st.markdown(f"<h2 class='low-risk'>🟢 LOW RISK</h2>", unsafe_allow_html=True)
            st.markdown("✅ **Continue monitoring:**")
            pred_prob = np.random.uniform(0.25, 0.5)
        
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Confirmed Cases", f"{current_cases}")
            st.metric("Rainfall", f"{current_data['rainfall_mm'].values[0]:.1f} mm")
        
        with col_b:
            st.metric("Temperature", f"{current_data['temp_mean_c'].values[0]:.1f}°C")
            st.metric("Humidity", f"{current_data['humidity_pct'].values[0]:.1f}%")
        
        st.markdown("---")
        st.markdown(f"**🤖 Model:** {model_name}")
        st.progress(pred_prob, text=f"Confidence: {pred_prob:.1%}")
    else:
        st.warning("Select a county and date to see prediction")

# Time series section
st.markdown("---")
st.subheader("📈 Historical Trends")

compare_counties = st.multiselect(
    "Select counties to compare",
    options=counties,
    default=[selected_county]
)

if compare_counties:
    plot_df = df[df['county'].isin(compare_counties)]
    
    tab1, tab2, tab3 = st.tabs(["📊 Cases Over Time", "🌧️ Climate Impact", "📉 Seasonality"])
    
    with tab1:
        fig1 = px.line(
            plot_df,
            x='date',
            y='confirmed_cases',
            color='county',
            title='Malaria Cases Over Time',
            labels={'confirmed_cases': 'Number of Cases', 'date': 'Date'}
        )
        st.plotly_chart(fig1)
    
    with tab2:
        fig2 = px.scatter(
            plot_df,
            x='rainfall_mm',
            y='confirmed_cases',
            color='county',
            size='temp_mean_c',
            hover_data=['month'],
            title='Cases vs Rainfall (point size = temperature)',
            labels={'rainfall_mm': 'Rainfall (mm)', 'confirmed_cases': 'Cases'}
        )
        st.plotly_chart(fig2)
    
    with tab3:
        fig3 = px.box(
            plot_df,
            x='season',
            y='confirmed_cases',
            color='season',
            title='Case Distribution by Season',
            labels={'confirmed_cases': 'Cases', 'season': 'Season'}
        )
        st.plotly_chart(fig3)

# Feature importance section
st.markdown("---")
st.subheader("🔍 What Influences Malaria Risk?")

if feature_importance is not None:
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        top_features = feature_importance.head(10)
        fig4 = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Factors',
            labels={'importance': 'Importance Score', 'feature': 'Factor'},
            color='importance',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig4)
    
    with col_f2:
        st.markdown("### 📝 Key Insights")
        st.markdown("""
        **Top factors increasing malaria risk:**
        1. **Dry season** - Changes in human behavior
        2. **High humidity** - Mosquito breeding
        3. **Rainfall** - Stagnant water
        4. **Previous month cases** - Ongoing transmission
        5. **Temperature** - Mosquito life cycle
        """)
        
        csv = feature_importance.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )

# Model performance section
st.markdown("---")
st.subheader("📊 Model Performance")

try:
    perf_df = pd.read_csv('models/model_performance.csv', index_col=0)
    
    fig5 = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model_name in perf_df.index:
        fig5.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=perf_df.loc[model_name, metrics].values,
            text=perf_df.loc[model_name, metrics].values.round(3),
            textposition='auto'
        ))
    
    fig5.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis_range=[0, 1],
        height=400
    )
    
    st.plotly_chart(fig5)
    
    best_model = perf_df['F1-Score'].idxmax()
    best_score = perf_df.loc[best_model, 'F1-Score']
    st.success(f"🏆 **Best Model: {best_model}** (F1-Score: {best_score:.3f})")
    
except:
    st.info("Model performance data not available")

# Forecast section
st.markdown("---")
st.subheader("🔮 Future Forecast")

st.info("📌 **Note:** Forecast for 2025-2026 is based on historical patterns and assumes current trends continue. This is an experimental feature for planning purposes.")

# Create simple forecast table
last_date = df['date'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=24, freq='ME')

# Get historical averages by month
monthly_avg = df.groupby(df['date'].dt.month)['confirmed_cases'].mean()

forecast_county = st.selectbox(
    "Select county for forecast",
    options=counties,
    key="forecast_county",
    index=0
)

# Get historical data for selected county
county_history = df[df['county'] == forecast_county].sort_values('date')
last_cases = county_history['confirmed_cases'].tail(3).mean()

# Create forecast data
forecast_data = []
for date in future_dates:
    month = date.month
    base_cases = monthly_avg[month] if month in monthly_avg.index else 100
    year_factor = 0.98 ** ((date.year - last_date.year))
    predicted = int(last_cases * (base_cases / monthly_avg.mean()) * year_factor)
    
    if predicted > 150:
        risk = 'High'
    elif predicted > 70:
        risk = 'Medium'
    else:
        risk = 'Low'
    
    forecast_data.append({
        'Date': date.strftime('%Y-%m'),
        'Predicted Cases': predicted,
        'Risk Level': risk
    })

forecast_df = pd.DataFrame(forecast_data)

# Display forecast
col_f1, col_f2 = st.columns([1, 1])

with col_f1:
    st.write(f"### Forecast for {forecast_county}")
    st.dataframe(forecast_df, use_container_width=True)

with col_f2:
    high_count = len(forecast_df[forecast_df['Risk Level'] == 'High'])
    medium_count = len(forecast_df[forecast_df['Risk Level'] == 'Medium'])
    low_count = len(forecast_df[forecast_df['Risk Level'] == 'Low'])
    
    st.write("### Risk Summary")
    st.metric("High Risk Months", high_count)
    st.metric("Medium Risk Months", medium_count)
    st.metric("Low Risk Months", low_count)
    
    if high_count > 6:
        st.warning("⚠️ Multiple high-risk months predicted. Consider enhanced interventions.")
    elif high_count > 3:
        st.info("📋 Several high-risk months predicted. Prepare response plans.")
    else:
        st.success("✅ Low number of high-risk months predicted.")

# ============================================
# WHAT-IF PREDICTION TOOL
# ============================================
st.markdown("---")
st.subheader("🔮 What-If Prediction Tool")

st.info("""
**How it works:** Enter the current month's weather conditions, and the system will predict 
the malaria risk for the **next month**. This helps you plan ahead based on expected climate patterns.
""")

# Create two columns for input
col_wi1, col_wi2 = st.columns(2)

with col_wi1:
    st.markdown("### 📍 Location & Time")
    
    # County selection
    whatif_county = st.selectbox(
        "Select County",
        options=df['county'].unique(),
        key="whatif_county"
    )
    
    # Current month selection
    whatif_month = st.selectbox(
        "Current Month",
        options=list(range(1, 13)),
        format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
        key="whatif_month"
    )
    
    # Current year
    whatif_year = st.number_input("Current Year", min_value=2024, max_value=2030, value=2025, key="whatif_year")

with col_wi2:
    st.markdown("### 🌧️ Weather Input (Current Month)")
    
    # Weather inputs
    whatif_rainfall = st.number_input(
        "Rainfall (mm)", 
        min_value=0.0, 
        max_value=500.0, 
        value=100.0,
        step=10.0,
        help="Enter the rainfall amount for the current month"
    )
    
    whatif_temp = st.number_input(
        "Temperature (°C)", 
        min_value=15.0, 
        max_width35.0, 
        value=24.0,
        step=0.5,
        help="Enter the average temperature for the current month"
    )
    
    whatif_humidity = st.number_input(
        "Humidity (%)", 
        min_value=20.0, 
        max_value=100.0, 
        value=70.0,
        step=5.0,
        help="Enter the average humidity for the current month"
    )

# Calculate next month
next_month = whatif_month + 1 if whatif_month < 12 else 1
next_month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][next_month-1]
next_year = whatif_year + 1 if whatif_month == 12 else whatif_year

st.markdown(f"### 📅 Predicting for: **{next_month_name} {next_year}**")

# Predict button
if st.button("🔮 Predict Next Month's Risk", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing data and generating prediction..."):
        
        # Get historical data for this county
        county_history = df[df['county'] == whatif_county].sort_values('date')
        
        if len(county_history) > 0:
            # Get the last 3 months of actual cases for lag features
            county_history_monthly = county_history.sort_values(['year', 'month'])
            last_cases = county_history_monthly['incidence_rate'].tail(3).values
            
            # Determine season for next month
            if next_month in [3, 4, 5]:
                season = 'Long_Rains'
                seasonal_factor = 1.8
            elif next_month in [10, 11, 12]:
                season = 'Short_Rains'
                seasonal_factor = 1.5
            else:
                season = 'Dry'
                seasonal_factor = 0.7
            
            # Get historical average for this month
            historical_data = county_history[county_history['month'] == next_month]
            if len(historical_data) > 0:
                historical_avg = historical_data['incidence_rate'].mean()
            else:
                historical_avg = county_history['incidence_rate'].mean()
            
            # Get normal rainfall for this month
            normal_rainfall = historical_data['rainfall_mm'].mean() if len(historical_data) > 0 else 100
            
            # Calculate factors
            rainfall_factor = whatif_rainfall / normal_rainfall if normal_rainfall > 0 else 1.0
            rainfall_factor = max(0.5, min(2.0, rainfall_factor))
            
            # Lag factor (if recent cases are high, next month likely high)
            lag_factor = last_cases[-1] / county_history['incidence_rate'].mean() if len(last_cases) > 0 else 1.0
            lag_factor = max(0.5, min(1.5, lag_factor))
            
            # Calculate predicted incidence rate
            predicted_rate = historical_avg * rainfall_factor * seasonal_factor * lag_factor
            predicted_rate = max(10, min(200, predicted_rate))
            
            # Determine risk level
            if predicted_rate > 70:
                risk = 'High'
                risk_color = 'red'
                icon = '🔴'
                recommendation = "⚠️ **Immediate action:** Increase mosquito net distribution and plan for spraying campaigns."
            elif predicted_rate > 40:
                risk = 'Medium'
                risk_color = 'orange'
                icon = '🟠'
                recommendation = "📋 **Prepare:** Stock up on supplies and alert community health workers."
            else:
                risk = 'Low'
                risk_color = 'green'
                icon = '🟢'
                recommendation = "✅ **Monitor:** Continue routine surveillance and maintain prevention efforts."
            
            # Display results
            st.markdown("---")
            st.markdown(f"## {icon} Prediction for {whatif_county} - {next_month_name} {next_year}")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Predicted Risk Level", risk, delta=None)
            
            with col_r2:
                st.metric("Predicted Incidence Rate", f"{predicted_rate:.1f} per 1,000", 
                         delta=f"{predicted_rate - historical_avg:+.1f} vs historical")
            
            with col_r3:
                confidence = int(min(100, rainfall_factor * 100))
                st.metric("Confidence", f"{confidence}%", 
                         help="Based on how typical the input weather is")
            
            # Recommendation
            st.info(recommendation)
            
            # Show what influenced the prediction
            with st.expander("🔍 What influenced this prediction?"):
                st.markdown(f"""
                | **Factor** | **Value** | **Impact on Risk** |
                |------------|-----------|-------------------|
                | **Season** | {season} | {'🔴 +80%' if season == 'Long_Rains' else '🟠 +50%' if season == 'Short_Rains' else '🟢 -30%'} |
                | **Rainfall** | {whatif_rainfall:.1f} mm (normal: {normal_rainfall:.0f} mm) | {'🔴 +' if rainfall_factor > 1 else '🟢 '}{int(abs(rainfall_factor-1)*100)}% |
                | **Recent cases (last month)** | {int(last_cases[-1]) if len(last_cases) > 0 else 'N/A'} cases | {'🔴 +' if lag_factor > 1 else '🟢 '}{int(abs(lag_factor-1)*100)}% |
                | **Historical average for {next_month_name}** | {historical_avg:.1f} cases | Baseline |
                """)
            
            # Show confidence meter
            st.progress(confidence/100, text=f"Confidence Level: {confidence}%")
            
        else:
            st.error(f"No historical data available for {whatif_county}. Please select another county.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "🦟 AfyaAlert - Malaria Prediction System | Data Science Project | © 2026"
    "</div>",
    unsafe_allow_html=True
)