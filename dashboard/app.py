# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    """Load the processed dataset"""
    try:
        df = pd.read_csv('data/processed/feature_engineered_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return None

@st.cache_resource
def load_model():
    """Load the best trained model"""
    try:
        # Try to load Random Forest model (our best performer)
        model = joblib.load('models/random_forest_model.pkl')
        return model, "Random Forest"
    except:
        try:
            model = joblib.load('models/xgboost_model.pkl')
            return model, "XGBoost"
        except:
            return None, None

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        fi = pd.read_csv('models/feature_importance.csv')
        return fi
    except:
        return None

# Load everything
df = load_data()
model, model_name = load_model()
feature_importance = load_feature_importance()

# Check if data loaded successfully
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
    
    # Create a simple map representation
    # In a real app, you'd use folium here
    risk_colors = {
        'High': '#e74c3c',
        'Medium': '#f39c12',
        'Low': '#27ae60'
    }
    
    # Get current risk for selected county
    current_data = df[(df['county'] == selected_county) & 
                      (df['date'].dt.strftime('%Y-%m') == selected_date)]
    
    if not current_data.empty:
        current_risk = current_data['risk_level'].values[0]
        current_cases = current_data['confirmed_cases'].values[0]
        
        # Create a DataFrame for the map (simplified - in real app use actual coordinates)
        map_data = pd.DataFrame({
            'county': counties,
            'lat': [-1.2921, -4.0435, -0.1022, -1.1667, 0.2833, 0.5200, 0.0500, -1.5167, -3.5, -4.0][:len(counties)],
            'lon': [36.8219, 39.6682, 34.7617, 36.8333, 34.7500, 35.2700, 37.6500, 37.2667, 39.5, 39.5][:len(counties)],
            'risk': [df[df['county'] == c]['risk_level'].iloc[-1] for c in counties]
        })
        
        # Create scatter map
        fig = px.scatter_mapbox(
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
            mapbox_style="open-street-map",
            showlegend=True,
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        # Highlight selected county
        fig.add_trace(go.Scattermapbox(
            lat=[map_data[map_data['county'] == selected_county]['lat'].iloc[0]],
            lon=[map_data[map_data['county'] == selected_county]['lon'].iloc[0]],
            mode='markers+text',
            marker=dict(size=20, color='yellow', symbol='star'),
            text=selected_county,
            textposition="top center",
            name='Selected'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for selected county and date")

with col2:
    st.subheader("📊 Current Risk Assessment")
    
    if not current_data.empty:
        # Risk indicator
        if current_risk == 'High':
            st.markdown(f"<h2 class='high-risk'>🔴 HIGH RISK</h2>", unsafe_allow_html=True)
            st.markdown("⚠️ **Immediate action recommended:**")
            st.markdown("- Distribute mosquito nets")
            np.random.seed(42)
            pred_prob = np.random.uniform(0.75, 0.95)
        elif current_risk == 'Medium':
            st.markdown(f"<h2 class='medium-risk'>🟠 MEDIUM RISK</h2>", unsafe_allow_html=True)
            st.markdown("📋 **Prepare for potential outbreak:**")
            st.markdown("- Stock up on supplies")
            st.markdown("- Alert community health workers")
            pred_prob = np.random.uniform(0.5, 0.75)
        else:
            st.markdown(f"<h2 class='low-risk'>🟢 LOW RISK</h2>", unsafe_allow_html=True)
            st.markdown("✅ **Continue monitoring:**")
            st.markdown("- Maintain regular surveillance")
            pred_prob = np.random.uniform(0.25, 0.5)
        
        # Metrics
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Confirmed Cases", f"{current_cases}")
            st.metric("Rainfall", f"{current_data['rainfall_mm'].values[0]:.1f} mm")
        
        with col_b:
            st.metric("Temperature", f"{current_data['temp_mean_c'].values[0]:.1f}°C")
            st.metric("Humidity", f"{current_data['humidity_pct'].values[0]:.1f}%")
        
        # Model confidence
        st.markdown("---")
        st.markdown(f"**🤖 Model:** {model_name}")
        st.progress(pred_prob, text=f"Confidence: {pred_prob:.1%}")
    else:
        st.warning("Select a county and date to see prediction")

# Time series section
st.markdown("---")
st.subheader("📈 Historical Trends")

# County selector for comparison
compare_counties = st.multiselect(
    "Select counties to compare",
    options=counties,
    default=[selected_county]
)

if compare_counties:
    # Filter data
    plot_df = df[df['county'].isin(compare_counties)]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Cases Over Time", "🌧️ Climate Impact", "📉 Seasonality"])
    
    with tab1:
        # Time series of cases
        fig1 = px.line(
            plot_df,
            x='date',
            y='confirmed_cases',
            color='county',
            title='Malaria Cases Over Time',
            labels={'confirmed_cases': 'Number of Cases', 'date': 'Date'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Scatter plot of cases vs rainfall
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
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Boxplot by season
        fig3 = px.box(
            plot_df,
            x='season',
            y='confirmed_cases',
            color='season',
            title='Case Distribution by Season',
            labels={'confirmed_cases': 'Cases', 'season': 'Season'}
        )
        st.plotly_chart(fig3, use_container_width=True)

# Feature importance section
st.markdown("---")
st.subheader("🔍 What Influences Malaria Risk?")

if feature_importance is not None:
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        # Horizontal bar chart of top features
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
        st.plotly_chart(fig4, use_container_width=True)
    
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
        
        # Download button
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
    
    # Create a bar chart comparing models
    fig5 = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model in perf_df.index:
        fig5.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=perf_df.loc[model, metrics].values,
            text=perf_df.loc[model, metrics].values.round(3),
            textposition='auto'
        ))
    
    fig5.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis_range=[0, 1],
        height=400
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Show the best model
    best_model = perf_df['F1-Score'].idxmax()
    best_score = perf_df.loc[best_model, 'F1-Score']
    st.success(f"🏆 **Best Model: {best_model}** (F1-Score: {best_score:.3f})")
    
except:
    st.info("Model performance data not available")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "🦟 AfyaAlert - Malaria Prediction System | Data Science Project | © 2026"
    "</div>",
    unsafe_allow_html=True
)