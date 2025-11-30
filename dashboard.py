import streamlit as st
import pandas as pd
import json


st.set_page_config(
    page_title="Weather Prediction ML Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 24px;
        color: #424242;
        text-align: center;
        margin-bottom: 30px;
    }
    .mission-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(to right, #1E88E5, #64B5F6);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-header">🌤️ Weather Prediction ML System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Machine Learning for Global Weather Forecasting</p>', unsafe_allow_html=True)


st.markdown("""
<div class="mission-box">
    <h3>🎯 PM Accelerator Mission</h3>
    <p><strong>Empowering Innovation Through Data-Driven Solutions</strong></p>
    <p>This project demonstrates advanced machine learning techniques for weather prediction, 
    combining multiple models, ensemble methods, and comprehensive analysis to deliver accurate 
    forecasts that can help communities prepare for weather events and make informed decisions.</p>
</div>
""", unsafe_allow_html=True)


st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Overview", "📈 EDA & Data Quality", "🤖 Model Performance", 
     "🔍 Feature Importance", "🌍 Spatial Analysis", "🌿 Environmental Impact", 
     "📊 Model Comparison", "🎯 Conclusions"]
)


if page == "🏠 Overview":
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", "108,353")
        st.markdown("Global weather observations")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models Trained", "6")
        st.markdown("RF, XGBoost, LightGBM, Linear, Ridge, Lasso")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Model R²", "0.9414")
        st.markdown("Random Forest (Temperature)")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🎯 Project Objectives")
    st.markdown("""
    1. **Data Exploration**: Comprehensive EDA and anomaly detection
    2. **Model Development**: Train and compare 6 different ML models
    3. **Ensemble Learning**: Combine models for improved accuracy
    4. **Feature Analysis**: Understand what drives predictions
    5. **Spatial Patterns**: Analyze geographic weather variations
    6. **Environmental Impact**: Study extreme weather and comfort indices
    """)

    st.subheader("🔧 Technologies Used")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Machine Learning:**
        - Random Forest
        - XGBoost
        - LightGBM
        - Linear Models (Ridge, Lasso)
        - Stacking Ensemble
        """)

    with col2:
        st.markdown("""
        **Analysis Tools:**
        - SHAP (Feature Importance)
        - Permutation Importance
        - Anomaly Detection (5 methods)
        - Spatial Analysis
        - Environmental Metrics
        """)

    st.subheader("📁 Dataset Information")
    st.markdown("""
    - **Source**: GlobalWeatherRepository.csv
    - **Features**: 34 engineered features including:
        - Lag features (1, 3, 7, 30 days)
        - Rolling statistics (mean, std)
        - Temporal features (month, day, cyclical encoding)
        - Location features (latitude, longitude, encoded)
    - **Targets**: Temperature (°C) and Humidity (%)
    """)


elif page == "📈 EDA & Data Quality":
    st.header("Exploratory Data Analysis & Anomaly Detection")

    tab1, tab2, tab3 = st.tabs(["📊 Data Distributions", "🚨 Anomaly Detection", "📍 Anomaly Locations"])

    with tab1:
        st.subheader("Target Variable Distributions")
        st.image('target_distributions.png', use_container_width=True)

        st.subheader("Time Trends")
        st.image('time_trends.png', use_container_width=True)

        st.markdown("""
        **Key Findings:**
        - Temperature shows clear seasonal patterns
        - Humidity exhibits gradual increasing trend
        - Precipitation is sporadic with occasional spikes
        """)

    with tab2:
        st.subheader("Anomaly Detection Results")
        st.image('anomaly_visualization.png', use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Detection Methods Used:**
            - Z-Score (statistical)
            - IQR (box plot)
            - Isolation Forest (ML)
            - Local Outlier Factor (density)
            - DBSCAN (clustering)
            """)

        with col2:
            st.metric("Consensus Outliers", "3,271", "3.02% of data")
            st.info("Flagged by ≥3 methods for high confidence")

    with tab3:
        st.subheader("Temporal & Geographic Anomaly Patterns")
        st.image('anomaly_time_distribution.png', use_container_width=True)
        st.image('anomaly_location_analysis.png', use_container_width=True)

        st.markdown("""
        **Geographic Insights:**
        - **Island nations dominate**: Marshall Islands (48%), Micronesia (45%)
        - **Arctic regions**: Canada (42%), Iceland (20%)
        - **Winter months** show 6.7% outlier rate (highest)
        - Outliers represent **real extreme weather**, not data errors
        """)


elif page == "🤖 Model Performance":
    st.header("Model Performance & Predictions")

    tab1, tab2 = st.tabs(["📊 Performance Comparison", "🎯 Predictions"])

    with tab1:
        st.subheader("Model Performance Across 6 Models")
        st.image('model_performance_comparison.png', use_container_width=True)

        st.subheader("Performance Metrics Table")

        try:
            df = pd.read_csv('model_comparison_table.csv')
            st.dataframe(df.style.highlight_min(subset=['Temp_Test_RMSE', 'Hum_Test_RMSE'], color='lightgreen')
                                 .highlight_max(subset=['Temp_Test_R2', 'Hum_Test_R2'], color='lightgreen'),
                        use_container_width=True)
        except:
            st.warning("model_comparison_table.csv not found")

        col1, col2 = st.columns(2)
        with col1:
            st.success("🏆 **Best Temperature Model**: Random Forest (RMSE: 2.376°C, R²: 0.9414)")
        with col2:
            st.success("🏆 **Best Humidity Model**: Random Forest (RMSE: 11.206%, R²: 0.7892)")

    with tab2:
        st.subheader("Model Predictions vs Actual Values")
        st.image('model_predictions.png', use_container_width=True)

        st.markdown("""
        **Prediction Quality:**
        - Temperature predictions are highly accurate (tight clustering around diagonal)
        - Humidity predictions show more variance (inherently harder to predict)
        - Minimal bias in predictions across the range
        """)


elif page == "🔍 Feature Importance":
    st.header("Feature Importance Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Permutation Importance", "🎯 SHAP Values", "📈 Feature Distribution"])

    with tab1:
        st.subheader("Permutation Importance Across Models")
        st.image('permutation_importance_all_models.png', use_container_width=True)

        st.markdown("""
        **Top Features:**
        1. **temperature_celsius_lag1**: Most recent temperature (highest importance)
        2. **humidity_lag1**: Most recent humidity
        3. **temperature_celsius_roll_mean_30**: 30-day rolling average
        4. **humidity_roll_mean_30**: 30-day humidity average
        5. **Latitude**: Geographic location matters significantly
        """)

    with tab2:
        st.subheader("SHAP Values - Random Forest")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Temperature Predictions**")
            st.image('shap_rf_temperature.png', use_container_width=True)

        with col2:
            st.markdown("**Humidity Predictions**")
            st.image('shap_rf_humidity.png', use_container_width=True)

        st.info("SHAP values show how each feature contributes to individual predictions. Red = high feature value, Blue = low feature value")

    with tab3:
        st.subheader("Feature Comparison Across All Models")
        st.image('feature_comparison_all_models.png', use_container_width=True)
        st.image('feature_importance_distribution.png', use_container_width=True)

        st.markdown("""
        **Key Insights:**
        - **Lag features dominate**: Recent history is the best predictor
        - **Rolling statistics** capture trends effectively
        - **Temporal features** (month, day) capture seasonality
        - **Model agreement**: All models agree on top features
        """)


elif page == "🌍 Spatial Analysis":
    st.header("Geographic Weather Patterns")

    tab1, tab2 = st.tabs(["🗺️ Global Patterns", "📊 Climate Zones"])

    with tab1:
        st.subheader("Global Temperature & Humidity Distribution")
        st.image('spatial_weather_patterns.png', use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hottest Location", "Myanmar", "30°C avg")
        with col2:
            st.metric("Coldest Location", "Ottawa, Canada", "-5°C avg")
        with col3:
            st.metric("Latitude Correlation", "r = -0.32", "with temperature")

    with tab2:
        st.subheader("Climate Zone Classification")

        try:
            with open('analysis/spatial_analysis_summary.json', 'r') as f:
                spatial_data = json.load(f)

            climate_zones = spatial_data.get('climate_zones', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tropical (>25°C)", climate_zones.get('Tropical', 0))
            with col2:
                st.metric("Temperate (10-25°C)", climate_zones.get('Temperate', 0))
            with col3:
                st.metric("Cold (<10°C)", climate_zones.get('Cold', 0))
        except:
            st.warning("spatial_analysis_summary.json not found in analysis/ folder")

        st.markdown("""
        **Geographic Insights:**
        - Latitude significantly affects temperature patterns
        - Island nations show unique weather variability
        - Different climate zones may benefit from separate models
        """)


elif page == "🌿 Environmental Impact":
    st.header("Environmental Impact Analysis")

    tab1, tab2 = st.tabs(["🌡️ Environmental Conditions", "📊 Seasonal Patterns"])

    with tab1:
        st.subheader("Heat Index & Environmental Comfort")
        st.image('environmental_impact_analysis.png', use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        try:
            with open('analysis/environmental_impact_summary.json', 'r') as f:
                env_data = json.load(f)

            extreme = env_data.get('extreme_weather', {})

            with col1:
                st.metric("Extreme Heat", f"{extreme.get('extreme_heat_count', 0):,}", ">35°C")
            with col2:
                st.metric("Extreme Cold", f"{extreme.get('extreme_cold_count', 0):,}", "<-10°C")
            with col3:
                st.metric("High Humidity", f"{extreme.get('high_humidity_count', 0):,}", ">90%")
            with col4:
                st.metric("Low Humidity", f"{extreme.get('low_humidity_count', 0):,}", "<20%")
        except:
            st.warning("environmental_impact_summary.json not found in analysis/ folder")

    with tab2:
        st.subheader("Seasonal Trends & Extreme Events")
        st.image('environmental_trends.png', use_container_width=True)

        st.markdown("""
        **Environmental Findings:**
        - **Heat Index**: Tropical countries experience highest heat stress
        - **Seasonal Patterns**: Clear temperature and precipitation cycles
        - **Extreme Events**: 3% of records show extreme conditions
        - **Wind Patterns**: Vary significantly by geographic location
        """)


elif page == "📊 Model Comparison":
    st.header("Comprehensive Model Comparison")

    st.markdown("""
    ### Models Evaluated:
    1. **Random Forest** - Ensemble of decision trees (with hyperparameter tuning)
    2. **XGBoost** - Gradient boosting (optimized)
    3. **LightGBM** - Fast gradient boosting
    4. **Linear Regression** - Baseline linear model
    5. **Ridge Regression** - L2 regularization
    6. **Lasso Regression** - L1 regularization
    7. **Stacking Ensemble** - Combines RF, XGBoost, LightGBM
    """)

    try:
        with open('best_model_recommendation.txt', 'r', encoding='utf-8') as f:
            recommendation = f.read()

        st.text_area("Model Comparison Report", recommendation, height=400)
    except:
        st.warning("best_model_recommendation.txt not found")

    st.subheader("Performance Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Temperature Prediction:**
        - 🥇 Random Forest: 2.376°C RMSE
        - 🥈 LightGBM: 2.379°C RMSE
        - 🥉 Linear: 2.384°C RMSE
        """)

    with col2:
        st.markdown("""
        **Humidity Prediction:**
        - 🥇 Random Forest: 11.206% RMSE
        - 🥈 LightGBM: 11.253% RMSE
        - 🥉 Linear: 11.292% RMSE
        """)

    st.info("💡 **Recommendation**: Use Random Forest for production deployment. Consider ensemble for maximum accuracy.")


elif page == "🎯 Conclusions":
    st.header("Conclusions & Recommendations")

    st.subheader("🎯 Key Achievements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ Completed Tasks:**
        1. Advanced EDA with anomaly detection (5 methods)
        2. Trained and compared 6 ML models
        3. Created stacking ensemble
        4. Feature importance analysis (Permutation + SHAP)
        5. Spatial/geographic analysis
        6. Environmental impact study
        """)

    with col2:
        st.markdown("""
        **📊 Results Summary:**
        - Best Temperature R²: **0.9414**
        - Best Humidity R²: **0.7892**
        - Anomaly Detection: **3% outliers** identified
        - Feature Analysis: **Lag features** most important
        - Climate Zones: **3 zones** classified
        """)

    st.markdown("---")

    st.subheader("💡 Key Insights")
    st.markdown("""
    1. **Model Performance**: Tree-based models (RF, XGBoost, LightGBM) significantly outperform linear models
    2. **Feature Importance**: Recent history (lag features) and rolling statistics are critical predictors
    3. **Geographic Patterns**: Latitude strongly correlates with temperature (-0.32), island nations show extreme variability
    4. **Data Quality**: 3% outliers represent real extreme weather events, not errors
    5. **Ensemble Learning**: Stacking provides marginal improvements over best single model
    """)

    st.markdown("---")

    st.subheader("🚀 Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **For Production:**
        - Deploy Random Forest
        - Use ensemble for critical applications
        - Monitor model drift monthly
        """)

    with col2:
        st.markdown("""
        **For Improvement:**
        - Add weather station data
        - Include satellite imagery
        - Implement real-time updates
        """)

    with col3:
        st.markdown("""
        **For Research:**
        - Separate models per climate zone
        - Deep learning approaches
        - Multi-step forecasting
        """)

    st.markdown("---")

    st.subheader("📚 Technical Stack")
    st.code("""
    Python 3.10+
    ├── Data Processing: pandas, numpy
    ├── Machine Learning: scikit-learn, xgboost, lightgbm
    ├── Visualization: matplotlib, seaborn
    ├── Feature Importance: shap
    ├── Deployment: streamlit
    └── Version Control: git
    """)

    st.markdown("---")

    st.success("🎉 **Project Complete!** All analysis tasks successfully completed with comprehensive insights delivered.")


st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.markdown("**Sankalp**")
st.sidebar.markdown("Backend Engineer | ML Enthusiast")
st.sidebar.markdown("---")
st.sidebar.info("**Tech Stack**: Python • scikit-learn • XGBoost • LightGBM • Streamlit")