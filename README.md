# 🌤️ Weather Prediction ML System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-Ensemble-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

Advanced machine learning system for global weather forecasting using ensemble methods with comprehensive analysis and interactive dashboard.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Dashboard](#-dashboard)

</div>

---

## 📋 Table of Contents

- [About](#-about-the-project)
- [Features](#-features)
- [Technologies](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Dashboard](#-dashboard)
- [Analysis](#-analysis)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 About The Project

This project implements an advanced machine learning pipeline for predicting temperature and humidity using ensemble methods. The system processes 108,353+ global weather observations, performs comprehensive exploratory data analysis, trains multiple models, and provides explainable predictions through an interactive Streamlit dashboard.

### 🎓 PM Accelerator Mission

**Empowering Innovation Through Data-Driven Solutions**

This project demonstrates advanced machine learning techniques for weather prediction, combining multiple models, ensemble methods, and comprehensive analysis to deliver accurate forecasts that can help communities prepare for weather events and make informed decisions.

### 🎥 Key Highlights

- **6 ML Models** trained and compared (Random Forest, XGBoost, LightGBM, Linear, Ridge, Lasso)
- **Best R² Score**: 0.9414 (Temperature prediction with Random Forest)
- **34 Engineered Features** including lag features, rolling statistics, and temporal encodings
- **5 Anomaly Detection Methods** with consensus-based outlier identification
- **SHAP Analysis** for model explainability
- **Interactive Dashboard** built with Streamlit

---

## ✨ Features

### 🔬 Data Analysis
- **Comprehensive EDA**: Distribution analysis, correlation studies, temporal trends
- **Anomaly Detection**: 5 methods (Z-Score, IQR, Isolation Forest, LOF, DBSCAN)
- **Consensus Outliers**: 3,271 outliers (3.02%) identified with high confidence
- **Data Quality**: Missing value analysis and handling

### 🤖 Machine Learning
- **Multiple Models**: Random Forest, XGBoost, LightGBM, Linear, Ridge, Lasso
- **Ensemble Stacking**: Meta-learner combining top 3 models
- **Hyperparameter Tuning**: Grid search optimization
- **Cross-Validation**: Robust model evaluation

### 🎨 Feature Engineering
- **Lag Features**: 1, 3, 7, 30-day historical values
- **Rolling Statistics**: Mean, std for 3, 7, 30-day windows
- **Temporal Features**: Month, day of year, cyclical encoding
- **Geographic Features**: Latitude, longitude, location encoding

### 📊 Analysis & Visualization
- **Feature Importance**: Permutation importance + SHAP values
- **Spatial Analysis**: Geographic weather patterns and climate zones
- **Environmental Impact**: Heat index, extreme weather events
- **Model Comparison**: Comprehensive performance metrics

### 🖥️ Interactive Dashboard
- **7 Navigation Pages**: Overview, EDA, Performance, Features, Spatial, Environmental, Conclusions
- **Real-time Visualizations**: 15+ charts and graphs
- **Model Metrics**: Live performance comparisons
- **Insights**: Actionable recommendations

---

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.10+**
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Fast gradient boosting
- **SHAP** - Model explainability

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scipy** - Scientific computing

### Visualization
- **matplotlib** - Plotting library
- **seaborn** - Statistical visualization
- **Streamlit** - Interactive dashboard

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Clone Repository

### Install Dependencies

### Download Dataset

Place `GlobalWeatherRepository.csv` in the root directory.

---

## 🚀 Usage

### 1. Feature Engineering
### 2. Train Models
Trains 6 different models with hyperparameter tuning and saves the best performers.
### 3. Anomaly Detection
Identifies outliers using 5 different methods and generates consensus results.
### 4. Feature Importance Analysis
Calculates permutation importance and SHAP values for model explainability.
### 5. Spatial Analysis
Analyzes geographic weather patterns and climate zones.
### 6. Environmental Analysis
Studies heat index, extreme weather, and environmental impact.


### ALL THESE ARE PRESENT IN THE analysis/  folder.

###  Launch Dashboard


## 🔬 Methodology

### 1. Data Collection & Exploration
- 108,353 global weather observations
- Features: temperature, humidity, wind, pressure, precipitation
- Time range: Multiple years of historical data

### 2. Feature Engineering
- **Lag Features**: Capture recent historical patterns
- **Rolling Statistics**: Smooth trends and variability
- **Temporal Features**: Encode seasonality
- **Geographic Features**: Location-based patterns

### 3. Anomaly Detection
- **Z-Score**: Statistical outlier detection
- **IQR**: Box plot method
- **Isolation Forest**: ML-based detection
- **LOF**: Density-based detection
- **DBSCAN**: Clustering-based detection

### 4. Model Training
- Train 6 models independently
- Hyperparameter tuning via GridSearchCV
- Cross-validation for robust evaluation
- Ensemble stacking with meta-learner

### 5. Evaluation & Analysis
- RMSE, MAE, R² metrics
- Feature importance via permutation
- SHAP values for explainability
- Spatial and environmental analysis

---

## 📈 Results

### Model Performance (Temperature Prediction)

| Model | Test RMSE | Test R² | Train RMSE |
|-------|-----------|---------|------------|
| **Random Forest** | **2.376°C** | **0.9414** | 0.727°C |
| XGBoost | 2.394°C | 0.9408 | 1.250°C |
| LightGBM | 2.379°C | 0.9413 | 1.623°C |
| Linear | 2.384°C | 0.9410 | 2.387°C |
| Ridge | 2.384°C | 0.9410 | 2.387°C |
| Lasso | 2.767°C | 0.9259 | 2.777°C |

### Model Performance (Humidity Prediction)

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| **Random Forest** | **11.206%** | **0.7892** |
| LightGBM | 11.253% | 0.7875 |
| XGBoost | 11.444% | 0.7804 |

### Key Findings

#### 🏆 Best Models
- **Temperature**: Random Forest (RMSE: 2.376°C, R²: 0.9414)
- **Humidity**: Random Forest (RMSE: 11.206%, R²: 0.7892)

#### 📊 Feature Importance
1. **temperature_celsius_lag1** - Most recent temperature (highest importance)
2. **humidity_lag1** - Most recent humidity
3. **temperature_celsius_roll_mean_30** - 30-day rolling average
4. **humidity_roll_mean_30** - 30-day humidity average
5. **latitude** - Geographic location

#### 🌍 Spatial Insights
- **Hottest**: Myanmar (30°C average)
- **Coldest**: Ottawa, Canada (-5°C average)
- **Latitude Correlation**: r = -0.32 with temperature
- **Climate Zones**: 3 zones identified (Tropical, Temperate, Cold)

#### 🚨 Anomaly Detection
- **3,271 consensus outliers** (3.02% of data)
- Island nations show highest outlier rates (48% Marshall Islands)
- Winter months have 6.7% outlier rate (highest)
- Outliers represent real extreme weather, not errors

---

## 🖥️ Dashboard

### Interactive Streamlit Dashboard

Access comprehensive analysis through 7 navigation pages:

#### 1. 🏠 Overview
- Project objectives and scope
- Technology stack
- Dataset information
- Key metrics

#### 2. 📈 EDA & Data Quality
- Target variable distributions
- Time series trends
- Anomaly detection results
- Geographic outlier patterns

#### 3. 🤖 Model Performance
- Performance comparison across 6 models
- Prediction vs actual visualizations
- Error analysis

#### 4. 🔍 Feature Importance
- Permutation importance rankings
- SHAP value analysis
- Feature distribution insights

#### 5. 🌍 Spatial Analysis
- Global temperature/humidity maps
- Climate zone classification
- Latitude correlation analysis

#### 6. 🌿 Environmental Impact
- Heat index distribution
- Extreme weather events
- Seasonal patterns

#### 7. 🎯 Conclusions
- Key achievements
- Insights and recommendations
- Future improvements


## 📊 Analysis

### Anomaly Detection

5 methods used for robust outlier identification:
- **Z-Score**: 5,459 outliers
- **IQR**: 39,444 outliers
- **Isolation Forest**: 5,418 outliers
- **LOF**: 500 outliers
- **DBSCAN**: 9,820 outliers

**Consensus Outliers**: 3,271 (flagged by ≥3 methods)

### Spatial Analysis

- **242 unique locations** across **130 countries**
- **Latitude effect**: Strong inverse correlation with temperature
- **Climate zones**: Tropical (25°C+), Temperate (10-25°C), Cold (<10°C)
- **Island nations**: Show extreme variability

### Environmental Impact

- **Extreme Heat**: 4,818 events (>35°C)
- **Extreme Cold**: 198 events (<-10°C)
- **High Humidity**: 15,438 events (>90%)
- **Heat Index**: Tropical countries experience highest stress

---
## 👤 Contact

**Sankalp**
- Role: Backend Engineer | ML Enthusiast
- GitHub: [sankalp895@gmail.com](https://github.com/Sankalp895)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/sankalp-singh-420b3a246)
- Email: sankalp895@gmail.com

---

## 🙏 Acknowledgments

- **PM Accelerator** - For project guidance and mission
- **scikit-learn** - Machine learning library
- **XGBoost & LightGBM** - Gradient boosting frameworks
- **SHAP** - Model explainability
- **Streamlit** - Dashboard framework
- **Weather Data Providers** - For comprehensive dataset

---

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">
