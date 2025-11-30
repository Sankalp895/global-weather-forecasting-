import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns

# ===== REUSABLE FUNCTION =====
def load_weather_data(filepath='GlobalWeatherRepository.csv'):
    """Load and preprocess weather data"""
    df = pd.read_csv(filepath)
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    return df