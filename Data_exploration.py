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

# ===== LOAD DATA USING FUNCTION =====
df = load_weather_data()

print(df.shape) #(rows, col)
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#print(df.info())
#print(df[df.duplicated()])
#print(df.duplicated().sum())
#print(list(df.columns))
print(f"Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")
#numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


#exclude = ['Unamed: 0', 'latitude', 'longitude']
#numerical_cols = [col for col in numerical_cols if col not in exclude]
#print(f"Found {len(numerical_cols)} numerical features")#28
#print(numerical_cols)
print("All columns in dataset:")
print(df.columns.tolist())
print(f"\nTotal columns: {len(df.columns)}")
"""
 Option 2: Multiple separate models


Model 1: Predicts temperature


Model 2: Predicts humidity


Model 3: Predicts precipitation


More work but shows breadth
"""
# Check data types
print("\nData types:")
print(df.dtypes)
targets = ['temperature_celsius', 'humidity', 'precip_mm']
print(df[targets].describe())
print(f"\nMissing values:\n{df[targets].isnull().sum()}")
