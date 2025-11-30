import pandas as pd
import numpy as np
from initialization import load_weather_data
from sklearn.preprocessing import LabelEncoder

# Load data
df = load_weather_data()
print("Starting feature engineering...")
print(f"Original shape: {df.shape}")

# Sort properly
df = df.sort_values(['location_name', 'last_updated'])

# ===== 1. TEMPORAL FEATURES =====
df['year'] = df['last_updated'].dt.year
df['month'] = df['last_updated'].dt.month
df['day'] = df['last_updated'].dt.day
df['day_of_week'] = df['last_updated'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_year'] = df['last_updated'].dt.dayofyear
df['quarter'] = df['last_updated'].dt.quarter

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

print("✅ Temporal features created")

# ===== 2. LAG FEATURES (Updated & Expanded) =====
targets = ['temperature_celsius', 'humidity']

for col in targets:
    print(f"Creating features for {col}...")
    
    # Standard Lags (Yesterday, last week, last month)
    df[f'{col}_lag1'] = df.groupby('location_name')[col].shift(1)
    df[f'{col}_lag3'] = df.groupby('location_name')[col].shift(3)
    df[f'{col}_lag7'] = df.groupby('location_name')[col].shift(7)
    df[f'{col}_lag30'] = df.groupby('location_name')[col].shift(30)
    
    # Rolling Means (Trends - using shift(1) to avoid leakage!)
    # Note: We shift(1) first so we don't include TODAY's value in the mean
    df[f'{col}_roll_mean_3'] = df.groupby('location_name')[col].shift(1).rolling(3).mean()
    df[f'{col}_roll_mean_7'] = df.groupby('location_name')[col].shift(1).rolling(7).mean()
    df[f'{col}_roll_mean_30'] = df.groupby('location_name')[col].shift(1).rolling(30).mean()
    
    # Rolling Volatility (Standard Deviation)
    df[f'{col}_roll_std_7'] = df.groupby('location_name')[col].shift(1).rolling(7).std()
    
    # Difference Features (Rate of Change)
    # diff(1) means (Today - Yesterday)
    # Note: We shift(1) the result to ensure we only know yesterday's change
    df[f'{col}_diff_1'] = df.groupby('location_name')[col].diff(1).shift(1)

print("✅ Advanced lag features created")

# ===== 3. ENCODING =====
le_location = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['location_name'])

le_country = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])

le_condition = LabelEncoder()
df['condition_encoded'] = le_condition.fit_transform(df['condition_text'])

if 'wind_direction' in df.columns:
    # Handle mixed types in wind_direction if any
    df['wind_direction'] = df['wind_direction'].astype(str)
    le_wind = LabelEncoder()
    df['wind_dir_encoded'] = le_wind.fit_transform(df['wind_direction'])

print("✅ Categorical features encoded")

# ===== 4. CLEANUP & SAVE =====
print(f"\nMissing values before dropping:\n{df.isnull().sum().sum()}")
df = df.dropna()
print(f"Shape after dropping NaN: {df.shape}")

# Save
df.to_csv('weather_featured.csv', index=False)
print("\n✅ Feature engineering complete!")
print(f"Final dataset saved: weather_featured.csv")
print(f"Total features: {len(df.columns)}")

print(f"\nSample of new features:")
cols_to_show = ['last_updated', 'temperature_celsius', 
                'temperature_celsius_lag1', 'temperature_celsius_roll_mean_7', 
                'temperature_celsius_diff_1']
print(df[cols_to_show].head())
