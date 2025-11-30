import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the featured data
df1 = pd.read_csv('weather_featured.csv')
df1['last_updated'] = pd.to_datetime(df1['last_updated'])

print(f"Loaded data: {df1.shape}")
targets = ['temperature_celsius', 'humidity']

exclude_cols = [
    'last_updated', 
    'temperature_celsius', 'humidity',  # targets
    
    # LEAKAGE: Remove all "current" weather metrics
    'feels_like_celsius', 'feels_like_fahrenheit',
    'temperature_fahrenheit',
    'wind_kph', 'wind_mph', 'wind_degree', 'wind_direction',
    'pressure_mb', 'pressure_in',
    'precip_mm', 'precip_in',
    'cloud', 'visibility_km', 'visibility_miles',
    'uv_index', 'gust_kph', 'gust_mph',
    
    # Categorical originals (we have encoded versions, but they are also current!)
    'condition_text', 'condition_encoded',  # This is "Sunny/Rainy" right now - can't use it!
    'country', 'location_name', 'timezone', # strings
    
    # Air quality (current state)
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
    'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
    'air_quality_PM2.5', 'air_quality_PM10', 
    'air_quality_us-epa-index', 'air_quality_gb-defra-index',
    
    # Misc
    'last_updated_epoch', 'sunrise', 'sunset', 'moonrise', 'moonset', 'moon_phase'
]

all_cols = df1.columns.tolist()
feature_cols = [col for col in all_cols if col not in exclude_cols]


print(f"\nTarget variables: {targets}")
print(f"Number of features: {len(feature_cols)}")

#X and Y
X = df1[feature_cols].copy()
y = df1[targets].copy()

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

split_index = int(len(df1) * 0.75) #splitting

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train date range: {df1['last_updated'].iloc[0]} to {df1['last_updated'].iloc[split_index-1]}")
print(f"Test date range: {df1['last_updated'].iloc[split_index]} to {df1['last_updated'].iloc[-1]}")


#Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print("\n✅ Data scaling complete!")

import pickle

# Save arrays
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train_scaled.npy', y_train_scaled)
np.save('y_test_scaled.npy', y_test_scaled)

with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("\n✅ All data saved!")
print("\nFiles created:")
print("- X_train_scaled.npy, X_test_scaled.npy")
print("- y_train_scaled.npy, y_test_scaled.npy")
print("- scaler_X.pkl, scaler_y.pkl")
print("- feature_names.pkl")   