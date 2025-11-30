import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading data...")
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train_scaled.npy')
y_test = np.load('y_test_scaled.npy')

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\nTraining Multi-Output Random Forest model...")

#random forest multi_output regression
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

#Prediction
print("\nMaking predictions...")
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

print("\nMaking predictions...")
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# Inverse transform to original scale
y_train_actual = scaler_y.inverse_transform(y_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

y_test_actual = scaler_y.inverse_transform(y_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

#EVALUALTION

# ===== 4. EVALUATE MODEL =====
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

targets = ['Temperature (°C)', 'Humidity (%)']

for i, target in enumerate(targets):
    print(f"\n{target}:")
    print("-" * 40)
    
    # CHANGED: added [:, i] to y_train_pred and y_test_pred
    # Train metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual[:, i], y_train_pred[:, i]))
    train_mae = mean_absolute_error(y_train_actual[:, i], y_train_pred[:, i])
    train_r2 = r2_score(y_train_actual[:, i], y_train_pred[:, i])
    
    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test_actual[:, i], y_test_pred[:, i]))
    test_mae = mean_absolute_error(y_test_actual[:, i], y_test_pred[:, i])
    test_r2 = r2_score(y_test_actual[:, i], y_test_pred[:, i])
    
    print(f"  TRAIN - RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
    print(f"  TEST  - RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")

#VISUALTIZATION
print("\nCreating prediction plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Temperature - Train
axes[0, 0].scatter(y_train_actual[:, 0], y_train_pred[:, 0], alpha=0.3, s=1)
axes[0, 0].plot([y_train_actual[:, 0].min(), y_train_actual[:, 0].max()], 
                 [y_train_actual[:, 0].min(), y_train_actual[:, 0].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Temperature (°C)')
axes[0, 0].set_ylabel('Predicted Temperature (°C)')
axes[0, 0].set_title('Temperature - Training Set')
axes[0, 0].grid(True, alpha=0.3)

# Temperature - Test
axes[0, 1].scatter(y_test_actual[:, 0], y_test_pred[:, 0], alpha=0.3, s=1, color='orange')
axes[0, 1].plot([y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 
                 [y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Temperature (°C)')
axes[0, 1].set_ylabel('Predicted Temperature (°C)')
axes[0, 1].set_title('Temperature - Test Set')
axes[0, 1].grid(True, alpha=0.3)

# Humidity - Train
axes[1, 0].scatter(y_train_actual[:, 1], y_train_pred[:, 1], alpha=0.3, s=1)
axes[1, 0].plot([y_train_actual[:, 1].min(), y_train_actual[:, 1].max()], 
                 [y_train_actual[:, 1].min(), y_train_actual[:, 1].max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Humidity (%)')
axes[1, 0].set_ylabel('Predicted Humidity (%)')
axes[1, 0].set_title('Humidity - Training Set')
axes[1, 0].grid(True, alpha=0.3)

# Humidity - Test
axes[1, 1].scatter(y_test_actual[:, 1], y_test_pred[:, 1], alpha=0.3, s=1, color='blue')
axes[1, 1].plot([y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 
                 [y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Humidity (%)')
axes[1, 1].set_ylabel('Predicted Humidity (%)')
axes[1, 1].set_title('Humidity - Test Set')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300)
print("✅ Plots saved: model_predictions.png")

# ===== 6. SAVE MODEL =====
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✅ Model saved: weather_model.pkl")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
