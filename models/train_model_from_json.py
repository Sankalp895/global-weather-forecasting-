import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("=" * 60)
print("TRAINING WEATHER PREDICTION MODEL")
print("=" * 60)

# ===== 1. LOAD DATA =====
print("\nLoading data...")
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train_scaled.npy')
y_test = np.load('y_test_scaled.npy')

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ===== 2. LOAD HYPERPARAMETERS =====
print("\n" + "=" * 60)
print("LOADING HYPERPARAMETERS")
print("=" * 60)

if os.path.exists('best_hyperparameters.json'):
    print("\n✅ Found best_hyperparameters.json - Using tuned parameters")

    with open('best_hyperparameters.json', 'r') as f:
        data = json.load(f)

    hyperparams = data['best_hyperparameters']

    print("\nLoaded hyperparameters:")
    for param, value in hyperparams.items():
        print(f"  {param:20s}: {value}")

    if 'tuning_metadata' in data:
        print(f"\nTuned on: {data['tuning_metadata']['timestamp']}")
        print(f"CV Score: {data['cv_score']:.4f}")
else:
    print("\n⚠️  best_hyperparameters.json not found")
    print("Using default hyperparameters instead")
    print("\nTo get tuned parameters, run:")
    print("  python hyperparameter_tuning.py")

    # Default parameters
    hyperparams = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True
    }

    print("\nDefault hyperparameters:")
    for param, value in hyperparams.items():
        print(f"  {param:20s}: {value}")

# ===== 3. TRAIN MODEL =====
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        max_features=hyperparams['max_features'],
        bootstrap=hyperparams['bootstrap'],
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
)

print("\nTraining in progress...")
model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# ===== 4. MAKE PREDICTIONS =====
print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# Inverse transform to original scale
y_train_actual = scaler_y.inverse_transform(y_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# ===== 5. EVALUATE MODEL =====
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

targets = ['Temperature (°C)', 'Humidity (%)']
evaluation_results = {}

for i, target in enumerate(targets):
    print(f"\n{target}:")
    print("-" * 40)

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

    # Store results
    evaluation_results[target.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°', '')] = {
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)}
    }

# ===== 6. VISUALIZATION =====
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

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

# ===== 7. SAVE MODEL AND RESULTS =====
print("\n" + "=" * 60)
print("SAVING MODEL AND RESULTS")
print("=" * 60)

# Save model
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved: weather_model.pkl")

# Save evaluation metrics
results_output = {
    "hyperparameters": hyperparams,
    "evaluation_metrics": evaluation_results,
    "model_info": {
        "model_type": "MultiOutputRegressor(RandomForestRegressor)",
        "n_training_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "targets": ["temperature_celsius", "humidity"]
    }
}

with open('model_results.json', 'w') as f:
    json.dump(results_output, f, indent=4)
print("✅ Results saved: model_results.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  - weather_model.pkl (trained model)")
print("  - model_predictions.png (visualization)")
print("  - model_results.json (metrics & hyperparameters)")