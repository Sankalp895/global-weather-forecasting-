import numpy as np
import pickle
import json
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os

print("=" * 80)
print("TRAINING LIGHTGBM MODEL")
print("=" * 80)

# ===== LOAD DATA =====
print("\nLoading data...")
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train_scaled.npy')
y_test = np.load('y_test_scaled.npy')

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ===== LOAD OR USE DEFAULT HYPERPARAMETERS =====
if os.path.exists('lightgbm_best_params.json'):
    print("\n✅ Loading tuned hyperparameters from lightgbm_best_params.json")
    with open('lightgbm_best_params.json', 'r') as f:
        data = json.load(f)
        params = data['best_hyperparameters']
else:
    print("\n⚠️  No tuned parameters found. Using default hyperparameters.")
    params = {
        'n_estimators': 300,
        'max_depth': 15,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20
    }

print("\nHyperparameters:")
for k, v in params.items():
    print(f"  {k:20s}: {v}")

# ===== TRAIN MODELS (SEPARATE FOR EACH TARGET) =====
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {}
predictions = {}

for i, target_name in enumerate(['temperature', 'humidity']):
    print(f"\nTraining LightGBM for {target_name}...")
    
    start_time = time.time()
    
    model = lgb.LGBMRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_samples=params['min_child_samples'],
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train[:, i])
    
    end_time = time.time()
    print(f"✅ Training complete in {end_time - start_time:.2f} seconds")
    
    models[target_name] = model
    predictions[f'{target_name}_train'] = model.predict(X_train)
    predictions[f'{target_name}_test'] = model.predict(X_test)

# ===== INVERSE TRANSFORM PREDICTIONS =====
print("\n" + "=" * 80)
print("MAKING PREDICTIONS")
print("=" * 80)

y_train_pred_scaled = np.column_stack([predictions['temperature_train'], 
                                       predictions['humidity_train']])
y_test_pred_scaled = np.column_stack([predictions['temperature_test'], 
                                      predictions['humidity_test']])

y_train_actual = scaler_y.inverse_transform(y_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# ===== EVALUATE =====
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

targets = ['Temperature (°C)', 'Humidity (%)']
evaluation_results = {}

for i, target in enumerate(targets):
    print(f"\n{target}:")
    print("-" * 40)
    
    train_rmse = np.sqrt(mean_squared_error(y_train_actual[:, i], y_train_pred[:, i]))
    train_mae = mean_absolute_error(y_train_actual[:, i], y_train_pred[:, i])
    train_r2 = r2_score(y_train_actual[:, i], y_train_pred[:, i])
    
    test_rmse = np.sqrt(mean_squared_error(y_test_actual[:, i], y_test_pred[:, i]))
    test_mae = mean_absolute_error(y_test_actual[:, i], y_test_pred[:, i])
    test_r2 = r2_score(y_test_actual[:, i], y_test_pred[:, i])
    
    print(f"  TRAIN - RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R²: {train_r2:.3f}")
    print(f"  TEST  - RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R²: {test_r2:.3f}")
    
    target_key = target.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°', '')
    evaluation_results[target_key] = {
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)}
    }

# ===== SAVE MODELS AND RESULTS =====
print("\n" + "=" * 80)
print("SAVING MODELS AND RESULTS")
print("=" * 80)

with open('lightgbm_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✅ Saved: lightgbm_models.pkl")

results_output = {
    "model_type": "LightGBM",
    "hyperparameters": params,
    "evaluation_metrics": evaluation_results,
    "model_info": {
        "n_training_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "targets": ["temperature_celsius", "humidity"]
    }
}

with open('lightgbm_results.json', 'w') as f:
    json.dump(results_output, f, indent=4)
print("✅ Saved: lightgbm_results.json")

print("\n" + "=" * 80)
print("LIGHTGBM TRAINING COMPLETE!")
print("=" * 80)
