import numpy as np
import pickle
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 80)
print("TRAINING LINEAR MODELS (Linear, Ridge, Lasso)")
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

# ===== TRAIN MODELS =====
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {}
results = {}

# 1. Linear Regression
print("\n1️⃣  Training Linear Regression...")
lr = MultiOutputRegressor(LinearRegression())
lr.fit(X_train, y_train)
print("✅ Linear Regression complete")

# 2. Ridge Regression (L2 regularization)
print("\n2️⃣  Training Ridge Regression (alpha=1.0)...")
ridge = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
ridge.fit(X_train, y_train)
print("✅ Ridge Regression complete")

# 3. Lasso Regression (L1 regularization)
print("\n3️⃣  Training Lasso Regression (alpha=0.1)...")
lasso = MultiOutputRegressor(Lasso(alpha=0.1, random_state=42, max_iter=10000))
lasso.fit(X_train, y_train)
print("✅ Lasso Regression complete")

models['linear'] = lr
models['ridge'] = ridge
models['lasso'] = lasso

# ===== EVALUATE ALL MODELS =====
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} REGRESSION")
    print(f"{'='*80}")
    
    # Predictions
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)
    
    # Inverse transform
    y_train_actual = scaler_y.inverse_transform(y_train)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    
    # Evaluate
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
    
    results[model_name] = {
        "model_type": f"{model_name.capitalize()} Regression",
        "evaluation_metrics": evaluation_results,
        "model_info": {
            "n_training_samples": int(X_train.shape[0]),
            "n_test_samples": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "targets": ["temperature_celsius", "humidity"]
        }
    }

# ===== SAVE MODELS AND RESULTS =====
print("\n" + "=" * 80)
print("SAVING MODELS AND RESULTS")
print("=" * 80)

with open('linear_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✅ Saved: linear_models.pkl")

for model_name, result_data in results.items():
    filename = f'{model_name}_results.json'
    with open(filename, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"✅ Saved: {filename}")

print("\n" + "=" * 80)
print("LINEAR MODELS TRAINING COMPLETE!")
print("=" * 80)
