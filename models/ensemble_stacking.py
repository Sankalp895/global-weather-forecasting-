import numpy as np
import pickle
import json
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("ENSEMBLE MODEL - STACKING")
print("=" * 80)

print("\nLoading data...")
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train_scaled.npy')
y_test = np.load('y_test_scaled.npy')

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\n" + "=" * 80)
print("LOADING BASE MODELS")
print("=" * 80)

with open('weather_model.pkl', 'rb') as f:
    rf_multioutput = pickle.load(f)

with open('xgboost_models.pkl', 'rb') as f:
    xgb_models = pickle.load(f)
    
with open('lightgbm_models.pkl', 'rb') as f:
    lgb_models = pickle.load(f)

print("✅ Loaded Random Forest")
print("✅ Loaded XGBoost")
print("✅ Loaded LightGBM")

print("\n" + "=" * 80)
print("CREATING STACKING ENSEMBLE")
print("=" * 80)

ensemble_models = {}

for i, target_name in enumerate(['temperature', 'humidity']):
    print(f"\nBuilding ensemble for {target_name}...")
    
    estimators = [
        ('rf', rf_multioutput.estimators_[i]),
        ('xgb', xgb_models[target_name]),
        ('lgb', lgb_models[target_name])
    ]
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    print(f"  Training stacking ensemble...")
    stacking_model.fit(X_train, y_train[:, i])
    print(f"  ✅ Ensemble trained for {target_name}")
    
    ensemble_models[target_name] = stacking_model

print("\n" + "=" * 80)
print("MAKING PREDICTIONS")
print("=" * 80)

predictions_train = np.column_stack([
    ensemble_models['temperature'].predict(X_train),
    ensemble_models['humidity'].predict(X_train)
])

predictions_test = np.column_stack([
    ensemble_models['temperature'].predict(X_test),
    ensemble_models['humidity'].predict(X_test)
])

y_train_actual = scaler_y.inverse_transform(y_train)
y_train_pred = scaler_y.inverse_transform(predictions_train)
y_test_actual = scaler_y.inverse_transform(y_test)
y_test_pred = scaler_y.inverse_transform(predictions_test)

print("\n" + "=" * 80)
print("ENSEMBLE EVALUATION")
print("=" * 80)

targets = ['Temperature (C)', 'Humidity (%)']
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
    
    print(f"  TRAIN - RMSE: {train_rmse:.3f} | MAE: {train_mae:.3f} | R2: {train_r2:.3f}")
    print(f"  TEST  - RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R2: {test_r2:.3f}")
    
    target_key = target.lower().replace(' ', '_').replace('(', '').replace(')', '')
    evaluation_results[target_key] = {
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)}
    }

print("\n" + "=" * 80)
print("COMPARISON WITH INDIVIDUAL MODELS")
print("=" * 80)

with open('model_results.json', 'r') as f:
    rf_results = json.load(f)
with open('xgboost_results.json', 'r') as f:
    xgb_results = json.load(f)
with open('lightgbm_results.json', 'r') as f:
    lgb_results = json.load(f)

print("\nTemperature Test RMSE:")
print(f"  Random Forest: {rf_results['evaluation_metrics']['temperature_c']['test']['rmse']:.3f}")
print(f"  XGBoost:       {xgb_results['evaluation_metrics']['temperature_c']['test']['rmse']:.3f}")
print(f"  LightGBM:      {lgb_results['evaluation_metrics']['temperature_c']['test']['rmse']:.3f}")
print(f"  ENSEMBLE:      {evaluation_results['temperature_c']['test']['rmse']:.3f}")

print("\nHumidity Test RMSE:")
print(f"  Random Forest: {rf_results['evaluation_metrics']['humidity_%']['test']['rmse']:.3f}")
print(f"  XGBoost:       {xgb_results['evaluation_metrics']['humidity_%']['test']['rmse']:.3f}")
print(f"  LightGBM:      {lgb_results['evaluation_metrics']['humidity_%']['test']['rmse']:.3f}")
print(f"  ENSEMBLE:      {evaluation_results['humidity_%']['test']['rmse']:.3f}")

improvement_temp = rf_results['evaluation_metrics']['temperature_c']['test']['rmse'] - evaluation_results['temperature_c']['test']['rmse']
improvement_hum = rf_results['evaluation_metrics']['humidity_%']['test']['rmse'] - evaluation_results['humidity_%']['test']['rmse']

print("\nImprovement over best single model (Random Forest):")
print(f"  Temperature: {improvement_temp:.3f} RMSE reduction ({improvement_temp/rf_results['evaluation_metrics']['temperature_c']['test']['rmse']*100:.2f}%)")
print(f"  Humidity:    {improvement_hum:.3f} RMSE reduction ({improvement_hum/rf_results['evaluation_metrics']['humidity_%']['test']['rmse']*100:.2f}%)")

print("\n" + "=" * 80)
print("SAVING ENSEMBLE MODEL")
print("=" * 80)

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_models, f)
print("✅ Saved: ensemble_model.pkl")

results_output = {
    "model_type": "Stacking Ensemble (RF + XGBoost + LightGBM)",
    "base_models": ["Random Forest", "XGBoost", "LightGBM"],
    "meta_model": "Ridge Regression",
    "evaluation_metrics": evaluation_results,
    "model_info": {
        "n_training_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "targets": ["temperature_celsius", "humidity"]
    }
}

with open('ensemble_results.json', 'w') as f:
    json.dump(results_output, f, indent=4)
print("✅ Saved: ensemble_results.json")

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].scatter(y_train_actual[:, 0], y_train_pred[:, 0], alpha=0.3, s=1, c='blue')
axes[0, 0].plot([y_train_actual[:, 0].min(), y_train_actual[:, 0].max()],
                [y_train_actual[:, 0].min(), y_train_actual[:, 0].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Temperature (C)')
axes[0, 0].set_ylabel('Predicted Temperature (C)')
axes[0, 0].set_title('Temperature - Training Set')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test_actual[:, 0], y_test_pred[:, 0], alpha=0.3, s=1, c='orange')
axes[0, 1].plot([y_test_actual[:, 0].min(), y_test_actual[:, 0].max()],
                [y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Temperature (C)')
axes[0, 1].set_ylabel('Predicted Temperature (C)')
axes[0, 1].set_title('Temperature - Test Set')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_train_actual[:, 1], y_train_pred[:, 1], alpha=0.3, s=1, c='blue')
axes[1, 0].plot([y_train_actual[:, 1].min(), y_train_actual[:, 1].max()],
                [y_train_actual[:, 1].min(), y_train_actual[:, 1].max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Humidity (%)')
axes[1, 0].set_ylabel('Predicted Humidity (%)')
axes[1, 0].set_title('Humidity - Training Set')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(y_test_actual[:, 1], y_test_pred[:, 1], alpha=0.3, s=1, c='green')
axes[1, 1].plot([y_test_actual[:, 1].min(), y_test_actual[:, 1].max()],
                [y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Humidity (%)')
axes[1, 1].set_ylabel('Predicted Humidity (%)')
axes[1, 1].set_title('Humidity - Test Set')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ensemble_predictions.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

models_list = ['Random Forest', 'XGBoost', 'LightGBM', 'Ensemble']
temp_rmse_list = [
    rf_results['evaluation_metrics']['temperature_c']['test']['rmse'],
    xgb_results['evaluation_metrics']['temperature_c']['test']['rmse'],
    lgb_results['evaluation_metrics']['temperature_c']['test']['rmse'],
    evaluation_results['temperature_c']['test']['rmse']
]
hum_rmse_list = [
    rf_results['evaluation_metrics']['humidity_%']['test']['rmse'],
    xgb_results['evaluation_metrics']['humidity_%']['test']['rmse'],
    lgb_results['evaluation_metrics']['humidity_%']['test']['rmse'],
    evaluation_results['humidity_%']['test']['rmse']
]

colors = ['steelblue', 'coral', 'lightgreen', 'gold']
axes2[0].bar(models_list, temp_rmse_list, color=colors, edgecolor='black')
axes2[0].set_ylabel('RMSE (C)')
axes2[0].set_title('Temperature Test RMSE Comparison')
axes2[0].grid(True, alpha=0.3, axis='y')
axes2[0].tick_params(axis='x', rotation=45)

axes2[1].bar(models_list, hum_rmse_list, color=colors, edgecolor='black')
axes2[1].set_ylabel('RMSE (%)')
axes2[1].set_title('Humidity Test RMSE Comparison')
axes2[1].grid(True, alpha=0.3, axis='y')
axes2[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ensemble_comparison.png")

report = f"""
===============================================================================
ENSEMBLE MODEL REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
===============================================================================

ENSEMBLE ARCHITECTURE: Stacking
Base Models: Random Forest + XGBoost + LightGBM
Meta-Model: Ridge Regression (combines base model predictions)

===============================================================================
PERFORMANCE COMPARISON
===============================================================================

TEMPERATURE PREDICTION (Test RMSE):
  Random Forest:  {rf_results['evaluation_metrics']['temperature_c']['test']['rmse']:.3f} C
  XGBoost:        {xgb_results['evaluation_metrics']['temperature_c']['test']['rmse']:.3f} C
  LightGBM:       {lgb_results['evaluation_metrics']['temperature_c']['test']['rmse']:.3f} C
  ENSEMBLE:       {evaluation_results['temperature_c']['test']['rmse']:.3f} C
  
  Improvement: {improvement_temp:.3f} C ({improvement_temp/rf_results['evaluation_metrics']['temperature_c']['test']['rmse']*100:.2f}%)

HUMIDITY PREDICTION (Test RMSE):
  Random Forest:  {rf_results['evaluation_metrics']['humidity_%']['test']['rmse']:.3f}%
  XGBoost:        {xgb_results['evaluation_metrics']['humidity_%']['test']['rmse']:.3f}%
  LightGBM:       {lgb_results['evaluation_metrics']['humidity_%']['test']['rmse']:.3f}%
  ENSEMBLE:       {evaluation_results['humidity_%']['test']['rmse']:.3f}%
  
  Improvement: {improvement_hum:.3f}% ({improvement_hum/rf_results['evaluation_metrics']['humidity_%']['test']['rmse']*100:.2f}%)

===============================================================================
KEY INSIGHTS
===============================================================================

1. The stacking ensemble combines strengths of all three base models
2. Meta-model (Ridge) learns optimal weights for each base model
3. Ensemble typically reduces variance and improves generalization
4. Best for production deployment when maximum accuracy is needed

TRADE-OFFS:
  + Higher accuracy than individual models
  - Slower inference (needs all 3 base models)
  - More complex deployment

===============================================================================
ASSIGNMENT COMPLETION STATUS
===============================================================================

DONE: Task 1 - Advanced EDA (Anomaly Detection)
DONE: Task 2 - Forecasting with Multiple Models (6 models)
DONE: Task 3 - Ensemble Methods (Stacking Ensemble)
TODO: Task 4 - Unique Analyses

===============================================================================
"""

with open('ensemble_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✅ Saved: ensemble_report.txt")

print("\n" + "=" * 80)
print("ENSEMBLE MODEL COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • ensemble_model.pkl")
print("  • ensemble_results.json")
print("  • ensemble_predictions.png")
print("  • ensemble_comparison.png")
print("  • ensemble_report.txt")
print("\n🎉 Task 3: Ensemble Methods - COMPLETE!")
