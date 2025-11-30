import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

print("=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

print("\nLoading model results...")

model_files = {
    'Random Forest': 'model_results.json',
    'XGBoost': 'xgboost_results.json',
    'LightGBM': 'lightgbm_results.json',
    'Linear': 'linear_results.json',
    'Ridge': 'ridge_results.json',
    'Lasso': 'lasso_results.json'
}

model_results = {}
for model_name, filename in model_files.items():
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            model_results[model_name] = json.load(f)
        print(f"✅ Loaded: {model_name}")
    else:
        print(f"⚠️  Missing: {filename} - skipping {model_name}")

if not model_results:
    print("\n❌ No model results found! Please train models first.")
    exit(1)

print(f"\n✅ Found {len(model_results)} models")

print("\n" + "=" * 80)
print("CREATING COMPARISON TABLE")
print("=" * 80)

comparison_data = []

for model_name, results in model_results.items():
    metrics = results['evaluation_metrics']
    
    temp_train = metrics['temperature_c']['train']
    temp_test = metrics['temperature_c']['test']
    hum_train = metrics['humidity_%']['train']
    hum_test = metrics['humidity_%']['test']
    
    comparison_data.append({
        'Model': model_name,
        'Temp_Train_RMSE': temp_train['rmse'],
        'Temp_Train_MAE': temp_train['mae'],
        'Temp_Train_R2': temp_train['r2'],
        'Temp_Test_RMSE': temp_test['rmse'],
        'Temp_Test_MAE': temp_test['mae'],
        'Temp_Test_R2': temp_test['r2'],
        'Hum_Train_RMSE': hum_train['rmse'],
        'Hum_Train_MAE': hum_train['mae'],
        'Hum_Train_R2': hum_train['r2'],
        'Hum_Test_RMSE': hum_test['rmse'],
        'Hum_Test_MAE': hum_test['mae'],
        'Hum_Test_R2': hum_test['r2']
    })

df_comparison = pd.DataFrame(comparison_data)

print("\n📊 TEMPERATURE PREDICTION COMPARISON")
print("=" * 80)
print(f"{'Model':<15} {'Train RMSE':<12} {'Train MAE':<12} {'Train R²':<10} "
      f"{'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10}")
print("-" * 80)

for _, row in df_comparison.iterrows():
    print(f"{row['Model']:<15} {row['Temp_Train_RMSE']:>10.3f}  {row['Temp_Train_MAE']:>10.3f}  "
          f"{row['Temp_Train_R2']:>8.4f}  {row['Temp_Test_RMSE']:>10.3f}  "
          f"{row['Temp_Test_MAE']:>10.3f}  {row['Temp_Test_R2']:>8.4f}")

print("\n💧 HUMIDITY PREDICTION COMPARISON")
print("=" * 80)
print(f"{'Model':<15} {'Train RMSE':<12} {'Train MAE':<12} {'Train R²':<10} "
      f"{'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10}")
print("-" * 80)

for _, row in df_comparison.iterrows():
    print(f"{row['Model']:<15} {row['Hum_Train_RMSE']:>10.3f}  {row['Hum_Train_MAE']:>10.3f}  "
          f"{row['Hum_Train_R2']:>8.4f}  {row['Hum_Test_RMSE']:>10.3f}  "
          f"{row['Hum_Test_MAE']:>10.3f}  {row['Hum_Test_R2']:>8.4f}")

print("\n" + "=" * 80)
print("BEST MODELS")
print("=" * 80)

best_temp_rmse = df_comparison.loc[df_comparison['Temp_Test_RMSE'].idxmin()]
best_hum_rmse = df_comparison.loc[df_comparison['Hum_Test_RMSE'].idxmin()]

print(f"\n🌡️  Best Temperature Model (by Test RMSE):")
print(f"   {best_temp_rmse['Model']} - RMSE: {best_temp_rmse['Temp_Test_RMSE']:.3f}, R²: {best_temp_rmse['Temp_Test_R2']:.4f}")

print(f"\n💧 Best Humidity Model (by Test RMSE):")
print(f"   {best_hum_rmse['Model']} - RMSE: {best_hum_rmse['Hum_Test_RMSE']:.3f}, R²: {best_hum_rmse['Hum_Test_R2']:.4f}")

df_comparison.to_csv('model_comparison_table.csv', index=False)
print("\n✅ Saved: model_comparison_table.csv")

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 3, 1)
models = df_comparison['Model']
temp_rmse = df_comparison['Temp_Test_RMSE']
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
bars = ax1.barh(models, temp_rmse, color=colors, edgecolor='black')
ax1.set_xlabel('RMSE (°C)')
ax1.set_title('Temperature - Test RMSE\n(Lower is Better)')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}', ha='left', va='center', fontsize=9)

ax2 = plt.subplot(2, 3, 2)
temp_r2 = df_comparison['Temp_Test_R2']
bars = ax2.barh(models, temp_r2, color=colors, edgecolor='black')
ax2.set_xlabel('R² Score')
ax2.set_title('Temperature - Test R²\n(Higher is Better)')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', ha='left', va='center', fontsize=9)

ax3 = plt.subplot(2, 3, 3)
hum_rmse = df_comparison['Hum_Test_RMSE']
bars = ax3.barh(models, hum_rmse, color=colors, edgecolor='black')
ax3.set_xlabel('RMSE (%)')
ax3.set_title('Humidity - Test RMSE\n(Lower is Better)')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    ax3.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}', ha='left', va='center', fontsize=9)

ax4 = plt.subplot(2, 3, 4)
hum_r2 = df_comparison['Hum_Test_R2']
bars = ax4.barh(models, hum_r2, color=colors, edgecolor='black')
ax4.set_xlabel('R² Score')
ax4.set_title('Humidity - Test R²\n(Higher is Better)')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', ha='left', va='center', fontsize=9)

ax5 = plt.subplot(2, 3, 5)
x = np.arange(len(models))
width = 0.35
train_mae = df_comparison['Temp_Train_MAE']
test_mae = df_comparison['Temp_Test_MAE']
ax5.bar(x - width/2, train_mae, width, label='Train', alpha=0.8, color='steelblue')
ax5.bar(x + width/2, test_mae, width, label='Test', alpha=0.8, color='coral')
ax5.set_ylabel('MAE (°C)')
ax5.set_title('Temperature MAE: Train vs Test')
ax5.set_xticks(x)
ax5.set_xticklabels(models, rotation=45, ha='right')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

ax6 = plt.subplot(2, 3, 6)
train_mae_hum = df_comparison['Hum_Train_MAE']
test_mae_hum = df_comparison['Hum_Test_MAE']
ax6.bar(x - width/2, train_mae_hum, width, label='Train', alpha=0.8, color='steelblue')
ax6.bar(x + width/2, test_mae_hum, width, label='Test', alpha=0.8, color='coral')
ax6.set_ylabel('MAE (%)')
ax6.set_title('Humidity MAE: Train vs Test')
ax6.set_xticks(x)
ax6.set_xticklabels(models, rotation=45, ha='right')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_performance_comparison.png")

print("\n" + "=" * 80)
print("GENERATING RECOMMENDATION REPORT")
print("=" * 80)

recommendation = f"""
\
MODEL COMPARISON & RECOMMENDATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
\

MODELS EVALUATED: {len(model_results)}
  {', '.join(model_results.keys())}

═══════════════════════════════════════════════════════════════════════════════
TEMPERATURE PREDICTION
═══════════════════════════════════════════════════════════════════════════════

🥇 BEST MODEL: {best_temp_rmse['Model']}
   Test RMSE: {best_temp_rmse['Temp_Test_RMSE']:.3f}°C
   Test MAE:  {best_temp_rmse['Temp_Test_MAE']:.3f}°C
   Test R²:   {best_temp_rmse['Temp_Test_R2']:.4f}

Rankings (by Test RMSE):
{chr(10).join([f"  {i+1}. {row['Model']}: RMSE={row['Temp_Test_RMSE']:.3f}°C, R²={row['Temp_Test_R2']:.4f}" 
               for i, (_, row) in enumerate(df_comparison.sort_values('Temp_Test_RMSE').iterrows())])}

═══════════════════════════════════════════════════════════════════════════════
HUMIDITY PREDICTION
═══════════════════════════════════════════════════════════════════════════════

🥇 BEST MODEL: {best_hum_rmse['Model']}
   Test RMSE: {best_hum_rmse['Hum_Test_RMSE']:.3f}%
   Test MAE:  {best_hum_rmse['Hum_Test_MAE']:.3f}%
   Test R²:   {best_hum_rmse['Hum_Test_R2']:.4f}

Rankings (by Test RMSE):
{chr(10).join([f"  {i+1}. {row['Model']}: RMSE={row['Hum_Test_RMSE']:.3f}%, R²={row['Hum_Test_R2']:.4f}" 
               for i, (_, row) in enumerate(df_comparison.sort_values('Hum_Test_RMSE').iterrows())])}

═══════════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

🎯 PRODUCTION: Use {best_temp_rmse['Model']} for temperature, {best_hum_rmse['Model']} for humidity

"""

with open('best_model_recommendation.txt', 'w', encoding='utf-8') as f:
    f.write(recommendation)
print("✅ Saved: best_model_recommendation.txt")

print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • model_comparison_table.csv")
print("  • model_performance_comparison.png")
print("  • best_model_recommendation.txt")
print("\n🎉 Task 2: Forecasting with Multiple Models - COMPLETE!")
