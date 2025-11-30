import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import time

print("=" * 60)
print("HYPERPARAMETER TUNING FOR WEATHER PREDICTION MODEL")
print("=" * 60)

# ===== LOAD DATA =====
print("\nLoading training data...")
X_train = np.load('X_train_scaled.npy')
y_train = np.load('y_train_scaled.npy')

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ===== DEFINE HYPERPARAMETER SEARCH SPACE =====
print("\n" + "=" * 60)
print("DEFINING HYPERPARAMETER SEARCH SPACE")
print("=" * 60)

param_distributions = {
    'estimator__n_estimators': randint(100, 500),           # Number of trees
    'estimator__max_depth': [10, 20, 30, 40, None],         # Maximum depth
    'estimator__min_samples_split': randint(2, 20),         # Min samples to split
    'estimator__min_samples_leaf': randint(1, 10),          # Min samples per leaf
    'estimator__max_features': ['sqrt', 'log2', None],      # Features per split
    'estimator__bootstrap': [True, False]                    # Bootstrap samples
}

print("\nSearch space defined:")
for param, values in param_distributions.items():
    print(f"  {param}: {values}")

# ===== CREATE BASE MODEL =====
base_model = MultiOutputRegressor(
    RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0)
)

# ===== PERFORM RANDOMIZED SEARCH =====
print("\n" + "=" * 60)
print("STARTING HYPERPARAMETER SEARCH")
print("=" * 60)
print("\nConfiguration:")
print("  - Search method: RandomizedSearchCV")
print("  - Number of iterations: 20")
print("  - Cross-validation folds: 3")
print("  - Scoring metric: Negative Mean Squared Error")
print("  - Parallel jobs: All CPU cores")
print("\n⏳ This will take 10-20 minutes...")
print("-" * 60)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=20,                    # Try 20 different combinations
    cv=3,                         # 3-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1,                    # Use all CPU cores
    verbose=2,                    # Show progress
    random_state=42,
    return_train_score=True
)

# Fit and time the search
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

print("\n" + "=" * 60)
print(f"✅ TUNING COMPLETE in {(end_time - start_time) / 60:.2f} minutes")
print("=" * 60)

# ===== EXTRACT AND CLEAN BEST PARAMETERS =====
best_params_raw = random_search.best_params_

# Remove 'estimator__' prefix for cleaner JSON
best_params = {
    key.replace('estimator__', ''): value 
    for key, value in best_params_raw.items()
}

# ===== DISPLAY RESULTS =====
print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS FOUND")
print("=" * 60)

for param, value in best_params.items():
    print(f"  {param:20s}: {value}")

print(f"\n  Best CV Score (Neg MSE): {random_search.best_score_:.4f}")

# ===== SAVE TO JSON =====
output_data = {
    "best_hyperparameters": best_params,
    "cv_score": float(random_search.best_score_),
    "tuning_metadata": {
        "n_iterations": 20,
        "cv_folds": 3,
        "scoring_metric": "neg_mean_squared_error",
        "training_samples": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "tuning_time_minutes": round((end_time - start_time) / 60, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
}

with open('best_hyperparameters.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("\n" + "=" * 60)
print("SAVED HYPERPARAMETERS")
print("=" * 60)
print("✅ File: best_hyperparameters.json")
print("\nYou can now use these parameters in train_model.py")

# ===== OPTIONALLY SAVE CV RESULTS =====
cv_results = {
    "mean_test_scores": random_search.cv_results_['mean_test_score'].tolist(),
    "std_test_scores": random_search.cv_results_['std_test_score'].tolist(),
    "params": [
        {k.replace('estimator__', ''): v for k, v in params.items()} 
        for params in random_search.cv_results_['params']
    ]
}

with open('cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent=4)

print("✅ File: cv_results.json (detailed CV results)")

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETE!")
print("=" * 60)