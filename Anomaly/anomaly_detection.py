import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANOMALY DETECTION IN WEATHER DATA")
print("=" * 80)

# ===== LOAD DATA =====
print("\nLoading data...")
df = pd.read_csv('GlobalWeatherRepository.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])
print(f"Loaded {len(df):,} records")

# Select numerical features for anomaly detection
numerical_features = [
    'temperature_celsius', 'humidity', 'precip_mm',
    'wind_kph', 'pressure_mb', 'cloud', 'feels_like_celsius',
    'visibility_km', 'uv_index'
]

# Create subset with complete data
df_subset = df[numerical_features + ['location_name', 'country', 'last_updated']].copy()
df_subset = df_subset.dropna()
print(f"Using {len(df_subset):,} records with complete data")

# ===== METHOD 1: Z-SCORE (STATISTICAL) =====
print("\n" + "=" * 80)
print("METHOD 1: Z-SCORE DETECTION")
print("=" * 80)

z_threshold = 3
z_outliers = pd.DataFrame()

for col in numerical_features:
    z_scores = np.abs((df_subset[col] - df_subset[col].mean()) / df_subset[col].std())
    z_outliers[col] = z_scores > z_threshold

z_outlier_count = z_outliers.sum(axis=1)
df_subset['z_score_outlier'] = z_outlier_count > 0
df_subset['z_score_count'] = z_outlier_count

n_z_outliers = df_subset['z_score_outlier'].sum()
pct_z_outliers = (n_z_outliers / len(df_subset)) * 100

print(f"\nZ-Score Results (threshold = {z_threshold}):")
print(f"  • Outliers detected: {n_z_outliers:,} ({pct_z_outliers:.2f}%)")
print(f"\nOutliers per feature:")
for col in numerical_features:
    n_outliers = z_outliers[col].sum()
    print(f"  • {col:25s}: {n_outliers:,} ({n_outliers/len(df_subset)*100:.2f}%)")

# ===== METHOD 2: IQR (INTERQUARTILE RANGE) =====
print("\n" + "=" * 80)
print("METHOD 2: IQR DETECTION")
print("=" * 80)

iqr_outliers = pd.DataFrame()

for col in numerical_features:
    Q1 = df_subset[col].quantile(0.25)
    Q3 = df_subset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    iqr_outliers[col] = (df_subset[col] < lower_bound) | (df_subset[col] > upper_bound)

iqr_outlier_count = iqr_outliers.sum(axis=1)
df_subset['iqr_outlier'] = iqr_outlier_count > 0
df_subset['iqr_count'] = iqr_outlier_count

n_iqr_outliers = df_subset['iqr_outlier'].sum()
pct_iqr_outliers = (n_iqr_outliers / len(df_subset)) * 100

print(f"\nIQR Results:")
print(f"  • Outliers detected: {n_iqr_outliers:,} ({pct_iqr_outliers:.2f}%)")
print(f"\nOutliers per feature:")
for col in numerical_features:
    n_outliers = iqr_outliers[col].sum()
    print(f"  • {col:25s}: {n_outliers:,} ({n_outliers/len(df_subset)*100:.2f}%)")

# ===== METHOD 3: ISOLATION FOREST =====
print("\n" + "=" * 80)
print("METHOD 3: ISOLATION FOREST")
print("=" * 80)

# Scale features for Isolation Forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_subset[numerical_features])

print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    contamination=0.05,  # Expect 5% outliers
    random_state=42,
    n_jobs=-1
)
iso_predictions = iso_forest.fit_predict(X_scaled)

df_subset['iso_forest_outlier'] = iso_predictions == -1
n_iso_outliers = df_subset['iso_forest_outlier'].sum()
pct_iso_outliers = (n_iso_outliers / len(df_subset)) * 100

print(f"\nIsolation Forest Results:")
print(f"  • Outliers detected: {n_iso_outliers:,} ({pct_iso_outliers:.2f}%)")

# ===== METHOD 4: LOCAL OUTLIER FACTOR (LOF) =====
print("\n" + "=" * 80)
print("METHOD 4: LOCAL OUTLIER FACTOR (LOF)")
print("=" * 80)

print("\nCalculating LOF scores...")
# Use subset for LOF (it's computationally expensive)
sample_size = min(10000, len(df_subset))
sample_indices = np.random.choice(len(df_subset), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_predictions = lof.fit_predict(X_sample)

# Create full array (mark non-sampled as not outlier)
df_subset['lof_outlier'] = False
df_subset.iloc[sample_indices, df_subset.columns.get_loc('lof_outlier')] = (lof_predictions == -1)

n_lof_outliers = df_subset['lof_outlier'].sum()
pct_lof_outliers = (n_lof_outliers / sample_size) * 100

print(f"\nLOF Results (on {sample_size:,} samples):")
print(f"  • Outliers detected: {n_lof_outliers:,} ({pct_lof_outliers:.2f}%)")

# ===== METHOD 5: DBSCAN CLUSTERING =====
print("\n" + "=" * 80)
print("METHOD 5: DBSCAN CLUSTERING")
print("=" * 80)

print("\nPerforming DBSCAN clustering...")
# Use subset for DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=50, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(X_sample)

# Points labeled as -1 are outliers (noise)
df_subset['dbscan_outlier'] = False
df_subset.iloc[sample_indices, df_subset.columns.get_loc('dbscan_outlier')] = (dbscan_labels == -1)

n_dbscan_outliers = df_subset['dbscan_outlier'].sum()
pct_dbscan_outliers = (n_dbscan_outliers / sample_size) * 100

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

print(f"\nDBSCAN Results (on {sample_size:,} samples):")
print(f"  • Clusters found: {n_clusters}")
print(f"  • Outliers detected: {n_dbscan_outliers:,} ({pct_dbscan_outliers:.2f}%)")

# ===== CONSENSUS OUTLIERS =====
print("\n" + "=" * 80)
print("CONSENSUS ANALYSIS")
print("=" * 80)

# Count how many methods flagged each point
df_subset['outlier_votes'] = (
    df_subset['z_score_outlier'].astype(int) +
    df_subset['iqr_outlier'].astype(int) +
    df_subset['iso_forest_outlier'].astype(int) +
    df_subset['lof_outlier'].astype(int) +
    df_subset['dbscan_outlier'].astype(int)
)

# Consensus: flagged by at least 3 methods
df_subset['consensus_outlier'] = df_subset['outlier_votes'] >= 3

n_consensus = df_subset['consensus_outlier'].sum()
pct_consensus = (n_consensus / len(df_subset)) * 100

print(f"\nConsensus Outliers (flagged by ≥3 methods):")
print(f"  • Outliers detected: {n_consensus:,} ({pct_consensus:.2f}%)")

print(f"\nVote distribution:")
vote_counts = df_subset['outlier_votes'].value_counts().sort_index()
for votes, count in vote_counts.items():
    print(f"  • {votes} methods: {count:,} points ({count/len(df_subset)*100:.2f}%)")

# ===== SAVE RESULTS =====
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save anomaly report JSON
anomaly_report = {
    "dataset_info": {
        "total_records": int(len(df_subset)),
        "features_analyzed": numerical_features
    },
    "methods": {
        "z_score": {
            "outliers": int(n_z_outliers),
            "percentage": float(pct_z_outliers),
            "threshold": z_threshold
        },
        "iqr": {
            "outliers": int(n_iqr_outliers),
            "percentage": float(pct_iqr_outliers)
        },
        "isolation_forest": {
            "outliers": int(n_iso_outliers),
            "percentage": float(pct_iso_outliers),
            "contamination": 0.05
        },
        "lof": {
            "outliers": int(n_lof_outliers),
            "percentage": float(pct_lof_outliers),
            "sample_size": int(sample_size)
        },
        "dbscan": {
            "outliers": int(n_dbscan_outliers),
            "percentage": float(pct_dbscan_outliers),
            "clusters": int(n_clusters),
            "sample_size": int(sample_size)
        }
    },
    "consensus": {
        "outliers": int(n_consensus),
        "percentage": float(pct_consensus),
        "threshold": "3 or more methods"
    }
}

with open('anomaly_report.json', 'w') as f:
    json.dump(anomaly_report, f, indent=4)
print("\n✅ Saved: anomaly_report.json")

# Save detailed outliers CSV
outlier_columns = [
    'location_name', 'country', 'last_updated',
    'temperature_celsius', 'humidity', 'precip_mm',
    'z_score_outlier', 'iqr_outlier', 'iso_forest_outlier',
    'lof_outlier', 'dbscan_outlier', 'outlier_votes', 'consensus_outlier'
]

df_outliers = df_subset[df_subset['consensus_outlier']][outlier_columns].copy()
df_outliers = df_outliers.sort_values('outlier_votes', ascending=False)
df_outliers.to_csv('consensus_outliers.csv', index=False)
print(f"✅ Saved: consensus_outliers.csv ({len(df_outliers):,} outliers)")

# Location-wise outlier analysis
location_outliers = df_subset[df_subset['consensus_outlier']].groupby('location_name').size()
location_outliers = location_outliers.sort_values(ascending=False).reset_index()
location_outliers.columns = ['location_name', 'outlier_count']
location_outliers.to_csv('outlier_locations.csv', index=False)
print(f"✅ Saved: outlier_locations.csv (top {len(location_outliers)} locations)")

# ===== VISUALIZATION =====
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Temperature vs Humidity with outliers
ax1 = plt.subplot(2, 3, 1)
normal = df_subset[~df_subset['consensus_outlier']]
outliers = df_subset[df_subset['consensus_outlier']]

ax1.scatter(normal['temperature_celsius'], normal['humidity'], 
           alpha=0.3, s=1, c='blue', label='Normal')
ax1.scatter(outliers['temperature_celsius'], outliers['humidity'], 
           alpha=0.7, s=10, c='red', label='Outliers')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Humidity (%)')
ax1.set_title('Temperature vs Humidity\n(Consensus Outliers)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Method comparison
ax2 = plt.subplot(2, 3, 2)
methods = ['Z-Score', 'IQR', 'Iso Forest', 'LOF', 'DBSCAN', 'Consensus']
counts = [n_z_outliers, n_iqr_outliers, n_iso_outliers, 
         n_lof_outliers, n_dbscan_outliers, n_consensus]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax2.barh(methods, counts, color=colors)
ax2.set_xlabel('Number of Outliers')
ax2.set_title('Outlier Count by Detection Method')
ax2.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, 
            f'{counts[i]:,}', ha='left', va='center', fontsize=9)

# Plot 3: Outlier votes distribution
ax3 = plt.subplot(2, 3, 3)
vote_counts_plot = df_subset['outlier_votes'].value_counts().sort_index()
ax3.bar(vote_counts_plot.index, vote_counts_plot.values, color='steelblue', edgecolor='black')
ax3.set_xlabel('Number of Methods Flagging as Outlier')
ax3.set_ylabel('Count')
ax3.set_title('Consensus Distribution')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(6))

# Plot 4: Precipitation outliers
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(normal['precip_mm'], normal['temperature_celsius'], 
           alpha=0.3, s=1, c='blue', label='Normal')
ax4.scatter(outliers['precip_mm'], outliers['temperature_celsius'], 
           alpha=0.7, s=10, c='red', label='Outliers')
ax4.set_xlabel('Precipitation (mm)')
ax4.set_ylabel('Temperature (°C)')
ax4.set_title('Precipitation vs Temperature\n(Consensus Outliers)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Top locations with outliers
ax5 = plt.subplot(2, 3, 5)
top_locations = location_outliers.head(15)
ax5.barh(range(len(top_locations)), top_locations['outlier_count'], color='coral')
ax5.set_yticks(range(len(top_locations)))
ax5.set_yticklabels(top_locations['location_name'], fontsize=8)
ax5.set_xlabel('Number of Outliers')
ax5.set_title('Top 15 Locations with Most Outliers')
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

# Plot 6: Wind speed vs Pressure with outliers
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(normal['wind_kph'], normal['pressure_mb'], 
           alpha=0.3, s=1, c='blue', label='Normal')
ax6.scatter(outliers['wind_kph'], outliers['pressure_mb'], 
           alpha=0.7, s=10, c='red', label='Outliers')
ax6.set_xlabel('Wind Speed (kph)')
ax6.set_ylabel('Pressure (mb)')
ax6.set_title('Wind Speed vs Pressure\n(Consensus Outliers)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Saved: anomaly_visualization.png")

print("\n" + "=" * 80)
print("ANOMALY DETECTION COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • anomaly_report.json - Summary statistics")
print("  • consensus_outliers.csv - Detailed outlier records")
print("  • outlier_locations.csv - Location-wise outlier counts")
print("  • anomaly_visualization.png - Visual analysis")
print("\nNext step: Run anomaly_analysis.py for deeper insights!")