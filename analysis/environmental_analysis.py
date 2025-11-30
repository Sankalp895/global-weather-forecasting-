import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import json

print("=" * 80)
print("ENVIRONMENTAL IMPACT ANALYSIS")
print("=" * 80)

print("\nLoading data...")
df = pd.read_csv('GlobalWeatherRepository.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

print(f"Total records: {len(df):,}")

air_quality_cols = ['air_quality_PM2.5', 'air_quality_PM10', 'air_quality_Carbon_Monoxide', 
                    'air_quality_Ozone', 'air_quality_us-epa-index']

available_cols = [col for col in air_quality_cols if col in df.columns]
print(f"Available air quality columns: {len(available_cols)}")
for col in available_cols:
    non_null = df[col].notna().sum()
    pct = (non_null / len(df)) * 100
    print(f"  {col:<35s}: {non_null:>7,} records ({pct:>5.1f}%)")

if len(available_cols) == 0:
    print("\n⚠️  No air quality data found in dataset!")
    print("Creating alternative environmental analysis...")

print("\n" + "=" * 80)
print("TEMPERATURE & HUMIDITY ENVIRONMENTAL IMPACT")
print("=" * 80)

df['heat_index'] = df['temperature_celsius'] + (0.5555 * (df['humidity'] - 10))
df['comfort_level'] = 'Comfortable'
df.loc[df['heat_index'] > 30, 'comfort_level'] = 'Hot'
df.loc[df['heat_index'] > 40, 'comfort_level'] = 'Very Hot'
df.loc[df['heat_index'] < 10, 'comfort_level'] = 'Cold'
df.loc[df['heat_index'] < 0, 'comfort_level'] = 'Very Cold'

comfort_dist = df['comfort_level'].value_counts()
print("\nHeat Index Distribution (Environmental Comfort):")
for level, count in comfort_dist.items():
    pct = (count / len(df)) * 100
    print(f"  {level:<15s}: {count:>7,} records ({pct:>5.1f}%)")

print("\n" + "=" * 80)
print("EXTREME WEATHER CONDITIONS")
print("=" * 80)

extreme_heat = df[df['temperature_celsius'] > 35]
extreme_cold = df[df['temperature_celsius'] < -10]
high_humidity = df[df['humidity'] > 90]
low_humidity = df[df['humidity'] < 20]

print(f"\nExtreme Heat (>35°C): {len(extreme_heat):,} records ({len(extreme_heat)/len(df)*100:.1f}%)")
print(f"Extreme Cold (<-10°C): {len(extreme_cold):,} records ({len(extreme_cold)/len(df)*100:.1f}%)")
print(f"High Humidity (>90%): {len(high_humidity):,} records ({len(high_humidity)/len(df)*100:.1f}%)")
print(f"Low Humidity (<20%): {len(low_humidity):,} records ({len(low_humidity)/len(df)*100:.1f}%)")

if len(extreme_heat) > 0:
    print(f"\nTop locations with extreme heat:")
    heat_locations = extreme_heat.groupby('location_name').size().sort_values(ascending=False).head(10)
    for loc, count in heat_locations.items():
        print(f"  {loc:<30s}: {count:>4} occurrences")

print("\n" + "=" * 80)
print("PRECIPITATION & ENVIRONMENTAL IMPACT")
print("=" * 80)

df['precip_category'] = 'No Rain'
df.loc[df['precip_mm'] > 0, 'precip_category'] = 'Light Rain'
df.loc[df['precip_mm'] > 5, 'precip_category'] = 'Moderate Rain'
df.loc[df['precip_mm'] > 20, 'precip_category'] = 'Heavy Rain'
df.loc[df['precip_mm'] > 50, 'precip_category'] = 'Extreme Rain'

precip_dist = df['precip_category'].value_counts()
print("\nPrecipitation Distribution:")
for cat, count in precip_dist.items():
    pct = (count / len(df)) * 100
    print(f"  {cat:<15s}: {count:>7,} records ({pct:>5.1f}%)")

print("\n" + "=" * 80)
print("WIND & AIR CIRCULATION")
print("=" * 80)

df['wind_category'] = 'Calm'
df.loc[df['wind_kph'] > 5, 'wind_category'] = 'Light Breeze'
df.loc[df['wind_kph'] > 20, 'wind_category'] = 'Moderate Wind'
df.loc[df['wind_kph'] > 40, 'wind_category'] = 'Strong Wind'
df.loc[df['wind_kph'] > 60, 'wind_category'] = 'Very Strong Wind'

wind_dist = df['wind_category'].value_counts()
print("\nWind Speed Distribution:")
for cat, count in wind_dist.items():
    pct = (count / len(df)) * 100
    print(f"  {cat:<20s}: {count:>7,} records ({pct:>5.1f}%)")

avg_wind_by_country = df.groupby('country')['wind_kph'].mean().sort_values(ascending=False).head(10)
print(f"\nWindiest Countries (by average wind speed):")
for country, wind in avg_wind_by_country.items():
    print(f"  {country:<25s}: {wind:>6.1f} kph")

print("\n" + "=" * 80)
print("UV INDEX ENVIRONMENTAL IMPACT")
print("=" * 80)

if 'uv_index' in df.columns:
    df['uv_category'] = 'Low'
    df.loc[df['uv_index'] >= 3, 'uv_category'] = 'Moderate'
    df.loc[df['uv_index'] >= 6, 'uv_category'] = 'High'
    df.loc[df['uv_index'] >= 8, 'uv_category'] = 'Very High'
    df.loc[df['uv_index'] >= 11, 'uv_category'] = 'Extreme'
    
    uv_dist = df['uv_category'].value_counts()
    print("\nUV Index Distribution:")
    for cat, count in uv_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {cat:<15s}: {count:>7,} records ({pct:>5.1f}%)")
    
    high_uv = df[df['uv_index'] >= 8]
    if len(high_uv) > 0:
        print(f"\nHigh UV Risk Locations:")
        uv_locations = high_uv.groupby('location_name').size().sort_values(ascending=False).head(10)
        for loc, count in uv_locations.items():
            print(f"  {loc:<30s}: {count:>4} high UV days")

print("\n" + "=" * 80)
print("WEATHER CORRELATIONS")
print("=" * 80)

corr_pairs = [
    ('temperature_celsius', 'humidity'),
    ('temperature_celsius', 'wind_kph'),
    ('humidity', 'precip_mm'),
    ('pressure_mb', 'temperature_celsius')
]

print("\nEnvironmental Variable Correlations:")
for var1, var2 in corr_pairs:
    if var1 in df.columns and var2 in df.columns:
        valid_data = df[[var1, var2]].dropna()
        if len(valid_data) > 100:
            corr, p_value = pearsonr(valid_data[var1], valid_data[var2])
            print(f"  {var1:<25s} vs {var2:<20s}: r = {corr:>6.3f} (p={p_value:.4f})")

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 3, 1)
comfort_dist.plot(kind='bar', ax=ax1, color='coral', edgecolor='black')
ax1.set_xlabel('Comfort Level')
ax1.set_ylabel('Number of Records')
ax1.set_title('Heat Index Distribution')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = plt.subplot(2, 3, 2)
ax2.scatter(df['temperature_celsius'], df['humidity'], 
           alpha=0.3, s=1, c='steelblue')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Humidity (%)')
ax2.set_title('Temperature vs Humidity Relationship')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
precip_dist.plot(kind='bar', ax=ax3, color='skyblue', edgecolor='black')
ax3.set_xlabel('Precipitation Category')
ax3.set_ylabel('Number of Records')
ax3.set_title('Precipitation Distribution')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

ax4 = plt.subplot(2, 3, 4)
wind_dist.plot(kind='barh', ax=ax4, color='lightgreen', edgecolor='black')
ax4.set_xlabel('Number of Records')
ax4.set_title('Wind Speed Distribution')
ax4.grid(True, alpha=0.3, axis='x')

ax5 = plt.subplot(2, 3, 5)
ax5.scatter(df['wind_kph'], df['temperature_celsius'], 
           alpha=0.3, s=1, c='orange')
ax5.set_xlabel('Wind Speed (kph)')
ax5.set_ylabel('Temperature (°C)')
ax5.set_title('Wind Speed vs Temperature')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
if 'uv_index' in df.columns:
    uv_dist.plot(kind='pie', ax=ax6, autopct='%1.1f%%', startangle=90)
    ax6.set_ylabel('')
    ax6.set_title('UV Index Distribution')
else:
    ax6.hist(df['heat_index'], bins=50, color='red', edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Heat Index')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Heat Index Distribution')
    ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('environmental_impact_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: environmental_impact_analysis.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

ax = axes2[0, 0]
monthly_avg = df.groupby(df['last_updated'].dt.month)['temperature_celsius'].mean()
ax.plot(range(1, 13), monthly_avg, marker='o', linewidth=2, color='red')
ax.set_xlabel('Month')
ax.set_ylabel('Average Temperature (°C)')
ax.set_title('Seasonal Temperature Variation')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.grid(True, alpha=0.3)

ax = axes2[0, 1]
monthly_precip = df.groupby(df['last_updated'].dt.month)['precip_mm'].mean()
ax.bar(range(1, 13), monthly_precip, color='blue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Month')
ax.set_ylabel('Average Precipitation (mm)')
ax.set_title('Seasonal Precipitation Pattern')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.grid(True, alpha=0.3, axis='y')

ax = axes2[1, 0]
country_heat_index = df.groupby('country')['heat_index'].mean().sort_values(ascending=False).head(15)
ax.barh(range(len(country_heat_index)), country_heat_index.values, color='orangered', edgecolor='black')
ax.set_yticks(range(len(country_heat_index)))
ax.set_yticklabels(country_heat_index.index, fontsize=8)
ax.set_xlabel('Average Heat Index')
ax.set_title('Countries with Highest Heat Index')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

ax = axes2[1, 1]
extreme_counts = pd.Series({
    'Extreme Heat': len(extreme_heat),
    'Extreme Cold': len(extreme_cold),
    'High Humidity': len(high_humidity),
    'Low Humidity': len(low_humidity)
})
ax.bar(range(len(extreme_counts)), extreme_counts.values, 
       color=['red', 'blue', 'cyan', 'orange'], edgecolor='black')
ax.set_xticks(range(len(extreme_counts)))
ax.set_xticklabels(extreme_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Records')
ax.set_title('Extreme Weather Events Count')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('environmental_trends.png', dpi=300, bbox_inches='tight')
print("✅ Saved: environmental_trends.png")

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

environmental_summary = {
    "extreme_weather": {
        "extreme_heat_count": int(len(extreme_heat)),
        "extreme_cold_count": int(len(extreme_cold)),
        "high_humidity_count": int(len(high_humidity)),
        "low_humidity_count": int(len(low_humidity))
    },
    "comfort_distribution": {level: int(count) for level, count in comfort_dist.items()},
    "precipitation_distribution": {cat: int(count) for cat, count in precip_dist.items()},
    "wind_distribution": {cat: int(count) for cat, count in wind_dist.items()},
    "average_heat_index": float(df['heat_index'].mean())
}

with open('environmental_impact_summary.json', 'w') as f:
    json.dump(environmental_summary, f, indent=4)
print("✅ Saved: environmental_impact_summary.json")

report = f"""
===============================================================================
ENVIRONMENTAL IMPACT ANALYSIS REPORT
===============================================================================

EXTREME WEATHER CONDITIONS:
  Extreme Heat (>35°C): {len(extreme_heat):,} records ({len(extreme_heat)/len(df)*100:.1f}%)
  Extreme Cold (<-10°C): {len(extreme_cold):,} records ({len(extreme_cold)/len(df)*100:.1f}%)
  High Humidity (>90%): {len(high_humidity):,} records ({len(high_humidity)/len(df)*100:.1f}%)
  Low Humidity (<20%): {len(low_humidity):,} records ({len(low_humidity)/len(df)*100:.1f}%)

HEAT INDEX ANALYSIS:
  Average Heat Index: {df['heat_index'].mean():.1f}
  Most common comfort level: {comfort_dist.index[0]}

PRECIPITATION PATTERNS:
  No Rain: {precip_dist.get('No Rain', 0):,} records
  Heavy/Extreme Rain: {precip_dist.get('Heavy Rain', 0) + precip_dist.get('Extreme Rain', 0):,} records

WIND PATTERNS:
  Windiest Country: {avg_wind_by_country.index[0]} ({avg_wind_by_country.iloc[0]:.1f} kph avg)
  Calm conditions: {wind_dist.get('Calm', 0):,} records

KEY FINDINGS:
1. Temperature and humidity strongly affect environmental comfort
2. Extreme weather events occur in {(len(extreme_heat) + len(extreme_cold))/len(df)*100:.1f}% of records
3. Wind patterns vary significantly by location
4. Precipitation shows seasonal patterns

ENVIRONMENTAL IMPLICATIONS:
- High heat index locations may face health risks
- Extreme weather locations need climate adaptation
- Wind patterns affect air quality and pollution dispersion
- Precipitation variability impacts water resources

===============================================================================
"""

with open('environmental_impact_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✅ Saved: environmental_impact_report.txt")

print("\n" + "=" * 80)
print("ENVIRONMENTAL IMPACT ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • environmental_impact_analysis.png")
print("  • environmental_trends.png")
print("  • environmental_impact_summary.json")
print("  • environmental_impact_report.txt")
print("\n🎉 Additional Unique Analysis COMPLETE!")
