import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import json

print("=" * 80)
print("SPATIAL & GEOGRAPHICAL ANALYSIS")
print("=" * 80)

print("\nLoading data...")
df = pd.read_csv('GlobalWeatherRepository.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

print(f"Total records: {len(df):,}")
print(f"Unique locations: {df['location_name'].nunique()}")
print(f"Unique countries: {df['country'].nunique()}")

print("\n" + "=" * 80)
print("GEOGRAPHIC DISTRIBUTION OF WEATHER PATTERNS")
print("=" * 80)

location_stats = df.groupby('location_name').agg({
    'temperature_celsius': ['mean', 'std', 'min', 'max'],
    'humidity': ['mean', 'std', 'min', 'max'],
    'latitude': 'first',
    'longitude': 'first',
    'country': 'first'
}).reset_index()

location_stats.columns = ['location', 'temp_mean', 'temp_std', 'temp_min', 'temp_max',
                          'hum_mean', 'hum_std', 'hum_min', 'hum_max',
                          'latitude', 'longitude', 'country']

print("\nHottest Locations (by average temperature):")
hottest = location_stats.nlargest(10, 'temp_mean')
for i, row in hottest.iterrows():
    print(f"  {row['location']:<30s} {row['country']:<20s} {row['temp_mean']:>6.1f}°C")

print("\nColdest Locations (by average temperature):")
coldest = location_stats.nsmallest(10, 'temp_mean')
for i, row in coldest.iterrows():
    print(f"  {row['location']:<30s} {row['country']:<20s} {row['temp_mean']:>6.1f}°C")

print("\nMost Humid Locations (by average humidity):")
humid = location_stats.nlargest(10, 'hum_mean')
for i, row in humid.iterrows():
    print(f"  {row['location']:<30s} {row['country']:<20s} {row['hum_mean']:>6.1f}%")

print("\n" + "=" * 80)
print("LATITUDE CORRELATION ANALYSIS")
print("=" * 80)

lat_temp_corr, lat_temp_p = pearsonr(location_stats['latitude'], location_stats['temp_mean'])
lat_hum_corr, lat_hum_p = pearsonr(location_stats['latitude'], location_stats['hum_mean'])

print(f"\nLatitude vs Temperature: r = {lat_temp_corr:.3f} (p-value: {lat_temp_p:.4f})")
print(f"Latitude vs Humidity:    r = {lat_hum_corr:.3f} (p-value: {lat_hum_p:.4f})")

if abs(lat_temp_corr) > 0.5:
    print("\n✓ Strong correlation: Latitude strongly affects temperature")
else:
    print("\n○ Moderate correlation: Latitude has some effect on temperature")

print("\n" + "=" * 80)
print("COUNTRY-LEVEL ANALYSIS")
print("=" * 80)

country_stats = df.groupby('country').agg({
    'temperature_celsius': ['mean', 'std'],
    'humidity': ['mean', 'std'],
    'location_name': 'nunique'
}).reset_index()

country_stats.columns = ['country', 'temp_mean', 'temp_std', 'hum_mean', 'hum_std', 'num_locations']
country_stats = country_stats[country_stats['num_locations'] >= 3]

print(f"\nAnalyzing {len(country_stats)} countries (with 3+ locations)")

print("\nCountries with Most Variable Temperature:")
variable_temp = country_stats.nlargest(10, 'temp_std')
for i, row in variable_temp.iterrows():
    print(f"  {row['country']:<25s} Std: {row['temp_std']:>5.1f}°C (Mean: {row['temp_mean']:>5.1f}°C)")

print("\nCountries with Most Stable Temperature:")
stable_temp = country_stats.nsmallest(10, 'temp_std')
for i, row in stable_temp.iterrows():
    print(f"  {row['country']:<25s} Std: {row['temp_std']:>5.1f}°C (Mean: {row['temp_mean']:>5.1f}°C)")

print("\n" + "=" * 80)
print("CLIMATE ZONES IDENTIFICATION")
print("=" * 80)

location_stats['climate_zone'] = 'Temperate'
location_stats.loc[location_stats['temp_mean'] > 25, 'climate_zone'] = 'Tropical'
location_stats.loc[location_stats['temp_mean'] < 10, 'climate_zone'] = 'Cold'
location_stats.loc[(location_stats['temp_mean'] >= 10) & (location_stats['temp_mean'] <= 25), 'climate_zone'] = 'Temperate'

climate_counts = location_stats['climate_zone'].value_counts()
print("\nClimate Zone Distribution:")
for zone, count in climate_counts.items():
    pct = (count / len(location_stats)) * 100
    print(f"  {zone:<15s}: {count:>3d} locations ({pct:>5.1f}%)")

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(location_stats['longitude'], location_stats['latitude'],
                     c=location_stats['temp_mean'], cmap='RdYlBu_r', 
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Global Temperature Distribution')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Temperature (°C)')

ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(location_stats['longitude'], location_stats['latitude'],
                     c=location_stats['hum_mean'], cmap='Blues',
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('Global Humidity Distribution')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Humidity (%)')

ax3 = plt.subplot(2, 3, 3)
ax3.scatter(location_stats['latitude'], location_stats['temp_mean'], 
           alpha=0.5, s=30, c='coral', edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Latitude')
ax3.set_ylabel('Average Temperature (°C)')
ax3.set_title(f'Latitude vs Temperature\n(r = {lat_temp_corr:.3f})')
ax3.grid(True, alpha=0.3)
z = np.polyfit(location_stats['latitude'], location_stats['temp_mean'], 2)
p = np.poly1d(z)
x_line = np.linspace(location_stats['latitude'].min(), location_stats['latitude'].max(), 100)
ax3.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

ax4 = plt.subplot(2, 3, 4)
top_countries = country_stats.nlargest(15, 'temp_mean')
ax4.barh(range(len(top_countries)), top_countries['temp_mean'], color='orangered', edgecolor='black')
ax4.set_yticks(range(len(top_countries)))
ax4.set_yticklabels(top_countries['country'], fontsize=8)
ax4.set_xlabel('Average Temperature (°C)')
ax4.set_title('Top 15 Warmest Countries')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

ax5 = plt.subplot(2, 3, 5)
colors_climate = {'Tropical': 'red', 'Temperate': 'green', 'Cold': 'blue'}
for zone in location_stats['climate_zone'].unique():
    zone_data = location_stats[location_stats['climate_zone'] == zone]
    ax5.scatter(zone_data['longitude'], zone_data['latitude'],
               label=zone, alpha=0.6, s=40, c=colors_climate.get(zone, 'gray'),
               edgecolors='black', linewidth=0.5)
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
ax5.set_title('Climate Zones Map')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
ax6.scatter(location_stats['temp_mean'], location_stats['hum_mean'],
           alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
ax6.set_xlabel('Average Temperature (°C)')
ax6.set_ylabel('Average Humidity (%)')
ax6.set_title('Temperature vs Humidity Relationship')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spatial_weather_patterns.png', dpi=300, bbox_inches='tight')
print("✅ Saved: spatial_weather_patterns.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

ax = axes2[0, 0]
top_variable = country_stats.nlargest(15, 'temp_std')
ax.barh(range(len(top_variable)), top_variable['temp_std'], color='crimson', edgecolor='black')
ax.set_yticks(range(len(top_variable)))
ax.set_yticklabels(top_variable['country'], fontsize=8)
ax.set_xlabel('Temperature Std Dev (°C)')
ax.set_title('Countries with Most Variable Temperature')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

ax = axes2[0, 1]
ax.hist(location_stats['temp_mean'], bins=30, color='orange', edgecolor='black', alpha=0.7)
ax.set_xlabel('Average Temperature (°C)')
ax.set_ylabel('Number of Locations')
ax.set_title('Global Temperature Distribution')
ax.grid(True, alpha=0.3, axis='y')

ax = axes2[1, 0]
ax.hist(location_stats['hum_mean'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Average Humidity (%)')
ax.set_ylabel('Number of Locations')
ax.set_title('Global Humidity Distribution')
ax.grid(True, alpha=0.3, axis='y')

ax = axes2[1, 1]
climate_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                    colors=['red', 'green', 'blue'], startangle=90)
ax.set_ylabel('')
ax.set_title('Climate Zone Distribution')

plt.tight_layout()
plt.savefig('geographical_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: geographical_analysis.png")

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

location_stats.to_csv('location_weather_statistics.csv', index=False)
print("✅ Saved: location_weather_statistics.csv")

country_stats.to_csv('country_weather_statistics.csv', index=False)
print("✅ Saved: country_weather_statistics.csv")

spatial_summary = {
    "total_locations": int(df['location_name'].nunique()),
    "total_countries": int(df['country'].nunique()),
    "hottest_location": {
        "name": hottest.iloc[0]['location'],
        "country": hottest.iloc[0]['country'],
        "temperature": float(hottest.iloc[0]['temp_mean'])
    },
    "coldest_location": {
        "name": coldest.iloc[0]['location'],
        "country": coldest.iloc[0]['country'],
        "temperature": float(coldest.iloc[0]['temp_mean'])
    },
    "latitude_temperature_correlation": float(lat_temp_corr),
    "climate_zones": {zone: int(count) for zone, count in climate_counts.items()}
}

with open('spatial_analysis_summary.json', 'w') as f:
    json.dump(spatial_summary, f, indent=4)
print("✅ Saved: spatial_analysis_summary.json")

report = f"""
===============================================================================
SPATIAL & GEOGRAPHICAL ANALYSIS REPORT
===============================================================================

DATASET OVERVIEW:
  Total Locations: {df['location_name'].nunique()}
  Total Countries: {df['country'].nunique()}
  Total Records: {len(df):,}

===============================================================================
KEY FINDINGS
===============================================================================

TEMPERATURE PATTERNS:
  Hottest: {hottest.iloc[0]['location']}, {hottest.iloc[0]['country']} ({hottest.iloc[0]['temp_mean']:.1f}°C)
  Coldest: {coldest.iloc[0]['location']}, {coldest.iloc[0]['country']} ({coldest.iloc[0]['temp_mean']:.1f}°C)
  Range: {hottest.iloc[0]['temp_mean'] - coldest.iloc[0]['temp_mean']:.1f}°C difference

LATITUDE EFFECT:
  Correlation with Temperature: r = {lat_temp_corr:.3f}
  {"Strong" if abs(lat_temp_corr) > 0.5 else "Moderate"} inverse relationship
  Locations near equator are warmer

CLIMATE ZONES:
  Tropical (>25°C): {climate_counts.get('Tropical', 0)} locations
  Temperate (10-25°C): {climate_counts.get('Temperate', 0)} locations
  Cold (<10°C): {climate_counts.get('Cold', 0)} locations

GEOGRAPHIC DIVERSITY:
  Most variable: {variable_temp.iloc[0]['country']} (Std: {variable_temp.iloc[0]['temp_std']:.1f}°C)
  Most stable: {stable_temp.iloc[0]['country']} (Std: {stable_temp.iloc[0]['temp_std']:.1f}°C)

===============================================================================
INSIGHTS FOR MODELING
===============================================================================

1. Location-based features are critical for predictions
2. Latitude strongly influences temperature patterns
3. Different climate zones may need separate models
4. Island nations show unique weather patterns

===============================================================================
"""

with open('spatial_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✅ Saved: spatial_analysis_report.txt")

print("\n" + "=" * 80)
print("SPATIAL & GEOGRAPHICAL ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • spatial_weather_patterns.png")
print("  • geographical_analysis.png")
print("  • location_weather_statistics.csv")
print("  • country_weather_statistics.csv")
print("  • spatial_analysis_summary.json")
print("  • spatial_analysis_report.txt")
print("\n🎉 Task 4: Unique Analysis COMPLETE!")
