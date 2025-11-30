import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data using the reusable function
from Data_exploration import load_weather_data  # Replace 'your_main_file' with actual filename
df = load_weather_data()

# Define targets
targets = ['temperature_celsius', 'humidity', 'precip_mm']

# Create distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df['temperature_celsius'].hist(bins=50, ax=axes[0], color='orange', edgecolor='black')
axes[0].set_title('Temperature Distribution')
axes[0].set_xlabel('Celsius')
axes[0].set_ylabel('Frequency')

df['humidity'].hist(bins=50, ax=axes[1], color='blue', edgecolor='black')
axes[1].set_title('Humidity Distribution')
axes[1].set_xlabel('Percentage')
axes[1].set_ylabel('Frequency')

df['precip_mm'].hist(bins=50, ax=axes[2], color='green', edgecolor='black')
axes[2].set_title('Precipitation Distribution')
axes[2].set_xlabel('mm')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=300)
plt.show()

print("✅ Histograms saved to target_distributions.png")
