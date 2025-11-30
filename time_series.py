import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Data_exploration import load_weather_data  # Replace 'your_main_file' with actual filename
df = load_weather_data()
# Plot trends over time
targets = ['temperature_celsius', 'humidity', 'precip_mm']
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Aggregate by date (daily average)
daily_avg = df.groupby(df['last_updated'].dt.date)[targets].mean()

daily_avg['temperature_celsius'].plot(ax=axes[0], color='orange')
axes[0].set_title('Temperature Trend Over Time')
axes[0].set_ylabel('Celsius')

daily_avg['humidity'].plot(ax=axes[1], color='blue')
axes[1].set_title('Humidity Trend Over Time')
axes[1].set_ylabel('Percentage')

daily_avg['precip_mm'].plot(ax=axes[2], color='green')
axes[2].set_title('Precipitation Trend Over Time')
axes[2].set_ylabel('mm')

plt.tight_layout()
plt.savefig('time_trends.png', dpi=300)
plt.show()

print("✅ Time trends saved!")
