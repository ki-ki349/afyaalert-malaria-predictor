# create_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("🚀 Starting to create synthetic malaria dataset for Kenya...")

# Create date range for 10 years (2015-2024)
dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='M')

# List of Kenyan counties (main ones)
counties = [
    'Nairobi', 'Mombasa', 'Kisumu', 'Kiambu', 'Kakamega', 
    'Uasin Gishu', 'Meru', 'Machakos', 'Kilifi', 'Kwale'
]

print(f"📊 Creating data for {len(counties)} counties over 10 years...")

data = []

for county in counties:
    print(f"   Processing {county}...")
    for date in dates:
        # Extract month for seasonal patterns
        month = date.month
        year = date.year
        
        # Seasonal factor: higher during rainy seasons
        # Long rains: March-May, Short rains: October-December
        if month in [3, 4, 5]:  # Long rains
            seasonal_factor = 1.8
        elif month in [10, 11, 12]:  # Short rains
            seasonal_factor = 1.5
        else:  # Dry season
            seasonal_factor = 0.7
        
        # Base malaria rates per county (different for different regions)
        base_rate = {
            'Nairobi': 30,      # Urban, lower risk
            'Mombasa': 120,      # Coastal, higher risk
            'Kisumu': 200,       # Lake region, highest risk
            'Kiambu': 45,        # Central highlands
            'Kakamega': 180,     # Western, high risk
            'Uasin Gishu': 40,   # Highlands
            'Meru': 70,          # Eastern slopes
            'Machakos': 55,      # Eastern region
            'Kilifi': 150,       # Coastal
            'Kwale': 140         # Coastal
        }.get(county, 80)  # Default for any missing county
        
        # Add some year-over-year trend (slight increase over time)
        time_trend = 1.0 + (year - 2015) * 0.02
        
        # Add random variation
        random_variation = np.random.normal(1.0, 0.2)
        
        # Calculate cases
        cases = int(base_rate * seasonal_factor * time_trend * random_variation)
        cases = max(5, cases)  # Ensure at least 5 cases
        
        # Generate climate data based on season
        if month in [3, 4, 5]:  # Long rains
            rainfall = np.random.uniform(150, 350)
            humidity = np.random.uniform(75, 95)
        elif month in [10, 11, 12]:  # Short rains
            rainfall = np.random.uniform(100, 250)
            humidity = np.random.uniform(70, 90)
        else:  # Dry season
            rainfall = np.random.uniform(10, 60)
            humidity = np.random.uniform(50, 70)
        
        # Temperature varies by county and season
        if county in ['Nairobi', 'Kiambu', 'Uasin Gishu', 'Meru']:  # Highland areas
            base_temp = 20
        elif county in ['Mombasa', 'Kilifi', 'Kwale']:  # Coastal
            base_temp = 28
        else:  # Other regions
            base_temp = 24
        
        # Seasonal temperature variation
        if month in [6, 7, 8]:  # Cooler months
            temp = base_temp - np.random.uniform(2, 4)
        elif month in [1, 2, 3]:  # Hotter months
            temp = base_temp + np.random.uniform(2, 4)
        else:
            temp = base_temp + np.random.uniform(-1, 2)
        
        data.append({
            'county': county,
            'date': date.strftime('%Y-%m'),
            'year': year,
            'month': month,
            'confirmed_cases': cases,
            'rainfall_mm': round(rainfall, 1),
            'temp_mean_c': round(temp, 1),
            'humidity_pct': round(humidity, 1)
        })

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"✅ Created {len(df)} records")

# Show basic statistics
print("\n📈 Dataset Overview:")
print(f"   Total records: {len(df)}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Counties: {', '.join(df['county'].unique())}")
print(f"\n   Case statistics:")
print(f"   - Average cases per month: {df['confirmed_cases'].mean():.1f}")
print(f"   - Minimum cases: {df['confirmed_cases'].min()}")
print(f"   - Maximum cases: {df['confirmed_cases'].max()}")

# Save to CSV
output_path = 'data/raw/malaria_dataset.csv'
df.to_csv(output_path, index=False)
print(f"\n💾 Dataset saved to: {output_path}")

# Show first few rows
print("\n👀 First 5 rows of data:")
print(df.head())

print("\n✅✅✅ PHASE 2 COMPLETE! Dataset created successfully!")