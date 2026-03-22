# src/features/feature_engineering.py
import pandas as pd
import numpy as np
import os

print("🔄 Starting Feature Engineering...")

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/metadata', exist_ok=True)

# Load the raw data
df = pd.read_csv('data/raw/malaria_dataset.csv')
print(f"✅ Loaded {len(df)} records")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['county', 'date'])

print("🔨 Creating lag features...")
# Create lag features (cases from previous months)
for lag in [1, 2, 3]:
    df[f'cases_lag_{lag}'] = df.groupby('county')['confirmed_cases'].shift(lag)
    print(f"   ✅ Created cases_lag_{lag}")

print("🔨 Creating rolling averages...")
# Rolling averages for climate variables
df['rainfall_3m_avg'] = df.groupby('county')['rainfall_mm'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df['temp_3m_avg'] = df.groupby('county')['temp_mean_c'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df['humidity_3m_avg'] = df.groupby('county')['humidity_pct'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
print("   ✅ Created rolling averages (3 months)")

print("🔨 Creating seasonal features...")
# Extract month and create seasonal indicators
df['month'] = pd.to_datetime(df['date']).dt.month

def get_season(month):
    if month in [3, 4, 5]:
        return 'Long_Rains'
    elif month in [10, 11, 12]:
        return 'Short_Rains'
    else:
        return 'Dry'

df['season'] = df['month'].apply(get_season)
print("   ✅ Created season column")

# Create dummy variables for season
season_dummies = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_dummies], axis=1)
print("   ✅ Created season dummy variables")

print("🔨 Creating target variable (risk level)...")
print("🔨 Creating target variable (risk level)...")
# Create target variable based on percentiles (per county)
def assign_risk(group):
    # Calculate percentiles
    p50 = group['confirmed_cases'].quantile(0.5)
    p85 = group['confirmed_cases'].quantile(0.85)
    
    # Create conditions - FIXED VERSION
    conditions = [
        group['confirmed_cases'] < p50,
        (group['confirmed_cases'] >= p50) & (group['confirmed_cases'] < p85),
        group['confirmed_cases'] >= p85
    ]
    choices = ['Low', 'Medium', 'High']
    
    # Use numpy select with explicit object dtype
    result = np.select(conditions, choices, default='Medium')
    return pd.Series(result, index=group.index, dtype='object')

df['risk_level'] = df.groupby('county').apply(assign_risk).reset_index(level=0, drop=True)
print("   ✅ Created risk_level (Low/Medium/High)")

# Drop rows with NaN from lag features
original_len = len(df)
df = df.dropna()
dropped = original_len - len(df)
print(f"✅ Dropped {dropped} rows with NaN values (first few months per county)")

# Save the processed dataset
output_path = 'data/processed/feature_engineered_dataset.csv'
df.to_csv(output_path, index=False)
print(f"💾 Saved feature engineered dataset to: {output_path}")

# Create and save a data dictionary
data_dict = {
    'county': 'Name of the county',
    'date': 'Month and year (YYYY-MM)',
    'confirmed_cases': 'Number of confirmed malaria cases',
    'rainfall_mm': 'Rainfall in millimeters',
    'temp_mean_c': 'Average temperature in Celsius',
    'humidity_pct': 'Relative humidity percentage',
    'cases_lag_1': 'Cases from previous month',
    'cases_lag_2': 'Cases from 2 months ago',
    'cases_lag_3': 'Cases from 3 months ago',
    'rainfall_3m_avg': 'Average rainfall over last 3 months',
    'temp_3m_avg': 'Average temperature over last 3 months',
    'humidity_3m_avg': 'Average humidity over last 3 months',
    'month': 'Month number (1-12)',
    'season': 'Season name (Dry/Long_Rains/Short_Rains)',
    'season_Dry': 'Dummy variable: 1 if Dry season',
    'season_Long_Rains': 'Dummy variable: 1 if Long Rains',
    'season_Short_Rains': 'Dummy variable: 1 if Short Rains',
    'risk_level': 'Target variable: Low/Medium/High risk'
}

# Save data dictionary
dict_df = pd.DataFrame(list(data_dict.items()), columns=['Feature', 'Description'])
dict_df.to_csv('data/metadata/data_dictionary.csv', index=False)
print("📝 Saved data dictionary to: data/metadata/data_dictionary.csv")

# Show sample of processed data
print("\n👀 Sample of processed data (first 3 rows):")
print(df[['county', 'date', 'confirmed_cases', 'cases_lag_1', 'rainfall_3m_avg', 'risk_level']].head(3))

print("\n📊 Final dataset shape:", df.shape)