"""Check data distribution and temporal splits"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from data_loader import preprocess_building_data, load_electricity_data

print("="*90)
print("  DATA DISTRIBUTION ANALYSIS")
print("="*90)

# Load data
electricity, metadata, valid_buildings = load_electricity_data()

# Load weather
weather_path = 'data/raw/building-data-genome-project-2/data/weather/weather.csv'
weather = pd.read_csv(weather_path)
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
weather = weather.set_index('timestamp')

# Process Colin building
building_id = 'Rat_education_Colin'
site_id = metadata[metadata['building_id'] == building_id]['site_id'].values[0]
weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
weather_building = weather_building.reindex(electricity.index)

data, scaler = preprocess_building_data(electricity, building_id, weather_building)

print(f"\n1. FULL DATASET:")
print(f"   Total samples: {len(data)}")
print(f"   Date range: {data.index.min()} to {data.index.max()}")
print(f"   Energy - Mean: {data['energy'].mean():.2f}, Std: {data['energy'].std():.2f}")
print(f"   Energy - Range: [{data['energy'].min():.2f}, {data['energy'].max():.2f}]")

# Split data same as training (60/20/20)
n = len(data)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_data = data.iloc[:train_end]
val_data = data.iloc[train_end:val_end]
test_data = data.iloc[val_end:]

print(f"\n2. DATA SPLITS (60/20/20):")
print(f"\n   TRAIN SET ({len(train_data)} samples):")
print(f"   Date range: {train_data.index.min()} to {train_data.index.max()}")
print(f"   Energy - Mean: {train_data['energy'].mean():.2f}, Std: {train_data['energy'].std():.2f}")
print(f"   Energy - Range: [{train_data['energy'].min():.2f}, {train_data['energy'].max():.2f}]")

print(f"\n   VALIDATION SET ({len(val_data)} samples):")
print(f"   Date range: {val_data.index.min()} to {val_data.index.max()}")
print(f"   Energy - Mean: {val_data['energy'].mean():.2f}, Std: {val_data['energy'].std():.2f}")
print(f"   Energy - Range: [{val_data['energy'].min():.2f}, {val_data['energy'].max():.2f}]")

print(f"\n   TEST SET ({len(test_data)} samples):")
print(f"   Date range: {test_data.index.min()} to {test_data.index.max()}")
print(f"   Energy - Mean: {test_data['energy'].mean():.2f}, Std: {test_data['energy'].std():.2f}")
print(f"   Energy - Range: [{test_data['energy'].min():.2f}, {test_data['energy'].max():.2f}]")

print(f"\n3. DISTRIBUTION COMPARISON:")
train_mean = train_data['energy'].mean()
val_mean = val_data['energy'].mean()
test_mean = test_data['energy'].mean()

print(f"   Mean shift (train → val): {((val_mean - train_mean) / train_mean * 100):+.1f}%")
print(f"   Mean shift (train → test): {((test_mean - train_mean) / train_mean * 100):+.1f}%")
print(f"   Mean shift (val → test): {((test_mean - val_mean) / val_mean * 100):+.1f}%")

train_std = train_data['energy'].std()
val_std = val_data['energy'].std()
test_std = test_data['energy'].std()

print(f"\n   Std shift (train → val): {((val_std - train_std) / train_std * 100):+.1f}%")
print(f"   Std shift (train → test): {((test_std - train_std) / train_std * 100):+.1f}%")

print(f"\n4. SEASONAL ANALYSIS:")
# Check which seasons are in each split
for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
    months = split_data.index.month.unique()
    seasons = []
    if any(m in [12, 1, 2] for m in months): seasons.append('Winter')
    if any(m in [3, 4, 5] for m in months): seasons.append('Spring')
    if any(m in [6, 7, 8] for m in months): seasons.append('Summer')
    if any(m in [9, 10, 11] for m in months): seasons.append('Fall')
    print(f"   {split_name}: Months {sorted(months)}, Seasons: {', '.join(seasons)}")

print(f"\n5. KEY FINDINGS:")
findings = []

# Check for mean shift
if abs((test_mean - train_mean) / train_mean) > 0.1:
    findings.append(f"⚠ Large mean shift between train ({train_mean:.1f}) and test ({test_mean:.1f})")
    findings.append("  This causes negative R² - model trained on different distribution!")

# Check for std shift
if abs((test_std - train_std) / train_std) > 0.2:
    findings.append(f"⚠ Large std shift between train ({train_std:.1f}) and test ({test_std:.1f})")

# Check if training stopped early
findings.append("⚠ Training stopped at epoch 7 with val_RMSE = 34.1 kWh")
findings.append("  But best val_RMSE was 29.1 kWh at epoch 6")
findings.append("  Model loaded from epoch 6, but still has RMSE ~29 kWh on validation")

# Check absolute performance
if val_mean > 0:
    relative_error = (29.1 / val_mean) * 100
    findings.append(f"  Relative error: {relative_error:.1f}% of mean consumption")
    if relative_error > 30:
        findings.append("  ⚠ This is > 30% relative error - model barely learned!")

if not findings:
    findings.append("✓ No major distribution issues detected")

for finding in findings:
    print(f"   {finding}")

print(f"\n6. WHY NEGATIVE R²:")
print(f"   - Model RMSE ≈ 29 kWh on validation")
print(f"   - If test distribution is different, predictions will be systematically off")
print(f"   - R² = 1 - (SS_res / SS_tot)")
print(f"   - If predictions are worse than mean, SS_res > SS_tot → R² < 0")
print(f"   - SOLUTION: Model needs to train LONGER and learn better patterns")

print("\n" + "="*90)
