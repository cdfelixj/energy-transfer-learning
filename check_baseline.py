import pandas as pd
import numpy as np

# Load electricity data
electricity = pd.read_csv('data/raw/building-data-genome-project-2/data/meters/raw/electricity.csv', index_col=0)

# Check Eagle building stats
building = 'Eagle_education_Raul'
values = electricity[building].dropna()

print(f"=== Building: {building} ===")
print(f"Mean: {values.mean():.2f} kWh")
print(f"Std:  {values.std():.2f} kWh")
print(f"Min:  {values.min():.2f} kWh")
print(f"Max:  {values.max():.2f} kWh")
print(f"Median: {values.median():.2f} kWh")

# Load baseline metrics
print("\n=== Baseline Model Performance (version_2) ===")
metrics_baseline = pd.read_csv('src/lightning_logs/version_2/metrics.csv')
test_metrics = metrics_baseline[metrics_baseline['test_loss'].notna()].iloc[0]
print(f"Test Loss (MSE):  {test_metrics['test_loss']:.2f}")
print(f"Test MAE:         {test_metrics['test_mae']:.2f} kWh")
print(f"Test RMSE:        {test_metrics['test_rmse']:.2f} kWh")

# Calculate MAPE (Mean Absolute Percentage Error)
mape = (test_metrics['test_mae'] / values.mean()) * 100
print(f"\nMAPE: {mape:.2f}%")
print(f"(Average prediction error as % of mean consumption)")

# Load transfer metrics
print("\n=== Transfer Model Performance (version_3) ===")
metrics_transfer = pd.read_csv('src/lightning_logs/version_3/metrics.csv')
test_metrics_transfer = metrics_transfer[metrics_transfer['test_loss'].notna()].iloc[0]
print(f"Test Loss (MSE):  {test_metrics_transfer['test_loss']:.2f}")
print(f"Test MAE:         {test_metrics_transfer['test_mae']:.2f} kWh")
print(f"Test RMSE:        {test_metrics_transfer['test_rmse']:.2f} kWh")

# Check target building
target_building = 'Lamb_education_Lazaro'
target_values = electricity[target_building].dropna()
mape_transfer = (test_metrics_transfer['test_mae'] / target_values.mean()) * 100
print(f"\nMAPE: {mape_transfer:.2f}%")
print(f"Target building mean: {target_values.mean():.2f} kWh")

print("\n=== Summary ===")
print(f"Baseline model predicts with ±{test_metrics['test_mae']:.2f} kWh error ({mape:.1f}% MAPE)")
print(f"Transfer model predicts with ±{test_metrics_transfer['test_mae']:.2f} kWh error ({mape_transfer:.1f}% MAPE)")
