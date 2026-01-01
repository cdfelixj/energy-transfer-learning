"""
Comprehensive Model Evaluation Script
Evaluates baseline and transfer learning model performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import torch
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import preprocess_building_data, create_dataloaders
from models import EnergyLSTM


def evaluate_model(model, test_loader, model_name="Model"):
    """Evaluate a model on test data and return detailed metrics"""
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            
            predictions.extend(y_hat.squeeze().numpy())
            actuals.extend(y.squeeze().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Calculate MAPE (avoiding division by zero and near-zero values)
    # Only calculate MAPE if mean is reasonable (> 1.0)
    mask = np.abs(actuals) > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = np.nan  # MAPE not meaningful for very small values
    
    # Calculate additional metrics
    median_ae = np.median(np.abs(actuals - predictions))
    max_error = np.max(np.abs(actuals - predictions))
    
    results = {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'r2': r2,
        'mape': mape,
        'median_ae': median_ae,
        'max_error': max_error,
        'predictions': predictions,
        'actuals': actuals,
        'mean_actual': np.mean(actuals),
        'std_actual': np.std(actuals)
    }
    
    return results


def print_evaluation_results(results):
    """Print formatted evaluation results"""
    print(f"\n{'='*70}")
    print(f"  {results['model_name']}")
    print(f"{'='*70}")
    print(f"Mean Absolute Error (MAE):        {results['mae']:.4f} kWh")
    print(f"Root Mean Squared Error (RMSE):   {results['rmse']:.4f} kWh")
    print(f"Mean Squared Error (MSE):         {results['mse']:.4f}")
    print(f"R² Score:                          {results['r2']:.4f}")
    if not np.isnan(results['mape']):
        print(f"Mean Absolute Percentage Error:    {results['mape']:.2f}%")
    else:
        print(f"Mean Absolute Percentage Error:    N/A (values too small)")
    print(f"Median Absolute Error:             {results['median_ae']:.4f} kWh")
    print(f"Maximum Error:                     {results['max_error']:.4f} kWh")
    print(f"\nActual Energy Statistics:")
    print(f"  Mean:  {results['mean_actual']:.2f} kWh")
    print(f"  Std:   {results['std_actual']:.2f} kWh")
    print(f"{'='*70}")


def compare_models(baseline_results, transfer_results):
    """Compare baseline and transfer model performance"""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Baseline':<20} {'Transfer':<20} {'Improvement'}")
    print(f"{'-'*70}")
    
    metrics = [
        ('MAE (kWh)', 'mae'),
        ('RMSE (kWh)', 'rmse'),
        ('MAPE (%)', 'mape'),
        ('R² Score', 'r2'),
        ('Median AE (kWh)', 'median_ae')
    ]
    
    for metric_name, metric_key in metrics:
        baseline_val = baseline_results[metric_key]
        transfer_val = transfer_results[metric_key]
        
        # For R2, higher is better; for others, lower is better
        if metric_key == 'r2':
            improvement = ((transfer_val - baseline_val) / abs(baseline_val)) * 100
            better = "✓ Better" if transfer_val > baseline_val else "✗ Worse"
        else:
            improvement = ((baseline_val - transfer_val) / baseline_val) * 100
            better = "✓ Better" if transfer_val < baseline_val else "✗ Worse"
        
        print(f"{metric_name:<30} {baseline_val:>18.4f} {transfer_val:>18.4f} {improvement:>8.1f}% {better}")
    
    print(f"{'='*70}")


def main():
    # Load selected buildings
    print("Loading building configuration...")
    selected_buildings = pd.read_csv('data/processed/selected_buildings.csv')
    source_building = selected_buildings['building_id'].iloc[0]
    target_building = selected_buildings['building_id'].iloc[1]
    
    print(f"Source Building: {source_building}")
    print(f"Target Building: {target_building}")
    
    # Load electricity and weather data
    print("\nLoading data...")
    electricity_path = 'data/raw/building-data-genome-project-2/data/meters/raw/electricity.csv'
    electricity = pd.read_csv(electricity_path, index_col=0)
    electricity.index = pd.to_datetime(electricity.index)
    
    weather_path = 'data/raw/building-data-genome-project-2/data/weather/weather.csv'
    metadata_path = 'data/raw/building-data-genome-project-2/data/metadata/metadata.csv'
    
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        metadata = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather = None
        metadata = None
    
    # ========== EVALUATE BASELINE MODEL ==========
    print("\n" + "="*70)
    print("  EVALUATING BASELINE MODEL")
    print("="*70)
    
    # Find baseline model
    baseline_models = glob.glob('models/baseline_*.ckpt')
    if not baseline_models:
        print("ERROR: No baseline model found!")
        return
    
    baseline_model_path = max(baseline_models, key=os.path.getmtime)
    print(f"Loading baseline model: {os.path.basename(baseline_model_path)}")
    
    # Prepare source building data
    if weather is not None and metadata is not None:
        source_site_id = metadata[metadata['building_id'] == source_building]['site_id'].values[0]
        source_weather = weather[weather['site_id'] == source_site_id].drop(columns=['site_id'])
        source_weather = source_weather.reindex(electricity.index)
    else:
        source_weather = None
    
    source_data, _ = preprocess_building_data(electricity, source_building, source_weather)
    
    # Get sequence length from baseline model
    baseline_model = EnergyLSTM.load_from_checkpoint(baseline_model_path)
    
    # Use 336 as default sequence length (2 weeks)
    seq_length = 336
    
    _, _, source_test_loader = create_dataloaders(source_data, seq_length=seq_length, batch_size=32)
    
    # Evaluate baseline
    baseline_results = evaluate_model(
        baseline_model, 
        source_test_loader,
        f"Baseline Model - {source_building}"
    )
    print_evaluation_results(baseline_results)
    
    # ========== EVALUATE TRANSFER MODEL ==========
    print("\n" + "="*70)
    print("  EVALUATING TRANSFER LEARNING MODEL")
    print("="*70)
    
    # Find transfer model
    transfer_models = glob.glob('models/transfer_*.ckpt')
    if not transfer_models:
        print("WARNING: No transfer model found! Skipping transfer evaluation.")
        print("\nRun train_transfer.py first to create a transfer learning model.")
        return
    
    transfer_model_path = max(transfer_models, key=os.path.getmtime)
    print(f"Loading transfer model: {os.path.basename(transfer_model_path)}")
    
    # Prepare target building data
    if weather is not None and metadata is not None:
        target_site_id = metadata[metadata['building_id'] == target_building]['site_id'].values[0]
        target_weather = weather[weather['site_id'] == target_site_id].drop(columns=['site_id'])
        target_weather = target_weather.reindex(electricity.index)
    else:
        target_weather = None
    
    target_data, _ = preprocess_building_data(electricity, target_building, target_weather)
    
    # Load transfer model
    transfer_model = EnergyLSTM.load_from_checkpoint(transfer_model_path)
    expected_input_size = transfer_model.hparams.input_size
    
    # Adjust features if needed
    actual_input_size = target_data.shape[1] - 1
    if actual_input_size < expected_input_size:
        missing_count = expected_input_size - actual_input_size
        print(f"Adding {missing_count} zero features to match model input size")
        for i in range(missing_count):
            target_data[f'missing_feature_{i}'] = 0.0
    elif actual_input_size > expected_input_size:
        feature_cols = [col for col in target_data.columns if col != 'energy']
        features_to_keep = feature_cols[:expected_input_size]
        target_data = target_data[['energy'] + features_to_keep]
    
    # Use 168 as default sequence length for transfer (1 week)
    transfer_seq_length = 168
    
    _, _, target_test_loader = create_dataloaders(target_data, seq_length=transfer_seq_length, batch_size=32)
    
    # Evaluate transfer model
    transfer_results = evaluate_model(
        transfer_model,
        target_test_loader,
        f"Transfer Model - {target_building}"
    )
    print_evaluation_results(transfer_results)
    
    # ========== COMPARE MODELS ==========
    # Note: Direct comparison is tricky since they're on different buildings
    # But we can show relative performance vs their respective data
    print("\n" + "="*70)
    print("  RELATIVE PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"\nBaseline Model (on {source_building}):")
    print(f"  Predicting with ±{baseline_results['mae']:.2f} kWh error")
    if not np.isnan(baseline_results['mape']):
        print(f"  This is {baseline_results['mape']:.1f}% of mean consumption ({baseline_results['mean_actual']:.2f} kWh)")
    else:
        print(f"  Mean consumption: {baseline_results['mean_actual']:.2f} kWh")
    
    print(f"\nTransfer Model (on {target_building}):")
    print(f"  Predicting with ±{transfer_results['mae']:.2f} kWh error")
    if not np.isnan(transfer_results['mape']):
        print(f"  This is {transfer_results['mape']:.1f}% of mean consumption ({transfer_results['mean_actual']:.2f} kWh)")
    else:
        print(f"  Mean consumption: {transfer_results['mean_actual']:.2f} kWh")
        print(f"  (MAPE not shown due to low consumption values)")
    
    print("\n" + "="*70)
    print("  KEY INSIGHTS")
    print("="*70)
    print(f"• Baseline R² = {baseline_results['r2']:.3f} (explains {baseline_results['r2']*100:.1f}% of variance)")
    print(f"• Transfer R² = {transfer_results['r2']:.3f} (explains {transfer_results['r2']*100:.1f}% of variance)")
    
    if not np.isnan(baseline_results['mape']) and not np.isnan(transfer_results['mape']):
        if transfer_results['mape'] < baseline_results['mape']:
            print(f"• Transfer learning achieved {baseline_results['mape'] - transfer_results['mape']:.1f}% better MAPE!")
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Baseline model tested on {len(baseline_results['predictions'])} samples")
    print(f"  Transfer model tested on {len(transfer_results['predictions'])} samples")
    

if __name__ == '__main__':
    main()
