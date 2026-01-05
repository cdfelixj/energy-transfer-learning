"""Diagnostic script to check baseline model predictions"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM

def diagnose_baseline():
    """Check what the baseline model is actually predicting"""
    
    print("="*90)
    print("  BASELINE MODEL DIAGNOSTIC")
    print("="*90)
    
    # Find baseline model
    models_dir = 'models'
    baseline_models = [f for f in os.listdir(models_dir) if f.startswith('baseline_')]
    
    if not baseline_models:
        print("\n✗ ERROR: No baseline model found!")
        return
    
    baseline_model_path = os.path.join(models_dir, max(baseline_models, key=lambda x: os.path.getmtime(os.path.join(models_dir, x))))
    print(f"\n✓ Loading model: {os.path.basename(baseline_model_path)}")
    
    # Load model
    model = EnergyLSTM.load_from_checkpoint(baseline_model_path)
    model.eval()
    print(f"  Model architecture: {model.hparams.hidden_size} hidden, {model.hparams.num_layers} layers")
    print(f"  Input size: {model.hparams.input_size}")
    
    # Load data (same as training)
    print("\nLoading data...")
    electricity, metadata, valid_buildings = load_electricity_data()
    
    # Load weather
    weather_path = os.path.join('data', 'raw', 'building-data-genome-project-2',
                                'data', 'weather', 'weather.csv')
    weather = pd.read_csv(weather_path)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    weather = weather.set_index('timestamp')
    
    # Process Colin building
    building_id = 'Rat_education_Colin'
    site_id = metadata[metadata['building_id'] == building_id]['site_id'].values[0]
    weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
    weather_building = weather_building.reindex(electricity.index)
    
    data, scaler = preprocess_building_data(electricity, building_id, weather_building)
    print(f"✓ Processed {building_id}: {data.shape[0]} hours")
    
    # Create same data splits as training (2 years, seq_length=336)
    train_loader, val_loader, test_loader = create_dataloaders(
        data, seq_length=336, batch_size=32
    )
    
    # Get predictions on each split
    for split_name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        print(f"\n{'='*60}")
        print(f"  {split_name.upper()} SET ANALYSIS")
        print(f"{'='*60}")
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                y_hat = model(x)
                predictions.extend(y_hat.squeeze().cpu().numpy())
                actuals.extend(y.squeeze().cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        print(f"\nPredictions - Mean: {np.mean(predictions):.2f}, Std: {np.std(predictions):.2f}")
        print(f"              Range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
        print(f"\nActuals     - Mean: {np.mean(actuals):.2f}, Std: {np.std(actuals):.2f}")
        print(f"              Range: [{np.min(actuals):.2f}, {np.max(actuals):.2f}]")
        
        print(f"\nMetrics:")
        print(f"  MAE:  {mae:.2f} kWh")
        print(f"  RMSE: {rmse:.2f} kWh")
        print(f"  R²:   {r2:.4f}")
        
        # Warnings
        if np.std(predictions) < 1.0:
            print(f"\n⚠ WARNING: Model is predicting near-constant values (Std={np.std(predictions):.2f})!")
        
        if r2 < 0:
            print(f"\n⚠ WARNING: Negative R² = {r2:.4f}")
            print(f"  This means the model is worse than just predicting the mean ({np.mean(actuals):.2f})")
        
        if r2 < 0.5:
            print(f"\n⚠ WARNING: Low R² = {r2:.4f} indicates poor fit")
    
    print(f"\n{'='*90}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'='*90}")

if __name__ == "__main__":
    diagnose_baseline()
