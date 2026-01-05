"""
Comprehensive 3-Model Evaluation Script

Compares three models on the same target building with limited data:
1. Baseline Model - Trained on 2 years of data from source building
2. Pre-Transfer Model - Trained from scratch on limited target data (1 month)
3. Transfer Model - Fine-tuned from baseline on limited target data (1 month)

This demonstrates the value of transfer learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
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
    
    # Calculate MAPE (avoiding division by zero)
    mask = np.abs(actuals) > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = np.nan
    
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
    print(f"R² Score:                          {results['r2']:.4f}")
    if not np.isnan(results['mape']):
        print(f"Mean Absolute Percentage Error:    {results['mape']:.2f}%")
    print(f"Median Absolute Error:             {results['median_ae']:.4f} kWh")
    print(f"Maximum Error:                     {results['max_error']:.4f} kWh")
    print(f"{'='*70}")


def compare_all_models(baseline_results, pretransfer_results, transfer_results):
    """Compare all three models"""
    print(f"\n{'='*90}")
    print(f"  THREE-MODEL COMPARISON: Transfer Learning Effectiveness")
    print(f"{'='*90}")
    print(f"{'Metric':<25} {'Baseline':<15} {'Pre-Transfer':<15} {'Transfer':<15} {'Improvement'}")
    print(f"{'-'*90}")
    
    metrics = [
        ('MAE (kWh)', 'mae'),
        ('RMSE (kWh)', 'rmse'),
        ('R² Score', 'r2'),
        ('MAPE (%)', 'mape'),
        ('Median AE (kWh)', 'median_ae')
    ]
    
    improvements = {}
    
    for metric_name, metric_key in metrics:
        baseline_val = baseline_results[metric_key]
        pretransfer_val = pretransfer_results[metric_key]
        transfer_val = transfer_results[metric_key]
        
        # Calculate improvement: Transfer vs Pre-Transfer
        if metric_key == 'r2':
            # Higher is better for R²
            improvement = ((transfer_val - pretransfer_val) / abs(pretransfer_val)) * 100
            better = "✓" if transfer_val > pretransfer_val else "✗"
        else:
            # Lower is better for error metrics
            improvement = ((pretransfer_val - transfer_val) / pretransfer_val) * 100
            better = "✓" if transfer_val < pretransfer_val else "✗"
        
        improvements[metric_key] = improvement
        
        # Handle NaN values
        baseline_str = f"{baseline_val:.4f}" if not np.isnan(baseline_val) else "N/A"
        pretransfer_str = f"{pretransfer_val:.4f}" if not np.isnan(pretransfer_val) else "N/A"
        transfer_str = f"{transfer_val:.4f}" if not np.isnan(transfer_val) else "N/A"
        
        print(f"{metric_name:<25} {baseline_str:>13} {pretransfer_str:>13} {transfer_str:>13}   "
              f"{improvement:>6.1f}% {better}")
    
    print(f"{'='*90}")
    
    # Summary
    print(f"\nKEY FINDINGS:")
    print(f"1. Baseline Model: Trained on 2 years of source building data")
    print(f"2. Pre-Transfer Model: Trained from scratch on 1 month of target building data")
    print(f"3. Transfer Model: Fine-tuned baseline on 1 month of target building data")
    print(f"\nTransfer Learning Improvement:")
    
    if improvements['rmse'] > 0:
        print(f"  ✓ RMSE reduced by {improvements['rmse']:.1f}% compared to pre-transfer")
    else:
        print(f"  ✗ RMSE increased by {abs(improvements['rmse']):.1f}% compared to pre-transfer")
    
    if improvements['mae'] > 0:
        print(f"  ✓ MAE reduced by {improvements['mae']:.1f}% compared to pre-transfer")
    else:
        print(f"  ✗ MAE increased by {abs(improvements['mae']):.1f}% compared to pre-transfer")
    
    return improvements


def prepare_test_data(target_building, data_limit_months=1, seq_length=24, architecture_match=None):
    """Prepare test data for target building (same as used in training)"""
    
    # Load filtered data
    electricity, metadata, valid_buildings = load_electricity_data()
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    # Load weather
    weather_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2',
                                'data', 'weather', 'weather.csv')
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        
        site_id = metadata[metadata['building_id'] == target_building]['site_id'].values[0]
        weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
        weather_building = weather_building.reindex(electricity.index)
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_building = None
    
    # Preprocess target building
    target_data, target_scaler = preprocess_building_data(
        electricity, target_building, weather_building
    )
    
    # Limit data
    hours_to_keep = data_limit_months * 30 * 24
    target_data = target_data.iloc[:hours_to_keep]
    
    # Match architecture if needed
    if architecture_match:
        baseline_model = EnergyLSTM.load_from_checkpoint(architecture_match)
        expected_input_size = baseline_model.hparams.input_size
        actual_input_size = target_data.shape[1] - 1
        
        if actual_input_size != expected_input_size:
            feature_cols = [col for col in target_data.columns if col != 'energy']
            
            if actual_input_size < expected_input_size:
                missing_count = expected_input_size - actual_input_size
                for i in range(missing_count):
                    target_data[f'missing_feature_{i}'] = 0.0
            else:
                features_to_keep = feature_cols[:expected_input_size]
                target_data = target_data[['energy'] + features_to_keep]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )
    
    return train_loader, val_loader, test_loader


def main():
    print("="*90)
    print("  COMPREHENSIVE 3-MODEL EVALUATION")
    print("="*90)
    
    # Configuration
    # Use same target building as training (Rat education building NOT in baseline)
    target_building = 'Rat_education_Denise'
    data_limit_months = 1
    seq_length = 24  # Match training (24 hours = 1 day)
    
    print(f"\nTarget Building: {target_building}")
    print(f"Limited Data: {data_limit_months} month(s)")
    print(f"Sequence Length: {seq_length} hours")
    
    # Find model checkpoints
    print("\nSearching for trained models...")
    
    baseline_models = glob.glob('models/baseline_*.ckpt')
    pretransfer_models = glob.glob('models/pretransfer_*.ckpt')
    transfer_models = glob.glob('models/transfer_*.ckpt')
    
    if not baseline_models:
        print("\n✗ ERROR: No baseline model found!")
        print("  Please run: python src/train_baseline.py")
        return
    
    if not pretransfer_models:
        print("\n✗ ERROR: No pre-transfer model found!")
        print("  Please run: python src/train_pretransfer.py")
        return
    
    if not transfer_models:
        print("\n✗ ERROR: No transfer model found!")
        print("  Please run: python src/train_transfer.py")
        return
    
    # Load most recent models
    baseline_model_path = max(baseline_models, key=os.path.getmtime)
    pretransfer_model_path = max(pretransfer_models, key=os.path.getmtime)
    transfer_model_path = max(transfer_models, key=os.path.getmtime)
    
    print(f"\n✓ Found all three models:")
    print(f"  1. Baseline:     {os.path.basename(baseline_model_path)}")
    print(f"  2. Pre-Transfer: {os.path.basename(pretransfer_model_path)}")
    print(f"  3. Transfer:     {os.path.basename(transfer_model_path)}")
    
    # Load models
    print("\nLoading models...")
    baseline_model = EnergyLSTM.load_from_checkpoint(baseline_model_path)
    pretransfer_model = EnergyLSTM.load_from_checkpoint(pretransfer_model_path)
    transfer_model = EnergyLSTM.load_from_checkpoint(transfer_model_path)
    print("✓ All models loaded")
    
    # Prepare test data
    print("\nPreparing test data...")
    train_loader, val_loader, test_loader = prepare_test_data(
        target_building, data_limit_months, seq_length, baseline_model_path
    )
    print(f"✓ Test data ready: {len(test_loader.dataset)} samples")
    
    # Evaluate all models
    print("\n" + "="*90)
    print("  EVALUATING MODELS ON TARGET BUILDING TEST SET")
    print("="*90)
    
    baseline_results = evaluate_model(baseline_model, test_loader, 
                                     "1. BASELINE MODEL (1 year source data)")
    print_evaluation_results(baseline_results)
    
    pretransfer_results = evaluate_model(pretransfer_model, test_loader,
                                        "2. PRE-TRANSFER MODEL (1 month target, no transfer)")
    print_evaluation_results(pretransfer_results)
    
    transfer_results = evaluate_model(transfer_model, test_loader,
                                     "3. TRANSFER MODEL (1 month target + transfer)")
    print_evaluation_results(transfer_results)
    
    # Compare all models
    improvements = compare_all_models(baseline_results, pretransfer_results, transfer_results)
    
    # Save results
    print("\nSaving results...")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_df = pd.DataFrame([
        {
            'model': 'Baseline',
            'description': '2 years source building data',
            'mae': baseline_results['mae'],
            'rmse': baseline_results['rmse'],
            'r2': baseline_results['r2'],
            'mape': baseline_results['mape']
        },
        {
            'model': 'Pre-Transfer',
            'description': '1 month target data (no transfer)',
            'mae': pretransfer_results['mae'],
            'rmse': pretransfer_results['rmse'],
            'r2': pretransfer_results['r2'],
            'mape': pretransfer_results['mape']
        },
        {
            'model': 'Transfer',
            'description': '1 month target data + transfer learning',
            'mae': transfer_results['mae'],
            'rmse': transfer_results['rmse'],
            'r2': transfer_results['r2'],
            'mape': transfer_results['mape']
        }
    ])
    
    comparison_path = os.path.join(results_dir, 'three_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison to: {comparison_path}")
    
    # Create visualization
    try:
        create_comparison_plot(comparison_df, results_dir)
        print(f"✓ Saved visualization to: results/model_comparison.png")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    print("\n" + "="*90)
    print("  EVALUATION COMPLETE")
    print("="*90)


def create_comparison_plot(df, results_dir):
    """Create bar plot comparing model performance"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Transfer Learning Model Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('mae', 'Mean Absolute Error (MAE)', 'kWh'),
        ('rmse', 'Root Mean Squared Error (RMSE)', 'kWh'),
        ('r2', 'R² Score', ''),
        ('mape', 'Mean Absolute Percentage Error (MAPE)', '%')
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (metric, title, unit) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = df[metric].values
        bars = ax.bar(df['model'], values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(f'{title} {f"({unit})" if unit else ""}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
