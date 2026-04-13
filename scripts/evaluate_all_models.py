"""
Comprehensive Transfer Learning Evaluation Script

EVALUATION SETUP:
==================

MODELS EVALUATED:
1. Baseline Model (2 evaluations):
   - Baseline-Source: Evaluated on SOURCE building (Rat_education_Colin) where it was trained
     * Purpose: Shows best-case performance with abundant training data
   - Baseline-Target: Evaluated on TARGET building (Rat_education_Denise) - NEW building
     * Purpose: Shows cross-building generalization (domain shift)

2. Pre-Transfer Model:
   - Trained from scratch on 8 weeks of TARGET building data
   - Evaluated on TARGET building
   - Purpose: Control group - performance WITHOUT transfer learning

3. Transfer Model:
   - Fine-tuned from baseline on 8 weeks of TARGET building data
   - Evaluated on TARGET building
   - Purpose: Experimental group - performance WITH transfer learning

KEY COMPARISONS:
================

1. Baseline-Source vs Baseline-Target:
   → Measures domain shift penalty (how much performance drops on new building)

2. Pre-Transfer vs Transfer (MAIN COMPARISON):
   → Measures transfer learning effectiveness
   → Both use SAME limited data (8 weeks)
   → Both evaluated on SAME building (target)
   → Difference shows pure benefit of transfer learning

3. Baseline-Target vs Pre-Transfer:
   → Compares: lots of data on different building vs little data on same building

4. Baseline-Target vs Transfer:
   → Shows if fine-tuning baseline on limited target data beats using baseline as-is

EXPECTED RESULTS:
=================
Baseline-Source: Best performance (trained and tested on same building)
Baseline-Target: Moderate (domain shift from Colin to Denise)
Pre-Transfer: Variable (limited data, no transfer)
Transfer: Should beat Pre-Transfer (transfer learning benefit)

Ideal outcome: Transfer > Pre-Transfer (proves transfer learning helps!)
"""

import sys
import os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import pandas as pd
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM, EnergyLSTMFrozen, EnergyLSTMAdapter


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
    
    # DIAGNOSTIC: Print statistics to identify scale issues
    print(f"\n  [DIAGNOSTIC] {model_name}:")
    print(f"    Predictions - Mean: {np.mean(predictions):.2f}, Std: {np.std(predictions):.2f}, Range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
    print(f"    Actuals     - Mean: {np.mean(actuals):.2f}, Std: {np.std(actuals):.2f}, Range: [{np.min(actuals):.2f}, {np.max(actuals):.2f}]")
    
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


def compare_all_models(baseline_source_results, baseline_target_results, pretransfer_results, transfer_results):
    """Compare all models with baseline evaluated on both source and target buildings"""
    print(f"\n{'='*110}")
    print(f"  COMPREHENSIVE MODEL COMPARISON: Transfer Learning Effectiveness")
    print(f"{'='*110}")
    print(f"\nNOTE: All models evaluated on TARGET building (Rat_education_Denise) test set,")
    print(f"      except 'Baseline-Source' which is evaluated on SOURCE building (Rat_education_Colin).")
    print(f"\n{'Metric':<20} {'Baseline-Source':<17} {'Baseline-Target':<17} {'Pre-Transfer':<15} {'Transfer':<15} {'TL Gain'}")
    print(f"{'-'*110}")
    
    metrics = [
        ('MAE (kWh)', 'mae'),
        ('RMSE (kWh)', 'rmse'),
        ('R² Score', 'r2'),
        ('MAPE (%)', 'mape'),
        ('Median AE (kWh)', 'median_ae')
    ]
    
    improvements = {}
    
    for metric_name, metric_key in metrics:
        baseline_source_val = baseline_source_results[metric_key]
        baseline_target_val = baseline_target_results[metric_key]
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
        baseline_source_str = f"{baseline_source_val:.4f}" if not np.isnan(baseline_source_val) else "N/A"
        baseline_target_str = f"{baseline_target_val:.4f}" if not np.isnan(baseline_target_val) else "N/A"
        pretransfer_str = f"{pretransfer_val:.4f}" if not np.isnan(pretransfer_val) else "N/A"
        transfer_str = f"{transfer_val:.4f}" if not np.isnan(transfer_val) else "N/A"
        
        print(f"{metric_name:<20} {baseline_source_str:>15} {baseline_target_str:>15} {pretransfer_str:>13} {transfer_str:>13}   "
              f"{improvement:>6.1f}% {better}")
    
    print(f"{'='*110}")
    
    # Summary
    print(f"\n" + "="*110)
    print(f"  KEY FINDINGS & INTERPRETATION")
    print(f"="*110)
    print(f"\n1. Baseline-Source: Baseline model on its training building (best-case performance)")
    print(f"2. Baseline-Target: Baseline model on NEW building (cross-building generalization)")
    print(f"3. Pre-Transfer: Train from scratch on 2 months of target building data")
    print(f"4. Transfer: Fine-tune baseline on 2 months of target building data")
    print(f"\nTRANSFER LEARNING EFFECTIVENESS (Transfer vs Pre-Transfer):")
    
    if improvements['rmse'] > 0:
        print(f"  ✓ RMSE reduced by {improvements['rmse']:.1f}% compared to pre-transfer")
    else:
        print(f"  ✗ RMSE increased by {abs(improvements['rmse']):.1f}% compared to pre-transfer")
    
    if improvements['mae'] > 0:
        print(f"  ✓ MAE reduced by {improvements['mae']:.1f}% compared to pre-transfer")
    else:
        print(f"  ✗ MAE increased by {abs(improvements['mae']):.1f}% compared to pre-transfer")
    
    return improvements


def prepare_test_data(target_building, data_limit_months=1, seq_length=24,
                      architecture_match=None, site_id='Rat', building_type='Education'):
    """Prepare test data for target building (same as used in training)"""
    
    # Load filtered data
    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
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
    
    # Limit data (use 4-week months to align with data_limit_weeks used in training scripts)
    weeks_per_month = 4
    hours_per_week = 7 * 24
    hours_to_keep = int(data_limit_months * weeks_per_month * hours_per_week)
    target_data = target_data.iloc[:hours_to_keep]
    
    # Match architecture if needed
    if architecture_match:
        baseline_model = EnergyLSTM.load_from_checkpoint(architecture_match, strict=False)
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


def evaluate_experiment(experiment_name, source_building, target_building,
                        site_id, building_type, baseline_model_path,
                        weeks_list=None, seq_length=24, data_limit_months=2):
    """
    Run a full 4-model evaluation + data efficiency analysis for one experiment.

    Saves results under results/experiments/{experiment_name}/:
      - baseline_comparison.csv      (4-model snapshot at data_limit_months of data)
      - data_efficiency_pretransfer.csv
      - data_efficiency_transfer.csv
      - analysis_summary.csv

    Args:
        experiment_name: e.g. 'eagle_education'
        source_building: Building the baseline was trained on
        target_building: Building pre-transfer and transfer were trained on
        site_id: Passed to load_electricity_data (None = any site)
        building_type: Passed to load_electricity_data (None = any type)
        baseline_model_path: Absolute path to the baseline .ckpt
        weeks_list: Weeks for data efficiency sweep (default [1,2,4,8,16,32,64,104])
        seq_length: Sequence length (default 24 h)
        data_limit_months: Months of target data for the 4-model comparison (default 2)
    """
    if weeks_list is None:
        weeks_list = [1, 2, 4, 8, 16, 32, 64, 104]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    exp_dir = os.path.join(project_root, 'models', 'experiments', experiment_name)
    results_dir = os.path.join(project_root, 'results', 'experiments', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'=' * 90}")
    print(f"  EVALUATING EXPERIMENT: {experiment_name}")
    print(f"  Source: {source_building}  |  Target: {target_building}")
    print(f"{'=' * 90}")

    # ------------------------------------------------------------------ #
    # Locate model checkpoints
    # ------------------------------------------------------------------ #
    def latest(pattern):
        files = glob.glob(pattern)
        return max(files, key=os.path.getmtime) if files else None

    baseline_path = baseline_model_path or latest(
        os.path.join(exp_dir, 'baseline_*.ckpt')
    )
    pretransfer_path = latest(os.path.join(exp_dir, 'pretransfer_*.ckpt'))
    transfer_path = latest(os.path.join(exp_dir, 'transfer_*.ckpt'))

    missing = []
    if not baseline_path:
        missing.append('baseline')
    if not pretransfer_path:
        missing.append('pretransfer')
    if not transfer_path:
        missing.append('transfer')

    if missing:
        print(f"  ⚠ Missing checkpoints: {missing} — skipping 4-model comparison.")
        models_ok = False
    else:
        print(f"  ✓ Baseline:     {os.path.basename(baseline_path)}")
        print(f"  ✓ Pre-Transfer: {os.path.basename(pretransfer_path)}")
        print(f"  ✓ Transfer:     {os.path.basename(transfer_path)}")
        models_ok = True

    # ------------------------------------------------------------------ #
    # 4-model comparison (Baseline-Source, Baseline-Target, Pre-Transfer, Transfer)
    # ------------------------------------------------------------------ #
    if models_ok:
        baseline_model = EnergyLSTM.load_from_checkpoint(baseline_path)
        pretransfer_model = EnergyLSTM.load_from_checkpoint(pretransfer_path)
        transfer_model = EnergyLSTM.load_from_checkpoint(transfer_path)

        # Source data (full)
        src_loader = prepare_test_data(
            source_building, data_limit_months=24, seq_length=336,
            architecture_match=baseline_path,
            site_id=site_id, building_type=building_type,
        )[2]

        # Target data (limited)
        tgt_loader = prepare_test_data(
            target_building, data_limit_months=data_limit_months, seq_length=seq_length,
            architecture_match=baseline_path,
            site_id=site_id, building_type=building_type,
        )[2]

        r_bs = evaluate_model(baseline_model, src_loader, f'Baseline-Source ({source_building})')
        r_bt = evaluate_model(baseline_model, tgt_loader, f'Baseline-Target ({target_building})')
        r_pt = evaluate_model(pretransfer_model, tgt_loader, f'Pre-Transfer ({target_building})')
        r_tr = evaluate_model(transfer_model, tgt_loader, f'Transfer ({target_building})')

        compare_all_models(r_bs, r_bt, r_pt, r_tr)

        comparison_df = pd.DataFrame([
            {'model': 'Baseline-Source', 'building': source_building,
             'mae': r_bs['mae'], 'rmse': r_bs['rmse'], 'r2': r_bs['r2'],
             'mape': r_bs['mape'], 'median_ae': r_bs['median_ae']},
            {'model': 'Baseline-Target', 'building': target_building,
             'mae': r_bt['mae'], 'rmse': r_bt['rmse'], 'r2': r_bt['r2'],
             'mape': r_bt['mape'], 'median_ae': r_bt['median_ae']},
            {'model': 'Pre-Transfer', 'building': target_building,
             'mae': r_pt['mae'], 'rmse': r_pt['rmse'], 'r2': r_pt['r2'],
             'mape': r_pt['mape'], 'median_ae': r_pt['median_ae']},
            {'model': 'Transfer', 'building': target_building,
             'mae': r_tr['mae'], 'rmse': r_tr['rmse'], 'r2': r_tr['r2'],
             'mape': r_tr['mape'], 'median_ae': r_tr['median_ae']},
        ])
        comparison_df.to_csv(
            os.path.join(results_dir, 'baseline_comparison.csv'), index=False
        )
        print(f"\n  ✓ Saved: results/experiments/{experiment_name}/baseline_comparison.csv")

        # Summary stats
        tl_mae_improv = (r_pt['mae'] - r_tr['mae']) / r_pt['mae'] * 100
        domain_shift = (r_bt['mae'] - r_bs['mae']) / r_bs['mae'] * 100
        summary_df = pd.DataFrame([{
            'experiment': experiment_name,
            'source_building': source_building,
            'target_building': target_building,
            'transfer_benefit_mae_pct': round(tl_mae_improv, 2),
            'domain_shift_penalty_pct': round(domain_shift, 2),
            'baseline_source_mae': round(r_bs['mae'], 4),
            'baseline_target_mae': round(r_bt['mae'], 4),
            'pretransfer_mae': round(r_pt['mae'], 4),
            'transfer_mae': round(r_tr['mae'], 4),
        }])
        summary_df.to_csv(
            os.path.join(results_dir, 'analysis_summary.csv'), index=False
        )
        print(f"  ✓ Saved: results/experiments/{experiment_name}/analysis_summary.csv")

    # ------------------------------------------------------------------ #
    # Data efficiency sweep  (4 strategies)
    # ------------------------------------------------------------------ #
    for model_type in ('pretransfer', 'transfer', 'frozen', 'adapter'):
        de_results = evaluate_data_efficiency(
            model_type=model_type,
            target_building=target_building,
            weeks_list=weeks_list,
            seq_length=seq_length,
            experiment_name=experiment_name,
            site_id=site_id,
            building_type=building_type,
        )
        compare_data_efficiency(de_results, model_type.capitalize())
        out_path = os.path.join(results_dir, f'data_efficiency_{model_type}.csv')
        de_results.to_csv(out_path, index=False)
        print(f"  ✓ Saved: results/experiments/{experiment_name}/data_efficiency_{model_type}.csv")

    print(f"\n  ✓ Experiment {experiment_name} evaluation complete.")


def main():
    print("="*90)
    print("  COMPREHENSIVE 3-MODEL EVALUATION")
    print("="*90)

    # Configuration
    # Use same target building as training (Rat education building NOT in baseline)
    target_building = 'Rat_education_Denise'
    data_limit_months = 2  # Changed to 2 months to match training
    seq_length = 24  # Match training (24 hours = 1 day)

    print(f"\nTarget Building: {target_building}")
    print(f"Limited Data: {data_limit_months} month(s)")
    print(f"Sequence Length: {seq_length} hours")
    
    # Find model checkpoints
    print("\nSearching for trained models...")
    
    exp_dir = os.path.join('models', 'experiments', 'rat_education')
    baseline_models = glob.glob(os.path.join(exp_dir, 'baseline_*.ckpt'))
    pretransfer_models = glob.glob(os.path.join(exp_dir, 'pretransfer_*.ckpt'))
    transfer_models = glob.glob(os.path.join(exp_dir, 'transfer_*.ckpt'))

    if not baseline_models:
        print("\n✗ ERROR: No baseline model found!")
        print(f"  Looked in: {exp_dir}")
        print("  Please run: python scripts/run_experiment_suite.py --experiment rat_education")
        return

    if not pretransfer_models:
        print("\n✗ ERROR: No pre-transfer model found!")
        print(f"  Looked in: {exp_dir}")
        return

    if not transfer_models:
        print("\n✗ ERROR: No transfer model found!")
        print(f"  Looked in: {exp_dir}")
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
    
    # Prepare test data for SOURCE building (where baseline was trained)
    print("\nPreparing test data for SOURCE building (Rat_education_Colin)...")
    source_building = 'Rat_education_Colin'
    # Use full 2 years for source building evaluation
    source_train_loader, source_val_loader, source_test_loader = prepare_test_data(
        source_building, data_limit_months=24, seq_length=336, architecture_match=baseline_model_path
    )
    print(f"✓ Source building test data ready: {len(source_test_loader.dataset)} samples")
    
    # Prepare test data for TARGET building
    print("\nPreparing test data for TARGET building (Rat_education_Denise)...")
    train_loader, val_loader, test_loader = prepare_test_data(
        target_building, data_limit_months, seq_length, baseline_model_path
    )
    print(f"✓ Target building test data ready: {len(test_loader.dataset)} samples")
    
    # Evaluate all models
    print("\n" + "="*90)
    print("  EVALUATING MODELS")
    print("="*90)
    
    print("\n[1/4] Evaluating Baseline on SOURCE building (where it was trained)...")
    baseline_source_results = evaluate_model(baseline_model, source_test_loader, 
                                            "BASELINE on SOURCE (Rat_education_Colin)")
    print_evaluation_results(baseline_source_results)
    
    print("\n[2/4] Evaluating Baseline on TARGET building (cross-building transfer)...")
    baseline_target_results = evaluate_model(baseline_model, test_loader, 
                                           "BASELINE on TARGET (Rat_education_Denise)")
    print_evaluation_results(baseline_target_results)
    
    print("\n[3/4] Evaluating Pre-Transfer on TARGET building...")
    pretransfer_results = evaluate_model(pretransfer_model, test_loader,
                                        "PRE-TRANSFER on TARGET (2 months, no transfer)")
    print_evaluation_results(pretransfer_results)
    
    print("\n[4/4] Evaluating Transfer on TARGET building...")
    transfer_results = evaluate_model(transfer_model, test_loader,
                                     "TRANSFER on TARGET (2 months + transfer)")
    print_evaluation_results(transfer_results)
    
    # Compare all models
    improvements = compare_all_models(baseline_source_results, baseline_target_results, 
                                     pretransfer_results, transfer_results)
    
    # Save results
    print("\nSaving results...")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_df = pd.DataFrame([
        {
            'model': 'Baseline-Source',
            'building': 'Rat_education_Colin (source)',
            'description': '2 years, evaluated on training building',
            'mae': baseline_source_results['mae'],
            'rmse': baseline_source_results['rmse'],
            'mse': baseline_source_results['mse'],
            'r2': baseline_source_results['r2'],
            'mape': baseline_source_results['mape'],
            'median_ae': baseline_source_results['median_ae'],
            'max_error': baseline_source_results['max_error']
        },
        {
            'model': 'Baseline-Target',
            'building': 'Rat_education_Denise (target)',
            'description': '2 years source, evaluated on NEW building',
            'mae': baseline_target_results['mae'],
            'rmse': baseline_target_results['rmse'],
            'mse': baseline_target_results['mse'],
            'r2': baseline_target_results['r2'],
            'mape': baseline_target_results['mape'],
            'median_ae': baseline_target_results['median_ae'],
            'max_error': baseline_target_results['max_error']
        },
        {
            'model': 'Pre-Transfer',
            'building': 'Rat_education_Denise (target)',
            'description': '2 months target data (no transfer)',
            'mae': pretransfer_results['mae'],
            'rmse': pretransfer_results['rmse'],
            'mse': pretransfer_results['mse'],
            'r2': pretransfer_results['r2'],
            'mape': pretransfer_results['mape'],
            'median_ae': pretransfer_results['median_ae'],
            'max_error': pretransfer_results['max_error']
        },
        {
            'model': 'Transfer',
            'building': 'Rat_education_Denise (target)',
            'description': '2 months target data + transfer learning',
            'mae': transfer_results['mae'],
            'rmse': transfer_results['rmse'],
            'mse': transfer_results['mse'],
            'r2': transfer_results['r2'],
            'mape': transfer_results['mape'],
            'median_ae': transfer_results['median_ae'],
            'max_error': transfer_results['max_error']
        }
    ])
    
    comparison_path = os.path.join(results_dir, 'three_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison to: {comparison_path}")
    
    # Save prediction arrays for detailed visualization
    print("\nSaving prediction arrays for visualization...")
    predictions_path = os.path.join(results_dir, 'model_predictions.npz')
    np.savez(predictions_path,
             baseline_source_preds=baseline_source_results['predictions'],
             baseline_source_actuals=baseline_source_results['actuals'],
             baseline_target_preds=baseline_target_results['predictions'],
             baseline_target_actuals=baseline_target_results['actuals'],
             pretransfer_preds=pretransfer_results['predictions'],
             pretransfer_actuals=pretransfer_results['actuals'],
             transfer_preds=transfer_results['predictions'],
             transfer_actuals=transfer_results['actuals'])
    print(f"✓ Saved prediction arrays to: {predictions_path}")
    
    # Save per-sample error analysis
    print("\nCreating per-sample error analysis...")
    error_analysis = pd.DataFrame({
        'sample_id': range(len(baseline_target_results['actuals'])),
        'actual': baseline_target_results['actuals'],
        'baseline_target_pred': baseline_target_results['predictions'],
        'pretransfer_pred': pretransfer_results['predictions'],
        'transfer_pred': transfer_results['predictions'],
        'baseline_target_error': np.abs(baseline_target_results['actuals'] - baseline_target_results['predictions']),
        'pretransfer_error': np.abs(pretransfer_results['actuals'] - pretransfer_results['predictions']),
        'transfer_error': np.abs(transfer_results['actuals'] - transfer_results['predictions'])
    })
    error_analysis_path = os.path.join(results_dir, 'per_sample_errors.csv')
    error_analysis.to_csv(error_analysis_path, index=False)
    print(f"✓ Saved per-sample errors to: {error_analysis_path}")
    
    # Create visualization
    try:
        create_comparison_plot(comparison_df, results_dir)
        print(f"✓ Saved visualization to: results/model_comparison.png")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    print("\n" + "="*90)
    print("  EVALUATION COMPLETE")
    print("="*90)
    
    # ========================================================================
    # DATA EFFICIENCY EVALUATION
    # ========================================================================
    
    print("\n\n")
    print("#"*90)
    print("#" + " "*88 + "#")
    print("#" + "  DATA EFFICIENCY EVALUATION: Impact of Training Data Amount".center(88) + "#")
    print("#" + " "*88 + "#")
    print("#"*90)
    
    # Evaluate Pre-Transfer models with different data amounts
    pretransfer_data_eff_results = evaluate_data_efficiency(
        model_type='pretransfer',
        target_building=target_building,
        weeks_list=[1, 2, 4, 8, 16, 32, 64, 104],
        seq_length=seq_length,
        experiment_name='rat_education',
    )
    
    # Display Pre-Transfer comparison table
    compare_data_efficiency(pretransfer_data_eff_results, 'Pre-Transfer')
    
    # Save Pre-Transfer data efficiency results
    pretransfer_eff_path = os.path.join(results_dir, 'pretransfer_data_efficiency.csv')
    pretransfer_data_eff_results.to_csv(pretransfer_eff_path, index=False)
    print(f"\n✓ Saved Pre-Transfer data efficiency results to: {pretransfer_eff_path}")
    
    # Evaluate Transfer models with different data amounts
    transfer_data_eff_results = evaluate_data_efficiency(
        model_type='transfer',
        target_building=target_building,
        weeks_list=[1, 2, 4, 8, 16, 32, 64, 104],
        seq_length=seq_length,
        experiment_name='rat_education',
    )
    
    # Display Transfer comparison table
    compare_data_efficiency(transfer_data_eff_results, 'Transfer')
    
    # Save Transfer data efficiency results
    transfer_eff_path = os.path.join(results_dir, 'transfer_data_efficiency.csv')
    transfer_data_eff_results.to_csv(transfer_eff_path, index=False)
    print(f"\n✓ Saved Transfer data efficiency results to: {transfer_eff_path}")
    
    # Final summary
    print("\n" + "#"*90)
    print("#" + " "*88 + "#")
    print("#" + "  ALL EVALUATIONS COMPLETE".center(88) + "#")
    print("#" + " "*88 + "#")
    print("#"*90)
    print("\nResults saved:")
    print(f"  • Main comparison: {comparison_path}")
    print(f"  • Pre-Transfer data efficiency: {pretransfer_eff_path}")
    print(f"  • Transfer data efficiency: {transfer_eff_path}")
    print("#"*90)


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


# Map model_type string → Lightning class used to save the checkpoint
_MODEL_CLASS = {
    'pretransfer':   EnergyLSTM,
    'transfer':      EnergyLSTM,
    'frozen':        EnergyLSTMFrozen,
    'adapter':       EnergyLSTMAdapter,
    'multitransfer':    EnergyLSTM,   # Multi-source pre-trained, same architecture as baseline
    'ensembletransfer': EnergyLSTM,   # Weight-averaged model soup, same architecture
    # N-source ablation variants (multitransfer with N=1..5 source buildings)
    'multitransfer_n1': EnergyLSTM,
    'multitransfer_n2': EnergyLSTM,
    'multitransfer_n3': EnergyLSTM,
    'multitransfer_n4': EnergyLSTM,
    'multitransfer_n5':  EnergyLSTM,
    'multitransfer_n10': EnergyLSTM,
    'multitransfer_n15': EnergyLSTM,
    # Cross-type transfer variants (same target, different source domain distance)
    'transfer_samesite':  EnergyLSTM,
    'transfer_sametype':  EnergyLSTM,
    'transfer_crosstype': EnergyLSTM,
}


def evaluate_data_efficiency(model_type, target_building, weeks_list=[1, 2, 4, 8, 16, 32, 64, 104],
                             seq_length=24, experiment_name='rat_education',
                             site_id='Rat', building_type='Education'):
    """
    Evaluate pretransfer / transfer / frozen / adapter models with different data amounts.

    Args:
        model_type: 'pretransfer', 'transfer', 'frozen', or 'adapter'
        target_building: Building ID to evaluate on
        weeks_list: List of week amounts to evaluate (104 weeks = 2 years)
        seq_length: Sequence length used in training
        experiment_name: Experiment directory name under models/experiments/
        site_id: Site filter forwarded to prepare_test_data
        building_type: Building type filter forwarded to prepare_test_data

    Returns:
        DataFrame with results for each data amount
    """
    print(f"\n{'='*90}")
    print(f"  DATA EFFICIENCY EVALUATION: {model_type.upper()} Models  [{experiment_name}]")
    print(f"{'='*90}")

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    results = []

    for weeks in weeks_list:
        print(f"\n[Evaluating {weeks} week(s) model...]")

        # Find model checkpoint (use absolute path so this works regardless of CWD)
        pattern = os.path.join(
            _project_root, 'models', 'experiments', experiment_name, 'data_efficiency',
            f'{model_type}_{target_building[:15]}_{weeks}week_*.ckpt'
        )
        model_files = glob.glob(pattern)
        
        if not model_files:
            print(f"  ⚠ WARNING: No {model_type} model found for {weeks} week(s)")
            print(f"  Searched: {pattern}")
            print(f"  Skipping... (will show N/A in results)")
            results.append({
                'weeks': weeks,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'median_ae': np.nan
            })
            continue
        
        # Load most recent model
        model_path = max(model_files, key=os.path.getmtime)
        print(f"  Found: {os.path.basename(model_path)}")
        
        try:
            # Load model using the correct class for this strategy
            model_cls = _MODEL_CLASS.get(model_type, EnergyLSTM)
            model = model_cls.load_from_checkpoint(model_path, strict=(model_type != 'adapter'))

            # Prepare test data with same weeks as training
            train_loader, val_loader, test_loader = prepare_test_data(
                target_building,
                data_limit_months=int(weeks / 4) if weeks >= 4 else 1,
                seq_length=seq_length,
                architecture_match=model_path,
                site_id=site_id,
                building_type=building_type,
            )
            
            # Evaluate
            eval_results = evaluate_model(
                model, test_loader, 
                f"{model_type.upper()} ({weeks} weeks)"
            )
            
            results.append({
                'weeks': weeks,
                'mae': eval_results['mae'],
                'rmse': eval_results['rmse'],
                'r2': eval_results['r2'],
                'mape': eval_results['mape'],
                'median_ae': eval_results['median_ae']
            })
            
            print(f"  ✓ MAE: {eval_results['mae']:.4f}, RMSE: {eval_results['rmse']:.4f}, R²: {eval_results['r2']:.4f}")
            
        except Exception as e:
            print(f"  ✗ ERROR: Failed to evaluate model: {e}")
            results.append({
                'weeks': weeks,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'median_ae': np.nan
            })
    
    return pd.DataFrame(results)


def compare_data_efficiency(results_df, model_type):
    """
    Print formatted comparison table for data efficiency results
    
    Args:
        results_df: DataFrame with columns [weeks, mae, rmse, r2, mape, median_ae]
        model_type: 'Pre-Transfer' or 'Transfer' for display
    """
    print(f"\n{'='*110}")
    print(f"  DATA EFFICIENCY ANALYSIS: {model_type} Models")
    print(f"{'='*110}")
    print(f"\nComparison of {model_type} model performance with varying amounts of training data:")
    print(f"(All models trained and evaluated on same building: Rat_education_Denise)")
    print(f"\n{'Metric':<25} {'1 Week':<15} {'2 Weeks':<15} {'4 Weeks':<15} {'8 Weeks':<15} {'16 Weeks':<15} {'32 Weeks':<15} {'64 Weeks':<15} {'2 Years':<15}")
    print(f"{'-'*155}")
    
    metrics = [
        ('mae', 'MAE (kWh)'),
        ('rmse', 'RMSE (kWh)'),
        ('r2', 'R² Score'),
        ('mape', 'MAPE (%)'),
        ('median_ae', 'Median AE (kWh)')
    ]
    
    for metric_key, metric_name in metrics:
        row = f"{metric_name:<25}"
        
        for weeks in [1, 2, 4, 8, 16, 32, 64, 104]:
            week_data = results_df[results_df['weeks'] == weeks]
            if len(week_data) > 0:
                value = week_data.iloc[0][metric_key]
                if np.isnan(value):
                    row += f"{'N/A':>13}  "
                else:
                    row += f"{value:>13.4f}  "
            else:
                row += f"{'N/A':>13}  "
        
        print(row)
    
    print(f"{'='*155}")
    
    # Calculate improvement from 1 week to 104 weeks (2 years)
    print(f"\n  IMPROVEMENT ANALYSIS (1 Week → 2 Years):")
    
    week1_data = results_df[results_df['weeks'] == 1].iloc[0] if len(results_df[results_df['weeks'] == 1]) > 0 else None
    week104_data = results_df[results_df['weeks'] == 104].iloc[0] if len(results_df[results_df['weeks'] == 104]) > 0 else None
    
    if week1_data is not None and week104_data is not None:
        if not np.isnan(week1_data['rmse']) and not np.isnan(week104_data['rmse']):
            rmse_improvement = ((week1_data['rmse'] - week104_data['rmse']) / week1_data['rmse']) * 100
            print(f"  • RMSE improved by {rmse_improvement:.1f}%")
        
        if not np.isnan(week1_data['mae']) and not np.isnan(week104_data['mae']):
            mae_improvement = ((week1_data['mae'] - week104_data['mae']) / week1_data['mae']) * 100
            print(f"  • MAE improved by {mae_improvement:.1f}%")
        
        if not np.isnan(week1_data['r2']) and not np.isnan(week104_data['r2']):
            r2_improvement = ((week104_data['r2'] - week1_data['r2']) / abs(week1_data['r2'])) * 100
            print(f"  • R² improved by {r2_improvement:.1f}%")
    else:
        print(f"  (Insufficient data for improvement calculation)")
    
    print(f"{'='*155}")


def evaluate_data_efficiency_with_switching(
    target_building,
    weeks_list=None,
    seq_length=24,
    experiment_name='rat_education',
    site_id='Rat',
    building_type='Education',
    margin_threshold_pct=2.0,
):
    """
    Evaluate PreTransfer and Transfer data-efficiency models then apply
    automatic model switching based on RMSE significance.

    For each week count the model with the lower RMSE is selected, provided the
    improvement exceeds *margin_threshold_pct* (default 2%). When the gap is
    smaller than the threshold, Transfer is preferred (warm-start bias).

    Args:
        target_building:       Building to evaluate on.
        weeks_list:            Week counts to sweep (default [1,2,4,8,16,32,64,104]).
        seq_length:            Sequence length used during training (default 24 h).
        experiment_name:       Subdirectory under models/experiments/.
        site_id:               Site filter forwarded to prepare_test_data.
        building_type:         Building-type filter forwarded to prepare_test_data.
        margin_threshold_pct:  Minimum RMSE % gap required to switch away from
                               the Transfer default (default 2.0).

    Returns:
        DataFrame with columns:
            weeks, pretransfer_mae, pretransfer_rmse, pretransfer_r2,
            transfer_mae, transfer_rmse, transfer_r2,
            selected_model, rmse_margin_pct, switched, decision_reason, confidence
    """
    import sys as _sys
    import os as _os
    _root_ = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    _sys.path.insert(0, _root_)
    _sys.path.insert(0, _os.path.join(_root_, 'src'))
    from src.switch_logic import apply_switching_to_df

    if weeks_list is None:
        weeks_list = [1, 2, 4, 8, 16, 32, 64, 104]

    print(f"\n{'=' * 90}")
    print(f"  SWITCH MODELLING EVALUATION  [{experiment_name}]")
    print(f"  Target: {target_building}  |  Threshold: {margin_threshold_pct}%")
    print(f"{'=' * 90}")

    pt_df = evaluate_data_efficiency(
        model_type='pretransfer',
        target_building=target_building,
        weeks_list=weeks_list,
        seq_length=seq_length,
        experiment_name=experiment_name,
        site_id=site_id,
        building_type=building_type,
    )

    tr_df = evaluate_data_efficiency(
        model_type='transfer',
        target_building=target_building,
        weeks_list=weeks_list,
        seq_length=seq_length,
        experiment_name=experiment_name,
        site_id=site_id,
        building_type=building_type,
    )

    switched_df = apply_switching_to_df(pt_df, tr_df, margin_threshold_pct)

    # Print per-week switching decisions
    print(f"\n{'─' * 90}")
    print(f"  {'Weeks':<8} {'PT RMSE':<12} {'TR RMSE':<12} {'Selected':<14} {'Margin %':<12} {'Switched':<10} {'Reason'}")
    print(f"{'─' * 90}")
    for _, row in switched_df.iterrows():
        pt_rmse = f"{row['pretransfer_rmse']:.4f}" if not pd.isna(row['pretransfer_rmse']) else 'N/A'
        tr_rmse = f"{row['transfer_rmse']:.4f}"    if not pd.isna(row['transfer_rmse'])    else 'N/A'
        margin  = f"{row['rmse_margin_pct']:.2f}%"  if not pd.isna(row['rmse_margin_pct'])  else 'N/A'
        switched_flag = '*** YES ***' if row['switched'] else 'no'
        print(
            f"  {int(row['weeks']):<8} {pt_rmse:<12} {tr_rmse:<12} "
            f"{str(row['selected_model']):<14} {margin:<12} {switched_flag:<10} {row['decision_reason']}"
        )
    print(f"{'─' * 90}")

    n_switches = int(switched_df['switched'].sum())
    total = len(switched_df)
    print(f"\n  Switches: {n_switches} / {total} week counts ({n_switches/total*100:.1f}%)")

    return switched_df


if __name__ == '__main__':
    main()
