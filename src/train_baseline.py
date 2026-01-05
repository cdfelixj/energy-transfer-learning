import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM

def train_baseline(building_ids, epochs=50, seq_length=336):
    """Train baseline LSTM on multiple buildings combined
    
    Args:
        building_ids: List of building identifiers to combine
        epochs: Number of training epochs
        seq_length: Sequence length in hours (default 336 = 2 weeks)
    """
    
    # Load filtered data (Education + Rat site + Electricity only)
    electricity, metadata, valid_buildings = load_electricity_data()
    
    # Validate that all requested building_ids are in the filtered set
    print(f"\nValidating {len(building_ids)} requested buildings...")
    invalid_buildings = [bid for bid in building_ids if bid not in valid_buildings]
    if invalid_buildings:
        raise ValueError(f"The following buildings are not available: {invalid_buildings}. "
                        f"Available buildings: {valid_buildings[:10]}...")
    print(f"✓ All buildings validated\n")
    
    # Load weather
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    weather_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2',
                                'data', 'weather', 'weather.csv')
    
    weather = pd.read_csv(weather_path)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    weather = weather.set_index('timestamp')
    
    # Combine data from all buildings
    combined_data = []
    
    for building_id in building_ids:
        print(f"\nProcessing {building_id}...")
        
        try:
            # Get weather for this building
            site_id = metadata[metadata['building_id'] == building_id]['site_id'].values[0]
            weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
            weather_building = weather_building.reindex(electricity.index)
            print(f"  Weather data loaded for site: {site_id}")
        except Exception as e:
            print(f"  Warning: Could not load weather data: {e}")
            weather_building = None
        
        # Preprocess this building
        data, scaler = preprocess_building_data(electricity, building_id, weather_building)
        combined_data.append(data)
        print(f"  Preprocessed shape: {data.shape}")
    
    # Concatenate all buildings - first align features
    print(f"\nCombining data from {len(building_ids)} buildings...")
    
    # Find common columns across all buildings
    common_cols = set(combined_data[0].columns)
    for df in combined_data[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    common_cols = sorted(list(common_cols))  # Sort for consistency
    print(f"  Using {len(common_cols)} common features across all buildings")
    
    # Filter each dataframe to only use common columns
    combined_data = [df[common_cols] for df in combined_data]
    
    data = pd.concat(combined_data, axis=0)
    print(f"Combined data shape: {data.shape}")
    
    # Create dataloaders (smaller batch size for long sequences)
    train_loader, val_loader, test_loader = create_dataloaders(
        data, seq_length=seq_length, batch_size=32
    )
    
    # Initialize model
    input_size = train_loader.dataset.features.shape[1]
    model = EnergyLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        learning_rate=5e-4  # Lower learning rate for better convergence
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input features: {input_size}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, 'models'),
        filename=f'baseline_{buildings_to_train[0][:20]}_2yr_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased from 10 - give model more time to improve
        mode='min'
    )
    
    # Trainer with gradient clipping
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',  # 'cpu' for Windows without GPU, 'gpu' if you have CUDA
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nTesting...")
    results = trainer.test(model, test_loader)
    
    # DIAGNOSTIC: Check if model is predicting constants
    print("\n=== DIAGNOSTIC: Model Predictions ===")
    model.eval()
    import numpy as np
    
    with torch.no_grad():
        # Get predictions from test set
        test_preds = []
        test_actuals = []
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            test_preds.extend(y_hat.squeeze().cpu().numpy())
            test_actuals.extend(y.squeeze().cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_actuals = np.array(test_actuals)
    
    print(f"Test Predictions - Mean: {np.mean(test_preds):.2f}, Std: {np.std(test_preds):.2f}, Range: [{np.min(test_preds):.2f}, {np.max(test_preds):.2f}]")
    print(f"Test Actuals     - Mean: {np.mean(test_actuals):.2f}, Std: {np.std(test_actuals):.2f}, Range: [{np.min(test_actuals):.2f}, {np.max(test_actuals):.2f}]")
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(test_actuals, test_preds)
    print(f"Test R²: {r2:.4f}")
    
    if np.std(test_preds) < 1.0:
        print("\n⚠ WARNING: Model is predicting near-constant values!")
        print("  This indicates model collapse - the model is not learning patterns.")
    
    if r2 < 0:
        print("\n⚠ WARNING: Negative R² indicates model is worse than predicting the mean!")
        print(f"  Mean prediction would give R² = 0, but this model gives R² = {r2:.4f}")
    
    print(f"\nTraining complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE: {results[0]['test_rmse']:.4f}")
    print(f"Test MAE: {results[0]['test_mae']:.4f}")
    
    return model, results

if __name__ == '__main__':
    # Load selected buildings from Rat site (only available buildings)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Use only 1 Rat education building with 2 years of data (2016-2017)
    # This provides abundant source data for baseline model
    buildings_to_train = [
        'Rat_education_Colin'  # Has high data quality (99.57% non-null)
    ]
    
    print("="*70)
    print(f"  TRAINING BASELINE MODEL ON {len(buildings_to_train)} BUILDING (2 YEARS)")
    print("="*70)
    print(f"\nSource building: {buildings_to_train[0]}")
    print(f"Data: 2016-2017 (~2 years)")
    print()
    
    # Train single model on all buildings combined
    print("="*70)
    print("  TRAINING BASELINE MODEL ON 2 YEARS OF DATA")
    print("="*70)
    
    try:
        # Train baseline model - try shorter sequence length for better convergence
        # seq_length=168 (1 week) instead of 336 (2 weeks) - easier to learn
        model, results = train_baseline(buildings_to_train, epochs=50, seq_length=168)
        
        print(f"\n✓ Training complete!")
        print(f"  Test RMSE: {results[0]['test_rmse']:.4f}")
        print(f"  Test MAE:  {results[0]['test_mae']:.4f}")
        
        # Save summary
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        summary_df = pd.DataFrame([{
            'model_type': 'Baseline (Single Building, 2 Years)',
            'building': buildings_to_train[0],
            'data_years': 2,
            'test_rmse': results[0]['test_rmse'],
            'test_mae': results[0]['test_mae'],
            'test_loss': results[0]['test_loss'],
            'status': 'Success'
        }])
        
        summary_df.to_csv(os.path.join(results_dir, 'baseline_training_summary.csv'), index=False)
        print(f"\n✓ Summary saved to: results/baseline_training_summary.csv")
        
    except Exception as e:
        print(f"\n✗ Training failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
