"""
Pre-Transfer Model Training Script
Trains a model from scratch on limited data from the target building.
This serves as a baseline comparison to show how well a model can perform
WITHOUT transfer learning when only limited data is available.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM


def train_pretransfer(target_building, epochs=50, seq_length=24, 
                     data_limit_weeks=4, architecture_match=None):
    """
    Train a model from scratch on limited data from the target building.
    
    Args:
        target_building: Building ID to train on (with limited data)
        epochs: Number of training epochs
        seq_length: Sequence length in hours (default 24 = 1 day, suitable for limited data)
        data_limit_weeks: Number of weeks of data to use (default 4)
        architecture_match: Optional path to baseline model to match architecture
    """
    
    print(f"\n{'='*70}")
    print(f"  PRE-TRANSFER MODEL: Training from scratch on limited data")
    print(f"  Target Building: {target_building}")
    print(f"  Data limit: {data_limit_weeks} week(s)")
    print(f"{'='*70}")
    
    # Load filtered data (Education + Rat site + Electricity only)
    electricity, metadata, valid_buildings = load_electricity_data()
    
    # Validate that building is available
    print(f"\nValidating target building...")
    if target_building not in valid_buildings:
        raise ValueError(f"Target building '{target_building}' is not available. "
                        f"Available buildings: {valid_buildings[:10]}...")
    print(f"✓ Building validated\n")
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load weather data
    weather_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2',
                                'data', 'weather', 'weather.csv')
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        
        # Get site_id for target building
        site_id = metadata[metadata['building_id'] == target_building]['site_id'].values[0]
        weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
        weather_building = weather_building.reindex(electricity.index)
        print(f"Weather data loaded for site: {site_id}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_building = None
    
    # Preprocess target building
    target_data, target_scaler = preprocess_building_data(
        electricity, target_building, weather_building
    )
    print(f"Full target data shape: {target_data.shape}")
    
    # Limit data to simulate limited availability (e.g., 8 weeks)
    # Take the first N weeks of data
    hours_to_keep = data_limit_weeks * 7 * 24  # 7 days per week, 24 hours per day
    target_data = target_data.iloc[:hours_to_keep]
    print(f"Limited to {data_limit_weeks} week(s): {target_data.shape}")
    print(f"Date range: {target_data.index[0]} to {target_data.index[-1]}")
    
    # If architecture_match is provided, match the baseline model's architecture
    if architecture_match:
        print(f"\nMatching architecture from: {architecture_match}")
        baseline_model = EnergyLSTM.load_from_checkpoint(architecture_match)
        
        hidden_size = baseline_model.hparams.hidden_size
        num_layers = baseline_model.hparams.num_layers
        expected_input_size = baseline_model.hparams.input_size
        
        print(f"Baseline model architecture:")
        print(f"  Input size: {expected_input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
        
        # Check if target data has the right number of features
        actual_input_size = target_data.shape[1] - 1  # -1 for energy column
        print(f"\nTarget data has {actual_input_size} features (excluding energy)")
        
        if actual_input_size != expected_input_size:
            print(f"\nFeature mismatch detected!")
            print(f"Baseline model expects: {expected_input_size} features")
            print(f"Target data has: {actual_input_size} features")
            
            # Adjust features to match
            feature_cols = [col for col in target_data.columns if col != 'energy']
            
            if actual_input_size < expected_input_size:
                # Add missing features as zeros
                missing_count = expected_input_size - actual_input_size
                print(f"Adding {missing_count} zero-filled feature(s)")
                for i in range(missing_count):
                    target_data[f'missing_feature_{i}'] = 0.0
            else:
                # Remove extra features
                print(f"Removing {actual_input_size - expected_input_size} extra feature(s)")
                features_to_keep = feature_cols[:expected_input_size]
                target_data = target_data[['energy'] + features_to_keep]
            
            print(f"Adjusted target data shape: {target_data.shape}")
    else:
        # Use default architecture
        hidden_size = 128
        num_layers = 3
        expected_input_size = target_data.shape[1] - 1
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )
    
    # Initialize model from scratch (no pre-trained weights)
    input_size = train_loader.dataset.features.shape[1]
    
    # Use SIMPLER architecture for limited data to avoid collapse
    # 4 weeks = approximately 1 month
    if data_limit_weeks <= 4:
        # With very limited data, use smaller/simpler model
        model = EnergyLSTM(
            input_size=input_size,
            hidden_size=64,  # Reduced from 128
            num_layers=2,    # Reduced from 3
            dropout=0.1,     # Reduced dropout
            learning_rate=1e-3
        )
        print(f"\nUsing SIMPLIFIED architecture for limited data:")
        print(f"  Hidden size: 64 (reduced)")
        print(f"  Num layers: 2 (reduced)")
    else:
        # With more data, use full architecture
        model = EnergyLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            learning_rate=1e-3
        )
    
    print(f"  Input size: {input_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, 'models'),
        filename=f'pretransfer_{target_building[:15]}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
        gradient_clip_val=1.0
    )
    
    # Train
    print("\nStarting training from scratch on limited data...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nTesting...")
    results = trainer.test(model, test_loader)
    
    print(f"\nPre-transfer training complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE: {results[0]['test_rmse']:.4f}")
    print(f"Test MAE: {results[0]['test_mae']:.4f}")
    
    return model, results


if __name__ == '__main__':
    import glob
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Use a Rat education building NOT used in baseline training
    # Baseline uses: Angelica, Moises, Colin
    # We'll use a different one for target
    target_building = 'Rat_education_Denise'
    
    # Find baseline model to match architecture
    model_files = glob.glob(os.path.join(project_root, 'models', 'baseline_*.ckpt'))
    
    if not model_files:
        print("WARNING: No baseline model found")
        print("Training with default architecture")
        baseline_model_path = None
    else:
        baseline_model_path = max(model_files, key=os.path.getmtime)
        print(f"Found baseline model: {os.path.basename(baseline_model_path)}")
    
    print(f"\nTraining pre-transfer model:")
    print(f"  Target building: {target_building}")
    print(f"  Data limit: 1 month")
    print(f"  Training from scratch (no transfer learning)")
    
    try:
        model, results = train_pretransfer(
            target_building=target_building,
            epochs=100,  # Increased from 50 for better convergence
            seq_length=24,  # Use 24 hours (1 day) for limited data
            data_limit_weeks=8,  # 8 weeks of data
            architecture_match=baseline_model_path
        )
        
        print(f"\n✓ Pre-transfer model training complete!")
        print(f"  Test RMSE: {results[0]['test_rmse']:.4f}")
        print(f"  Test MAE:  {results[0]['test_mae']:.4f}")
        
        # Save summary
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        summary_df = pd.DataFrame([{
            'model_type': 'Pre-Transfer (No Transfer Learning)',
            'target_building': target_building,
            'data_months': 1,
            'test_rmse': results[0]['test_rmse'],
            'test_mae': results[0]['test_mae'],
            'test_loss': results[0]['test_loss'],
            'status': 'Success'
        }])
        
        summary_path = os.path.join(results_dir, 'pretransfer_training_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: results/pretransfer_training_summary.csv")
        
    except Exception as e:
        print(f"\n✗ Training failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
