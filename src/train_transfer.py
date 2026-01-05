import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
import glob
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM

def train_transfer(source_building, target_building, 
                   source_model_path, epochs=20, seq_length=24, data_limit_weeks=4):
    """Transfer learning: fine-tune on target building with limited data
    
    Args:
        source_building: Building used to train baseline model
        target_building: Building to transfer to (with limited data)
        source_model_path: Path to pre-trained baseline model
        epochs: Number of fine-tuning epochs
        seq_length: Sequence length in hours (default 24 = 1 day, suitable for limited data)
        data_limit_weeks: Number of weeks of target data to use (default 4)
    """
    
    print(f"\n{'='*70}")
    print(f"  TRANSFER LEARNING: Fine-tuning on limited data")
    print(f"  {source_building} → {target_building}")
    print(f"  Data limit: {data_limit_weeks} week(s)")
    print(f"{'='*70}")
    
    # Load filtered data (Education + Rat site + Electricity only)
    electricity, metadata, valid_buildings = load_electricity_data()
    
    # Validate that both buildings are available
    print(f"\nValidating buildings...")
    if target_building not in valid_buildings:
        raise ValueError(f"Target building '{target_building}' is not available. "
                        f"Available buildings: {valid_buildings[:10]}...")
    if source_building not in valid_buildings:
        raise ValueError(f"Source building '{source_building}' is not available. "
                        f"Available buildings: {valid_buildings[:10]}...")
    print(f"✓ Both buildings validated\n")
    
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
    target_data, target_scaler = preprocess_building_data(electricity, target_building, weather_building)
    print(f"Full target data shape: {target_data.shape}")
    
    # Limit data to simulate limited availability (same as pre-transfer model)
    hours_to_keep = data_limit_weeks * 7 * 24  # 7 days per week, 24 hours per day
    target_data = target_data.iloc[:hours_to_keep]
    print(f"Limited to {data_limit_weeks} week(s): {target_data.shape}")
    print(f"Date range: {target_data.index[0]} to {target_data.index[-1]}")
    
    # Load pre-trained model first to get expected input size
    print(f"Loading pre-trained model from: {source_model_path}")
    source_model = EnergyLSTM.load_from_checkpoint(source_model_path)
    expected_input_size = source_model.hparams.input_size
    print(f"Source model expects {expected_input_size} input features")
    
    # Check if target data has the right number of features
    actual_input_size = target_data.shape[1] - 1  # -1 for energy column
    print(f"Target data has {actual_input_size} features (excluding energy)")
    
    if actual_input_size != expected_input_size:
        print(f"\nFeature mismatch detected!")
        print(f"Source model expects: {expected_input_size} features")
        print(f"Target data has: {actual_input_size} features")
        print(f"Target features: {[col for col in target_data.columns if col != 'energy']}")
        
        # Adjust features to match
        feature_cols = [col for col in target_data.columns if col != 'energy']
        
        if actual_input_size < expected_input_size:
            # Add missing features as zeros
            missing_count = expected_input_size - actual_input_size
            print(f"Adding {missing_count} zero-filled feature(s) to match source model")
            for i in range(missing_count):
                target_data[f'missing_feature_{i}'] = 0.0
        else:
            # Remove extra features (keep only the first expected_input_size)
            print(f"Removing {actual_input_size - expected_input_size} extra feature(s)")
            features_to_keep = feature_cols[:expected_input_size]
            target_data = target_data[['energy'] + features_to_keep]
        
        print(f"Adjusted target data shape: {target_data.shape}")
    
    # Create dataloaders (smaller batch size)
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )
    
    # Create transfer model (clone weights)
    transfer_model = EnergyLSTM(
        input_size=source_model.hparams.input_size,
        hidden_size=source_model.hparams.hidden_size,
        num_layers=source_model.hparams.num_layers,
        dropout=0.2,
        learning_rate=1e-4  # Lower learning rate for fine-tuning
    )
    transfer_model.load_state_dict(source_model.state_dict())
    
    print(f"Model loaded successfully")
    print(f"   Input size: {transfer_model.hparams.input_size}")
    print(f"   Hidden size: {transfer_model.hparams.hidden_size}")
    print(f"   Num layers: {transfer_model.hparams.num_layers}")
    
    # Train on target
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, 'models'),
        filename=f'transfer_{source_building[:15]}_{target_building[:15]}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=5,
        gradient_clip_val=1.0
    )
    
    print("\nStarting transfer learning...")
    trainer.fit(transfer_model, train_loader, val_loader)
    
    print("\nTesting...")
    results = trainer.test(transfer_model, test_loader)
    
    print(f"\nTransfer learning complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE: {results[0]['test_rmse']:.4f}")
    print(f"Test MAE: {results[0]['test_mae']:.4f}")
    
    return transfer_model, results

if __name__ == '__main__':
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Source building: one of the buildings used in baseline
    source_building = 'Rat_education_Angelica'
    
    # Target building: a Rat education building NOT used in baseline training
    # Baseline uses: Angelica, Moises, Colin
    # We'll use Denise for transfer (same as pre-transfer)
    target_building = 'Rat_education_Denise'
    
    # Automatically find the latest baseline model
    model_files = glob.glob(os.path.join(project_root, 'models', 'baseline_*.ckpt'))
    
    if not model_files:
        print("ERROR: No baseline model found in ../models/")
        print("Please run train_baseline.py first to create a source model.")
        sys.exit(1)
    
    # Use the most recent model
    source_model_path = max(model_files, key=os.path.getmtime)
    
    print(f"\nFound {len(model_files)} baseline model(s)")
    print(f"Using: {os.path.basename(source_model_path)}")
    print(f"Source building: {source_building} (Education - has baseline model)")
    print(f"Target building: {target_building} (Dormitory - NO baseline model)")
    print("This ensures proper transfer learning evaluation!\n")
    
    # Train transfer model with same limited data as pre-transfer (8 weeks)
    train_transfer(source_building, target_building, source_model_path, 
                  epochs=20, seq_length=24, data_limit_weeks=8)
