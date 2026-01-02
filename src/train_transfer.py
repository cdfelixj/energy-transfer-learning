import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
import glob
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders
from models import EnergyLSTM

def train_transfer(source_building, target_building, 
                   source_model_path, epochs=20, seq_length=168):
    """Transfer learning: fine-tune on target building"""
    
    print(f"\n{'='*60}")
    print(f"Transfer Learning: {source_building} → {target_building}")
    print(f"{'='*60}")
    
    # Load electricity data
    electricity_path = r'../data/raw/building-data-genome-project-2/data/meters/raw/electricity.csv'
    electricity = pd.read_csv(electricity_path, index_col=0)
    electricity.index = pd.to_datetime(electricity.index)
    
    # Load weather data
    weather_path = r'../data/raw/building-data-genome-project-2/data/weather/weather.csv'
    metadata_path = r'../data/raw/building-data-genome-project-2/data/metadata/metadata.csv'
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        
        # Get site_id for target building
        metadata = pd.read_csv(metadata_path)
        site_id = metadata[metadata['building_id'] == target_building]['site_id'].values[0]
        weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
        weather_building = weather_building.reindex(electricity.index)
        print(f"Weather data loaded for site: {site_id}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_building = None
    
    # Preprocess target building
    target_data, target_scaler = preprocess_building_data(electricity, target_building, weather_building)
    print(f"Target data shape: {target_data.shape}")
    
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
        dirpath='../models',
        filename=f'transfer_{source_building[:6]}_{target_building[:6]}_{{epoch:02d}}_{{val_loss:.4f}}',
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
    # Load selected buildings
    selected_buildings = pd.read_csv('../data/processed/selected_buildings.csv')
    
    # Use first education building as source (has baseline model)
    education_buildings = selected_buildings[selected_buildings['primaryspaceusage'] == 'Education']
    source_building = education_buildings['building_id'].iloc[0]  # Eagle_education_Raul
    
    # Use an education-related building NOT used in baseline training
    # Cockatoo_lodging_Emory is a dormitory for College/University (education industry)
    # but primaryspaceusage is "Lodging/residential" so it wasn't trained in baseline
    target_building = 'Cockatoo_lodging_Emory'
    
    # Automatically find the latest baseline model
    model_files = glob.glob('../models/baseline_*.ckpt')
    
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
    
    train_transfer(source_building, target_building, source_model_path, epochs=20)
