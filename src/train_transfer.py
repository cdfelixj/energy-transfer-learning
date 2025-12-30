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
    
    # Create dataloaders (smaller batch size)
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )
    
    # Load pre-trained model
    print(f"Loading pre-trained model from: {source_model_path}")
    source_model = EnergyLSTM.load_from_checkpoint(source_model_path)
    
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
    source_building = selected_buildings['building_id'].iloc[0]
    target_building = selected_buildings['building_id'].iloc[1]
    
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
    print(f"Source building: {source_building}")
    print(f"Target building: {target_building}")
    
    train_transfer(source_building, target_building, source_model_path, epochs=20)
