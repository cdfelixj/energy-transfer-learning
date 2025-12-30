import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders
from models import EnergyLSTM

def train_baseline(building_id, epochs=50, seq_length=2160):
    """Train baseline LSTM on single building"""
    
    # Load data (Windows paths with r'' or \\ escaping)
    print(f"Loading data for building: {building_id}")
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
        
        # Get site_id for this building
        metadata = pd.read_csv(metadata_path)
        site_id = metadata[metadata['building_id'] == building_id]['site_id'].values[0]
        weather_building = weather[weather['site_id'] == site_id].drop(columns=['site_id'])
        weather_building = weather_building.reindex(electricity.index)
        print(f"Weather data loaded for site: {site_id}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_building = None
    
    # Preprocess
    data, scaler = preprocess_building_data(electricity, building_id, weather_building)
    print(f"Preprocessed shape: {data.shape}")
    
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
        learning_rate=1e-3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input features: {input_size}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='../models',
        filename=f'baseline_{building_id}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
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
    
    print(f"\nTraining complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE: {results[0]['test_rmse']:.4f}")
    print(f"Test MAE: {results[0]['test_mae']:.4f}")
    
    return model, results

if __name__ == '__main__':
    # Load selected buildings
    selected_buildings = pd.read_csv('../data/processed/selected_buildings.csv')
    source_building = selected_buildings['building_id'].iloc[0]
    
    print(f"Training baseline on building: {source_building}")
    model, results = train_baseline(source_building, epochs=50)
