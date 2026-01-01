import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders
from models import EnergyLSTM

def train_baseline(building_ids, epochs=50, seq_length=336):
    """Train baseline LSTM on multiple buildings combined
    
    Args:
        building_ids: List of building identifiers to combine
        epochs: Number of training epochs
        seq_length: Sequence length in hours (default 336 = 2 weeks)
    """
    
    # Load data
    print(f"Loading data for {len(building_ids)} buildings...")
    
    # Get the project root directory (parent of src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    electricity_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2', 
                                     'data', 'meters', 'raw', 'electricity.csv')
    electricity = pd.read_csv(electricity_path, index_col=0)
    electricity.index = pd.to_datetime(electricity.index)
    
    print(f"Using full dataset: {electricity.index[0]} to {electricity.index[-1]}")
    print(f"Total hours available: {len(electricity)}")
    
    # Load weather and metadata
    weather_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2',
                                'data', 'weather', 'weather.csv')
    metadata_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2',
                                  'data', 'metadata', 'metadata.csv')
    
    weather = pd.read_csv(weather_path)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    weather = weather.set_index('timestamp')
    metadata = pd.read_csv(metadata_path)
    
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
        learning_rate=1e-3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input features: {input_size}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, 'models'),
        filename=f'baseline_combined_education_{{epoch:02d}}_{{val_loss:.4f}}',
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    selected_buildings = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'selected_buildings.csv'))
    
    # Filter to only education buildings
    education_buildings = selected_buildings[selected_buildings['primaryspaceusage'] == 'Education']
    buildings_to_train = education_buildings['building_id'].tolist()
    
    print("="*70)
    print(f"  TRAINING ONE BASELINE MODEL ON {len(buildings_to_train)} EDUCATION BUILDINGS")
    print("="*70)
    print("\nEducation buildings (combined training):")
    for i, building in enumerate(buildings_to_train, 1):
        print(f"  {i}. {building}")
    print()
    
    # Train single model on all buildings combined
    print("="*70)
    print("  TRAINING COMBINED BASELINE MODEL")
    print("="*70)
    
    try:
        # Train baseline model on ALL education buildings
        model, results = train_baseline(buildings_to_train, epochs=50)
        
        print(f"\n✓ Training complete!")
        print(f"  Test RMSE: {results[0]['test_rmse']:.4f}")
        print(f"  Test MAE:  {results[0]['test_mae']:.4f}")
        
        # Save summary
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        summary_df = pd.DataFrame([{
            'model_type': 'Combined Education Buildings',
            'buildings': ', '.join(buildings_to_train),
            'num_buildings': len(buildings_to_train),
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
