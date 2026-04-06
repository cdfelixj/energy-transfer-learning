import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import torch
import glob
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM, EnergyLSTMAdapter


def train_adapter(source_building, target_building,
                  source_model_path, epochs=50, seq_length=24, data_limit_weeks=4,
                  site_id='Rat', building_type='Education',
                  experiment_name='rat_education', adapter_bottleneck=32):
    """Adapter-layer transfer: frozen LSTM + small trainable bottleneck.

    The pre-trained LSTM weights are frozen after loading.  A lightweight
    LSTMAdapter module (Linear(128→32)→ReLU→Linear(32→128), residual) is
    inserted between the LSTM output and the MLP head.  Only the adapter
    (~8 K params) and the head (~8 K params) are updated during training.

    Args:
        source_building: Building used to train the baseline model.
        target_building: Building to adapt to (limited data).
        source_model_path: Path to the pre-trained baseline checkpoint.
        epochs: Maximum fine-tuning epochs.
        seq_length: Sequence length in hours.
        data_limit_weeks: Weeks of target data to use.
        site_id: Site filter for load_electricity_data.
        building_type: Building type filter for load_electricity_data.
        experiment_name: Determines checkpoint save directory.
        adapter_bottleneck: Bottleneck dimension for the adapter module.
    """

    print(f"\n{'='*70}")
    print(f"  ADAPTER LAYERS: Frozen LSTM + trainable adapter (bottleneck={adapter_bottleneck})")
    print(f"  {source_building} → {target_building}")
    print(f"  Data limit: {data_limit_weeks} week(s)")
    print(f"{'='*70}")

    # Load filtered data
    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    print(f"\nValidating buildings...")
    if target_building not in valid_buildings:
        raise ValueError(f"Target building '{target_building}' is not available. "
                         f"Available buildings: {valid_buildings[:10]}...")
    if source_building not in valid_buildings:
        raise ValueError(f"Source building '{source_building}' is not available. "
                         f"Available buildings: {valid_buildings[:10]}...")
    print(f"✓ Both buildings validated\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Load weather data
    weather_path = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2',
                                'data', 'weather', 'weather.csv')
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        site_id_target = metadata[metadata['building_id'] == target_building]['site_id'].values[0]
        weather_building = weather[weather['site_id'] == site_id_target].drop(columns=['site_id'])
        weather_building = weather_building.reindex(electricity.index)
        print(f"Weather data loaded for site: {site_id_target}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_building = None

    # Preprocess and limit target data
    target_data, target_scaler = preprocess_building_data(electricity, target_building, weather_building)
    print(f"Full target data shape: {target_data.shape}")

    hours_to_keep = data_limit_weeks * 7 * 24
    target_data = target_data.iloc[:hours_to_keep]
    print(f"Limited to {data_limit_weeks} week(s): {target_data.shape}")
    print(f"Date range: {target_data.index[0]} to {target_data.index[-1]}")

    # Load source model and resolve feature mismatch
    print(f"Loading pre-trained model from: {source_model_path}")
    source_model = EnergyLSTM.load_from_checkpoint(source_model_path)
    expected_input_size = source_model.hparams.input_size
    print(f"Source model expects {expected_input_size} input features")

    actual_input_size = target_data.shape[1] - 1
    print(f"Target data has {actual_input_size} features (excluding energy)")

    if actual_input_size != expected_input_size:
        print(f"\nFeature mismatch detected!")
        feature_cols = [col for col in target_data.columns if col != 'energy']
        if actual_input_size < expected_input_size:
            missing_count = expected_input_size - actual_input_size
            print(f"Adding {missing_count} zero-filled feature(s)")
            for i in range(missing_count):
                target_data[f'missing_feature_{i}'] = 0.0
        else:
            features_to_keep = feature_cols[:expected_input_size]
            target_data = target_data[['energy'] + features_to_keep]
        print(f"Adjusted target data shape: {target_data.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )

    # Build adapter model and copy LSTM + head weights from source
    adapter_model = EnergyLSTMAdapter(
        input_size=source_model.hparams.input_size,
        hidden_size=source_model.hparams.hidden_size,
        num_layers=source_model.hparams.num_layers,
        dropout=0.2,
        learning_rate=1e-3,
        adapter_bottleneck=adapter_bottleneck,
    )
    # Copy matching keys (lstm.* and fc.*); adapter.* is left with near-zero init
    source_state = source_model.state_dict()
    adapter_state = adapter_model.state_dict()
    for key in source_state:
        if key in adapter_state and adapter_state[key].shape == source_state[key].shape:
            adapter_state[key] = source_state[key]
    adapter_model.load_state_dict(adapter_state)

    trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in adapter_model.parameters())
    print(f"\nModel loaded successfully")
    print(f"   Input size       : {adapter_model.hparams.input_size}")
    print(f"   Hidden size      : {adapter_model.hparams.hidden_size}")
    print(f"   Num layers       : {adapter_model.hparams.num_layers}")
    print(f"   Adapter bottleneck: {adapter_bottleneck}")
    print(f"   Total params     : {total_params:,}")
    print(f"   Trainable        : {trainable_params:,}  (adapter + head)")

    exp_dir = os.path.join(project_root, 'models', 'experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename=f'adapter_{source_building[:15]}_{target_building[:15]}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    print("\nStarting adapter fine-tuning...")
    trainer.fit(adapter_model, train_loader, val_loader)

    print("\nTesting...")
    results = trainer.test(adapter_model, test_loader)

    print(f"\nAdapter training complete!")
    print(f"Best model : {checkpoint_callback.best_model_path}")
    print(f"Test RMSE  : {results[0]['test_rmse']:.4f}")
    print(f"Test MAE   : {results[0]['test_mae']:.4f}")

    return adapter_model, results


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    source_building = 'Rat_education_Colin'
    target_building = 'Rat_education_Denise'

    model_files = glob.glob(os.path.join(project_root, 'models', 'baseline_*.ckpt'))
    if not model_files:
        print("ERROR: No baseline model found in ../models/")
        print("Please run train_baseline.py first.")
        sys.exit(1)

    source_model_path = max(model_files, key=os.path.getmtime)
    print(f"Using: {os.path.basename(source_model_path)}")

    model, results = train_adapter(
        source_building, target_building, source_model_path,
        epochs=50, seq_length=24, data_limit_weeks=8
    )
    print(f"\n✓ Complete  |  RMSE={results[0]['test_rmse']:.4f}  MAE={results[0]['test_mae']:.4f}")
