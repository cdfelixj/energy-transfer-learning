"""
Multi-Transfer Training Script

Fine-tunes a multi-source pre-trained baseline on a target building with limited
data.  The pre-trained weights come from a baseline model trained across diverse
buildings/sites/types (the Multi-Source Baseline), rather than a single source
building as in train_transfer.py.

The fine-tuning protocol is identical to train_transfer.py:
  - All parameters are updated (full fine-tuning, no frozen backbone)
  - Low learning rate (1e-4) to preserve generalised representations
  - Early stopping with patience=5
  - Max 50 epochs
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import glob
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM


def train_multi_transfer(target_building,
                         multi_baseline_model_path,
                         epochs=50,
                         seq_length=24,
                         data_limit_weeks=4,
                         site_id='Eagle',
                         building_type='Education',
                         experiment_name='multi_transfer'):
    """Fine-tune a multi-source pre-trained model on a target building.

    Unlike train_transfer.py there is no single ``source_building`` — the
    backbone has been pre-trained on a pool of diverse buildings.  This
    function loads that checkpoint, clones its weights into a fresh
    EnergyLSTM, and fine-tunes on ``data_limit_weeks`` weeks of target data.

    Args:
        target_building: Building to adapt to (limited data).
        multi_baseline_model_path: Path to the multi-source baseline checkpoint.
        epochs: Maximum fine-tuning epochs (default 50).
        seq_length: Sequence length in hours (default 24).
        data_limit_weeks: Weeks of target data to use (default 4).
        site_id: Site filter for load_electricity_data (None = any site).
        building_type: Building type filter (None = any type).
        experiment_name: Determines checkpoint save directory.

    Returns:
        (model, results): Trained EnergyLSTM and test metrics dict list.
    """
    print(f"\n{'='*70}")
    print(f"  MULTI-TRANSFER: Fine-tuning multi-source pre-trained model")
    print(f"  Target: {target_building}")
    print(f"  Data limit: {data_limit_weeks} week(s)")
    print(f"{'='*70}")

    # ------------------------------------------------------------------ #
    # Load electricity data (filtered to target site/type)
    # ------------------------------------------------------------------ #
    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    print(f"\nValidating target building...")
    if target_building not in valid_buildings:
        raise ValueError(
            f"Target building '{target_building}' not found. "
            f"Available: {valid_buildings[:10]}..."
        )
    print(f"✓ Target building validated\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # ------------------------------------------------------------------ #
    # Load weather for target building's site
    # ------------------------------------------------------------------ #
    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv'
    )
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        target_site = metadata[
            metadata['building_id'] == target_building
        ]['site_id'].values[0]
        weather_building = weather[
            weather['site_id'] == target_site
        ].drop(columns=['site_id'])
        weather_building = weather_building.reindex(electricity.index)
        print(f"Weather data loaded for site: {target_site}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_building = None

    # ------------------------------------------------------------------ #
    # Preprocess & limit target data
    # ------------------------------------------------------------------ #
    target_data, _ = preprocess_building_data(
        electricity, target_building, weather_building
    )
    print(f"Full target data shape: {target_data.shape}")

    hours_to_keep = data_limit_weeks * 7 * 24
    target_data = target_data.iloc[:hours_to_keep]
    print(f"Limited to {data_limit_weeks} week(s): {target_data.shape}")
    print(f"Date range: {target_data.index[0]} to {target_data.index[-1]}")

    # ------------------------------------------------------------------ #
    # Load multi-source baseline to get architecture & weights
    # ------------------------------------------------------------------ #
    print(f"\nLoading multi-source baseline from: {multi_baseline_model_path}")
    source_model = EnergyLSTM.load_from_checkpoint(multi_baseline_model_path)
    expected_input_size = source_model.hparams.input_size
    print(f"Multi-source model expects {expected_input_size} input features")

    actual_input_size = target_data.shape[1] - 1  # exclude energy column
    print(f"Target data has {actual_input_size} features (excluding energy)")

    # ------------------------------------------------------------------ #
    # Feature alignment — multi-source baseline may have fewer features
    # (common_cols intersection across diverse sites)
    # ------------------------------------------------------------------ #
    if actual_input_size != expected_input_size:
        print(f"\nFeature mismatch — adjusting target data to {expected_input_size} features...")
        feature_cols = [col for col in target_data.columns if col != 'energy']

        if actual_input_size < expected_input_size:
            missing_count = expected_input_size - actual_input_size
            print(f"Adding {missing_count} zero-filled feature column(s)")
            for i in range(missing_count):
                target_data[f'missing_feature_{i}'] = 0.0
        else:
            print(f"Removing {actual_input_size - expected_input_size} extra feature column(s)")
            features_to_keep = feature_cols[:expected_input_size]
            target_data = target_data[['energy'] + features_to_keep]

        print(f"Adjusted target data shape: {target_data.shape}")

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )

    # ------------------------------------------------------------------ #
    # Clone multi-source baseline weights into new model
    # ------------------------------------------------------------------ #
    model = EnergyLSTM(
        input_size=source_model.hparams.input_size,
        hidden_size=source_model.hparams.hidden_size,
        num_layers=source_model.hparams.num_layers,
        dropout=0.2,
        learning_rate=1e-4,  # Low LR: preserve generalised multi-source representations
    )
    model.load_state_dict(source_model.state_dict())

    print(f"\nModel cloned from multi-source baseline")
    print(f"  Input size:  {model.hparams.input_size}")
    print(f"  Hidden size: {model.hparams.hidden_size}")
    print(f"  Num layers:  {model.hparams.num_layers}")
    print(f"  Parameters:  {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    exp_dir = os.path.join(project_root, 'models', 'experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename=f'multitransfer_{target_building[:15]}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    print("\nStarting multi-transfer fine-tuning...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTesting...")
    results = trainer.test(model, test_loader)

    print(f"\nMulti-Transfer complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE:  {results[0]['test_rmse']:.4f}")
    print(f"Test MAE:   {results[0]['test_mae']:.4f}")

    return model, results
