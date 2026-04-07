"""
Ensemble Transfer Training Script

Implements Model-Soup / Weighted-Average Transfer Learning:

  1. Load N individually-trained source-building models
  2. Compute a weighted average of their parameters
  3. Fine-tune the averaged model on the target building with limited data

All source models MUST share the same architecture (input_size, hidden_size,
num_layers).  Use ``train_individual_source`` (below) to train each source model
with a common feature set derived from a reference checkpoint.

Weighting schemes
─────────────────
  weights=None  → uniform (1/N each)
  weights=[w1, w2, ...]  → custom; automatically normalised to sum=1

Fine-tuning protocol (identical to Full Fine-Tuning / train_transfer.py):
  lr = 1e-4  |  patience = 5  |  max_epochs = 50  |  full parameter update
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Weight averaging
# ─────────────────────────────────────────────────────────────────────────────

def average_model_weights(model_paths, weights=None):
    """Load EnergyLSTM checkpoints and return a weighted-average state dict.

    All checkpoints must share the same architecture.  Weights are normalised
    to sum to 1.0 before averaging.

    Args:
        model_paths: List of paths to EnergyLSTM .ckpt files.
        weights: Optional list of non-negative floats (same length as
                 model_paths).  Pass None for uniform averaging.

    Returns:
        (avg_state_dict, hparams): Averaged state dict and hyperparameters
        from the first model (all models share the same architecture).

    Raises:
        ValueError: If models have incompatible architectures.
    """
    if not model_paths:
        raise ValueError('model_paths must not be empty.')

    print(f"\nLoading {len(model_paths)} source models for weight averaging...")
    models = []
    for path in model_paths:
        print(f"  Loading: {os.path.basename(path)}")
        models.append(EnergyLSTM.load_from_checkpoint(path))

    # ── Architecture compatibility check ─────────────────────────────────
    input_sizes  = [m.hparams.input_size  for m in models]
    hidden_sizes = [m.hparams.hidden_size for m in models]
    num_layers   = [m.hparams.num_layers  for m in models]

    if len(set(input_sizes)) > 1:
        raise ValueError(
            f"Source models have mismatched input sizes: {input_sizes}.\n"
            "Train all individual models with the same feature reference "
            "(use train_individual_source with feature_reference_path)."
        )
    if len(set(hidden_sizes)) > 1 or len(set(num_layers)) > 1:
        raise ValueError(
            f"Source models have incompatible architectures. "
            f"hidden={hidden_sizes}, layers={num_layers}"
        )

    # ── Normalise weights ─────────────────────────────────────────────────
    N = len(models)
    if weights is None:
        weights = [1.0 / N] * N
    else:
        if len(weights) != N:
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(model_paths)={N}."
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number.")
        weights = [w / total for w in weights]

    print(f"\nAveraging weights:")
    for path, w in zip(model_paths, weights):
        print(f"  {os.path.basename(path):50s}  weight={w:.4f}")

    # ── Weighted sum of state dicts ───────────────────────────────────────
    avg_state = {}
    ref_state = models[0].state_dict()
    for key in ref_state:
        avg_state[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(weights, models)
        )

    print(f"\n✓ Weight averaging complete  (input_size={input_sizes[0]})")
    return avg_state, models[0].hparams


# ─────────────────────────────────────────────────────────────────────────────
# Individual source-model training (feature-aligned)
# ─────────────────────────────────────────────────────────────────────────────

def train_individual_source(source_building,
                             epochs=50,
                             seq_length=168,
                             feature_reference_path=None,
                             site_id=None,
                             building_type=None,
                             experiment_name='ensemble_transfer'):
    """Train a SINGLE source building baseline, optionally aligned to a reference
    model's input size.

    This is used to produce N individually-trained source models that all share
    the same input_size so their weights can later be averaged.

    Args:
        source_building: Building ID to train on (full 2-year data).
        epochs: Training epochs (default 50).
        seq_length: Sequence length in hours (default 168 = 1 week).
        feature_reference_path: Path to any EnergyLSTM checkpoint.  If given,
            the source building's features are truncated / zero-padded to match
            that model's input_size.  This ensures all individual models share
            the same architecture for subsequent weight averaging.
        site_id: Site filter for load_electricity_data (None = all sites).
        building_type: Building type filter (None = all types).
        experiment_name: Determines checkpoint save directory.

    Returns:
        (model, results): Trained EnergyLSTM and test metrics.
    """
    print(f"\n{'='*70}")
    print(f"  INDIVIDUAL SOURCE BASELINE: {source_building}")
    if feature_reference_path:
        print(f"  Feature reference: {os.path.basename(feature_reference_path)}")
    print(f"{'='*70}")

    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    if source_building not in valid_buildings:
        raise ValueError(
            f"Source building '{source_building}' not found. "
            f"Available: {valid_buildings[:10]}..."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Weather for this building's site
    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv'
    )
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        src_site = metadata[
            metadata['building_id'] == source_building
        ]['site_id'].values[0]
        weather_src = weather[weather['site_id'] == src_site].drop(columns=['site_id'])
        weather_src = weather_src.reindex(electricity.index)
        print(f"Weather data loaded for site: {src_site}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_src = None

    src_data, _ = preprocess_building_data(electricity, source_building, weather_src)
    print(f"Full source data shape: {src_data.shape}")

    # ── Feature alignment ─────────────────────────────────────────────────
    if feature_reference_path:
        ref_model = EnergyLSTM.load_from_checkpoint(feature_reference_path)
        target_input_size = ref_model.hparams.input_size
        actual_input_size = src_data.shape[1] - 1  # excluding energy column

        if actual_input_size != target_input_size:
            print(f"\nAligning features: {actual_input_size} → {target_input_size}")
            feature_cols = [c for c in src_data.columns if c != 'energy']
            if actual_input_size > target_input_size:
                features_to_keep = feature_cols[:target_input_size]
                src_data = src_data[['energy'] + features_to_keep]
            else:
                for i in range(target_input_size - actual_input_size):
                    src_data[f'missing_feature_{i}'] = 0.0
            print(f"  Adjusted shape: {src_data.shape}")
    else:
        target_input_size = src_data.shape[1] - 1

    train_loader, val_loader, test_loader = create_dataloaders(
        src_data, seq_length=seq_length, batch_size=32
    )

    input_size = train_loader.dataset.features.shape[1]
    model = EnergyLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        learning_rate=5e-4,
    )
    print(f"\nModel: input={input_size}  hidden=128  layers=3  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    exp_dir = os.path.join(project_root, 'models', 'experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename=f'individual_{source_building[:20]}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=7, mode='min')

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    print(f"\nTraining {source_building} (full data)...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTesting...")
    results = trainer.test(model, test_loader)

    print(f"\nIndividual source training complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE:  {results[0]['test_rmse']:.4f}")
    print(f"Test MAE:   {results[0]['test_mae']:.4f}")

    return model, results


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Transfer (fine-tune averaged model on target)
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble_transfer(target_building,
                             source_model_paths,
                             model_weights=None,
                             epochs=50,
                             seq_length=24,
                             data_limit_weeks=4,
                             site_id='Eagle',
                             building_type='Education',
                             experiment_name='ensemble_transfer'):
    """Average source model weights, then fine-tune on a target building.

    Args:
        target_building: Building to adapt to (limited data).
        source_model_paths: List of EnergyLSTM checkpoint paths to average.
            All must have the same architecture.
        model_weights: Optional list of relative weights for averaging.
            None → uniform (1/N).
        epochs: Max fine-tuning epochs (default 50).
        seq_length: Sequence length in hours (default 24).
        data_limit_weeks: Weeks of target data to use.
        site_id: Site filter for load_electricity_data.
        building_type: Building type filter.
        experiment_name: Determines checkpoint save directory.

    Returns:
        (model, results): Fine-tuned EnergyLSTM and test metrics.
    """
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE TRANSFER: Averaged-weights → fine-tune")
    print(f"  Target: {target_building}")
    print(f"  Sources: {len(source_model_paths)} models")
    print(f"  Data limit: {data_limit_weeks} week(s)")
    print(f"{'='*70}")

    # ── Average source model weights ──────────────────────────────────────
    avg_state, source_hparams = average_model_weights(source_model_paths, model_weights)

    # ── Load target building data ─────────────────────────────────────────
    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    if target_building not in valid_buildings:
        raise ValueError(
            f"Target building '{target_building}' not found. "
            f"Available: {valid_buildings[:10]}..."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv'
    )
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        tgt_site = metadata[
            metadata['building_id'] == target_building
        ]['site_id'].values[0]
        weather_tgt = weather[weather['site_id'] == tgt_site].drop(columns=['site_id'])
        weather_tgt = weather_tgt.reindex(electricity.index)
        print(f"\nWeather data loaded for site: {tgt_site}")
    except Exception as e:
        print(f"Warning: Could not load weather data: {e}")
        weather_tgt = None

    target_data, _ = preprocess_building_data(electricity, target_building, weather_tgt)
    print(f"Full target data shape: {target_data.shape}")

    hours_to_keep = data_limit_weeks * 7 * 24
    target_data = target_data.iloc[:hours_to_keep]
    print(f"Limited to {data_limit_weeks} week(s): {target_data.shape}")
    print(f"Date range: {target_data.index[0]} to {target_data.index[-1]}")

    # ── Feature alignment: target → averaged model's input_size ──────────
    expected_input_size = source_hparams.input_size
    actual_input_size = target_data.shape[1] - 1

    if actual_input_size != expected_input_size:
        print(f"\nFeature mismatch — aligning {actual_input_size} → {expected_input_size} features...")
        feature_cols = [c for c in target_data.columns if c != 'energy']
        if actual_input_size > expected_input_size:
            features_to_keep = feature_cols[:expected_input_size]
            target_data = target_data[['energy'] + features_to_keep]
        else:
            for i in range(expected_input_size - actual_input_size):
                target_data[f'missing_feature_{i}'] = 0.0
        print(f"  Adjusted shape: {target_data.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, seq_length=seq_length, batch_size=32
    )

    # ── Initialise model from averaged weights ────────────────────────────
    model = EnergyLSTM(
        input_size=source_hparams.input_size,
        hidden_size=source_hparams.hidden_size,
        num_layers=source_hparams.num_layers,
        dropout=0.2,
        learning_rate=1e-4,   # low LR: preserve averaged representations
    )
    model.load_state_dict(avg_state)

    print(f"\nModel initialised from {len(source_model_paths)}-model weight average")
    print(f"  Input size:  {model.hparams.input_size}")
    print(f"  Hidden size: {model.hparams.hidden_size}")
    print(f"  Num layers:  {model.hparams.num_layers}")
    print(f"  Parameters:  {sum(p.numel() for p in model.parameters()):,}")

    # ── Fine-tuning ───────────────────────────────────────────────────────
    exp_dir = os.path.join(project_root, 'models', 'experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename=f'ensembletransfer_{target_building[:15]}_{{epoch:02d}}_{{val_loss:.4f}}',
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

    print("\nStarting ensemble-transfer fine-tuning...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTesting...")
    results = trainer.test(model, test_loader)

    print(f"\nEnsemble Transfer complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Test RMSE:  {results[0]['test_rmse']:.4f}")
    print(f"Test MAE:   {results[0]['test_mae']:.4f}")

    return model, results
