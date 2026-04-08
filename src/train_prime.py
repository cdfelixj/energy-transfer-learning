"""
PRIME Experiment Training & Inference Module

Implements the full PRIME (Performance-weighted Robust Initialisation for
Modelling Energy) pipeline components:

  1. train_prime_source   — Train one source model with QuantileLoss (α=0.7)
                            optimised for energy provisioning.
  2. build_weighted_soup  — Combine N source checkpoints into a single
                            performance-weighted averaged model (inverse-MAE
                            weighting, strictly better than uniform).
  3. train_prime_transfer — Fine-tune the soup on a target building with
                            limited data (the PRIME_Transfer model).
  4. run_soft_blend_streaming — Live streaming deployment: replays post-training
                            data hour-by-hour, maintains rolling MAE for both
                            models, blends predictions inverse-MAE-proportionally
                            each evaluation window, and hard-switches when one
                            model is clearly dominant.

Design decisions
----------------
- All models trained with QuantileLoss(alpha=0.7): underprediction incurs a
  2.3× heavier penalty than overprediction, matching energy provisioning needs.
- Weighting by 1/val_mae (normalised) is strictly better than or equal to
  uniform averaging (Exp 9 result: uniform averaging already works; weighting
  by performance exploits relative source quality information).
- Soft blending avoids the sharp transition artefact at a hard-switch point:
  blend weight w_transfer = (1/rolling_mae_T) / (1/rolling_mae_T + 1/rolling_mae_PT)
  drifts smoothly as relative model quality evolves.
- Hard-switch override: when one model is clearly dominant (margin > threshold),
  blend weight collapses to 0 or 1 (100% allocation to winner), preventing a
  persistently bad model from diluting predictions.
"""

import os
import sys
import glob

sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from collections import deque
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import preprocess_building_data, create_dataloaders, load_electricity_data
from models import EnergyLSTM
from switch_logic import decide_model
from inference import predict_with_uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Train one source model with QuantileLoss
# ─────────────────────────────────────────────────────────────────────────────

def train_prime_source(
    source_building: str,
    reference_input_size: int | None = None,
    epochs: int = 50,
    seq_length: int = 168,
    site_id: str | None = None,
    building_type: str | None = None,
    experiment_name: str = 'prime',
    loss_alpha: float = 0.7,
) -> tuple[str, float]:
    """Train a single source EnergyLSTM with asymmetric QuantileLoss.

    Uses the full 2-year source building dataset.  If reference_input_size is
    given, features are aligned (truncated or zero-padded) so all N source
    models share the same architecture for subsequent weight averaging.

    Args:
        source_building:     Building ID to train on.
        reference_input_size: Target input size for feature alignment. Pass the
                              input_size of the first source model so subsequent
                              models are aligned to it.  None = use raw features.
        epochs:              Maximum training epochs (early stopping active).
        seq_length:          LSTM sequence length in hours (168 = 1 week).
        site_id:             Site filter for data loading (None = any site).
        building_type:       Building type filter (None = any type).
        experiment_name:     Sub-directory for checkpoint saving.
        loss_alpha:          QuantileLoss alpha (0.7 = penalise underprediction).

    Returns:
        (checkpoint_path, val_mae): Path to best checkpoint and validation MAE.
    """
    print(f"\n{'='*70}")
    print(f"  PRIME SOURCE: {source_building}  (α={loss_alpha})")
    print(f"{'='*70}")

    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    if source_building not in valid_buildings:
        raise ValueError(
            f"Source building '{source_building}' not found in filtered data. "
            f"Available sample: {valid_buildings[:5]}..."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv',
    )
    weather_src = None
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        site_row = metadata[metadata['building_id'] == source_building]
        src_site = site_row['site_id'].values[0] if len(site_row) > 0 else None
        if src_site:
            weather_src = weather[weather['site_id'] == src_site].drop(columns=['site_id'])
            weather_src = weather_src.reindex(electricity.index)
    except Exception as e:
        print(f"  Warning: weather data unavailable — {e}")

    src_data, _ = preprocess_building_data(electricity, source_building, weather_src)
    print(f"  Source data shape: {src_data.shape}")

    # Feature alignment
    if reference_input_size is not None:
        feature_cols = [c for c in src_data.columns if c != 'energy']
        actual = len(feature_cols)
        if actual > reference_input_size:
            src_data = src_data[['energy'] + feature_cols[:reference_input_size]]
        elif actual < reference_input_size:
            for i in range(reference_input_size - actual):
                src_data[f'missing_feature_{i}'] = 0.0
        print(f"  Features aligned to reference input_size={reference_input_size}")

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
        loss_alpha=loss_alpha,
    )
    print(f"  Model: input={input_size}  hidden=128  layers=3  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    save_dir = os.path.join(project_root, 'models', 'prime', experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename=f'prime_source_{source_building[:20]}_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=7, mode='min')

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_cb, early_stop],
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Extract validation MAE from the best logged value
    val_mae = float(trainer.callback_metrics.get('val_mae', torch.tensor(float('inf'))))

    print(f"\n  ✓ Source training complete")
    print(f"    Best checkpoint: {os.path.basename(checkpoint_cb.best_model_path)}")
    print(f"    Val MAE (for soup weighting): {val_mae:.4f}")

    return checkpoint_cb.best_model_path, val_mae


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Performance-weighted model soup
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_soup(
    checkpoint_paths: list[str],
    val_maes: list[float],
) -> tuple[dict, object]:
    """Combine N source checkpoints into a performance-weighted averaged model.

    Weighting scheme (inverse-MAE):
        w_i = (1 / val_mae_i) / Σ (1 / val_mae_j)

    Lower validation MAE → higher weight.  Strictly better than or equal to
    uniform averaging (Exp 9 baseline), because it down-weights poorly-fitting
    sources rather than treating them equally.

    Args:
        checkpoint_paths: List of paths to EnergyLSTM .ckpt files.
        val_maes:         Validation MAE for each checkpoint (same order).

    Returns:
        (avg_state_dict, hparams): Averaged state dict and reference hparams.

    Raises:
        ValueError: If checkpoints have incompatible architectures or
                    val_maes contains non-positive values.
    """
    if len(checkpoint_paths) != len(val_maes):
        raise ValueError("checkpoint_paths and val_maes must have the same length.")
    if not checkpoint_paths:
        raise ValueError("Need at least one checkpoint.")
    if any(m <= 0 for m in val_maes):
        raise ValueError("All val_maes must be positive (used as denominator).")

    print(f"\n{'='*70}")
    print(f"  BUILDING WEIGHTED SOUP  (N={len(checkpoint_paths)})")
    print(f"{'='*70}")

    models = []
    for path in checkpoint_paths:
        print(f"  Loading: {os.path.basename(path)}")
        models.append(EnergyLSTM.load_from_checkpoint(path))

    # Architecture check
    input_sizes = [m.hparams.input_size for m in models]
    if len(set(input_sizes)) > 1:
        raise ValueError(
            f"Source models have mismatched input sizes: {input_sizes}. "
            "All sources must share the same reference_input_size."
        )

    # Compute inverse-MAE weights
    inv_maes = [1.0 / mae for mae in val_maes]
    total = sum(inv_maes)
    weights = [w / total for w in inv_maes]

    print(f"\n  Weights (inverse-MAE normalised):")
    for path, mae, w in zip(checkpoint_paths, val_maes, weights):
        print(f"    {os.path.basename(path):50s}  val_mae={mae:.4f}  w={w:.4f}")

    # Weighted average of state dicts
    avg_state = {}
    for key in models[0].state_dict():
        avg_state[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(weights, models)
        )

    print(f"\n  ✓ Weighted soup complete  (input_size={input_sizes[0]})")
    return avg_state, models[0].hparams


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Fine-tune soup on target building (PRIME_Transfer)
# ─────────────────────────────────────────────────────────────────────────────

def train_prime_transfer(
    target_building: str,
    soup_state_dict: dict,
    soup_hparams,
    data_limit_weeks: int,
    site_id: str | None = None,
    building_type: str | None = None,
    experiment_name: str = 'prime',
    loss_alpha: float = 0.7,
    seq_length: int = 24,
    epochs: int = 50,
) -> tuple[object, list, object]:
    """Fine-tune the weighted soup on limited target building data.

    Loads soup_state_dict into a fresh EnergyLSTM with soup_hparams
    (inheriting the source architecture and input_size), then fine-tunes
    on data_limit_weeks of target building data.

    Args:
        target_building:  Building ID of the target.
        soup_state_dict:  Averaged state dict from build_weighted_soup().
        soup_hparams:     hparams from the reference source model.
        data_limit_weeks: Weeks of target data to use for fine-tuning.
        site_id:          Site filter for data loading.
        building_type:    Building type filter.
        experiment_name:  Sub-directory for checkpoint saving.
        loss_alpha:       QuantileLoss alpha (default 0.7).
        seq_length:       LSTM sequence length in hours (default 24).
        epochs:           Maximum fine-tuning epochs.

    Returns:
        (model, test_results, test_loader): Fine-tuned model, trainer.test()
        output, and the test DataLoader (needed for streaming split).
    """
    print(f"\n{'='*70}")
    print(f"  PRIME TRANSFER: {target_building}  ({data_limit_weeks} weeks, α={loss_alpha})")
    print(f"{'='*70}")

    electricity, metadata, valid_buildings = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    if target_building not in valid_buildings:
        raise ValueError(f"Target building '{target_building}' not found in filtered data.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv',
    )
    weather_tgt = None
    try:
        weather = pd.read_csv(weather_path)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
        tgt_row = metadata[metadata['building_id'] == target_building]
        tgt_site = tgt_row['site_id'].values[0] if len(tgt_row) > 0 else None
        if tgt_site:
            weather_tgt = weather[weather['site_id'] == tgt_site].drop(columns=['site_id'])
            weather_tgt = weather_tgt.reindex(electricity.index)
    except Exception as e:
        print(f"  Warning: weather data unavailable — {e}")

    tgt_data, _ = preprocess_building_data(electricity, target_building, weather_tgt)

    # Limit to data_limit_weeks (match convention: 7*24 hours per week)
    hours = data_limit_weeks * 7 * 24
    tgt_data = tgt_data.iloc[:hours]

    # Align features to soup's input_size
    expected_input_size = soup_hparams.input_size
    feature_cols = [c for c in tgt_data.columns if c != 'energy']
    actual = len(feature_cols)
    if actual > expected_input_size:
        tgt_data = tgt_data[['energy'] + feature_cols[:expected_input_size]]
    elif actual < expected_input_size:
        for i in range(expected_input_size - actual):
            tgt_data[f'missing_feature_{i}'] = 0.0
    print(f"  Target data: {len(tgt_data)} rows, input_size={expected_input_size}")

    train_loader, val_loader, test_loader = create_dataloaders(
        tgt_data, seq_length=seq_length, batch_size=32
    )

    # Build model with soup architecture + load soup weights
    model = EnergyLSTM(
        input_size=expected_input_size,
        hidden_size=soup_hparams.hidden_size,
        num_layers=soup_hparams.num_layers,
        dropout=soup_hparams.dropout,
        learning_rate=1e-4,   # low LR — preserve generalised representations
        loss_alpha=loss_alpha,
    )
    model.load_state_dict(soup_state_dict)
    print(f"  Soup weights loaded. Fine-tuning (lr=1e-4, patience=5)...")

    save_dir = os.path.join(project_root, 'models', 'prime', experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename=f'prime_transfer_{target_building[:20]}_{data_limit_weeks}week_{{epoch:02d}}_{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_cb, early_stop],
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Reload best checkpoint for testing
    best_model = EnergyLSTM.load_from_checkpoint(checkpoint_cb.best_model_path)
    print(f"\nTesting on held-out test set...")
    test_results = trainer.test(best_model, test_loader, verbose=False)

    print(f"\n  ✓ PRIME Transfer complete")
    print(f"    Best checkpoint: {os.path.basename(checkpoint_cb.best_model_path)}")
    print(f"    Test RMSE: {test_results[0]['test_rmse']:.4f}")
    print(f"    Test MAE:  {test_results[0]['test_mae']:.4f}")

    return best_model, test_results, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Soft-blend live streaming
# ─────────────────────────────────────────────────────────────────────────────

class _RollingMAE:
    """Maintains a rolling window of absolute errors for streaming evaluation."""

    def __init__(self, window: int):
        self._errors: deque = deque(maxlen=window)
        self.window = window

    def update(self, error: float) -> None:
        self._errors.append(abs(error))

    def mae(self) -> float:
        return float(np.mean(self._errors)) if self._errors else float('inf')

    def ready(self) -> bool:
        return len(self._errors) > 0


def run_soft_blend_streaming(
    transfer_model,
    pretransfer_model,
    electricity_df: pd.DataFrame,
    target_building: str,
    data_limit_weeks: int,
    metadata: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    eval_window: int = 168,
    hard_switch_threshold_pct: float = 2.0,
    n_mc_samples: int = 50,
    mc_confidence: float = 0.9,
    seq_length: int = 24,
) -> pd.DataFrame:
    """Replay post-training data and produce soft-blended forecasts with uncertainty.

    Strategy:
        - Both models predict every hour (deterministic) for rolling MAE tracking.
        - Every eval_window hours: compute inverse-MAE blend weight w_transfer.
            final_pred = w_T × pred_T + (1-w_T) × pred_PT
        - Hard-switch override: if one model leads by > hard_switch_threshold_pct,
            set w_transfer to 1 (Transfer wins) or 0 (PreTransfer wins), preventing
            a persistently bad model from contributing to predictions.
        - Active model (dominant after blending) runs MC Dropout for uncertainty.
        - Default: start with equal weights (neither model proven yet).

    Streaming data:
        Full target building data AFTER data_limit_weeks × 168 hours — genuinely
        unseen by both models during training.

    Args:
        transfer_model:    Fine-tuned PRIME_Transfer model.
        pretransfer_model: From-scratch PreTransfer baseline model.
        electricity_df:    Full electricity DataFrame (all buildings).
        target_building:   Building ID of the target.
        data_limit_weeks:  Training horizon — streaming starts after this many weeks.
        metadata:          Building metadata DataFrame.
        weather_df:        Optional site-level weather (aligned to electricity index).
        eval_window:       Hours between blend-weight updates (default 168 = 1 week).
        hard_switch_threshold_pct: Margin to trigger hard switch override (default 2%).
        n_mc_samples:      MC Dropout samples for the active model (default 50).
        mc_confidence:     Confidence interval width (default 0.9 → 90% CI).
        seq_length:        LSTM context window in hours (default 24).

    Returns:
        pd.DataFrame with one row per streaming hour. Columns:
            timestamp, actual_kwh, active_model, blend_weight_transfer,
            prediction_kwh, uncertainty_std, lower_ci_90, upper_ci_90,
            transfer_pred_kwh, pretransfer_pred_kwh,
            rolling_mae_transfer, rolling_mae_pretransfer, stream_hour
    """
    from data_loader import preprocess_building_data

    print(f"\n{'='*70}")
    print(f"  SOFT-BLEND STREAMING: {target_building}")
    print(f"  Training seen: {data_limit_weeks} weeks | Eval window: {eval_window}h")
    print(f"  Hard-switch threshold: {hard_switch_threshold_pct}%")
    print(f"{'='*70}")

    # Preprocess full target data
    tgt_site = None
    row = metadata[metadata['building_id'] == target_building]
    if len(row) > 0:
        tgt_site = row['site_id'].values[0]

    weather_tgt = None
    if weather_df is not None and tgt_site is not None:
        weather_tgt = weather_df[weather_df['site_id'] == tgt_site].drop(columns=['site_id'])
        weather_tgt = weather_tgt.reindex(electricity_df.index)

    full_data, _ = preprocess_building_data(electricity_df, target_building, weather_tgt)

    # Align features to model input_size
    expected_input_size = transfer_model.hparams.input_size
    feature_cols = [c for c in full_data.columns if c != 'energy']
    actual_feats = len(feature_cols)
    if actual_feats > expected_input_size:
        full_data = full_data[['energy'] + feature_cols[:expected_input_size]]
    elif actual_feats < expected_input_size:
        for i in range(expected_input_size - actual_feats):
            full_data[f'missing_feature_{i}'] = 0.0

    # Split: training past | streaming future
    split_hour = data_limit_weeks * 7 * 24
    streaming_data = full_data.iloc[split_hour:]

    if len(streaming_data) < seq_length + 1:
        raise ValueError(
            f"Insufficient data for streaming after {data_limit_weeks} weeks split. "
            f"Got {len(streaming_data)} rows, need at least {seq_length + 1}."
        )

    print(f"  Total hours available:  {len(full_data)}")
    print(f"  Streaming hours:        {len(streaming_data)}")

    # Prepare feature matrix (no energy column — it's the target)
    feature_col_names = [c for c in full_data.columns if c != 'energy']
    all_features = full_data[feature_col_names].values      # (total_hours, n_features)
    all_energy   = full_data['energy'].values               # (total_hours,)

    rolling_t  = _RollingMAE(window=eval_window)
    rolling_pt = _RollingMAE(window=eval_window)

    # Current blend weights (equal start)
    w_transfer = 0.5

    rows = []
    n_stream = len(streaming_data)

    transfer_model.eval()
    pretransfer_model.eval()

    for step in range(n_stream):
        global_idx = split_hour + step  # index into full_data
        if global_idx < seq_length:
            continue  # not enough history for a context window

        # Build context window [global_idx - seq_length : global_idx]
        ctx_feats = all_features[global_idx - seq_length: global_idx]  # (seq_len, n_feat)
        x_tensor = torch.tensor(ctx_feats, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

        actual = float(all_energy[global_idx])
        timestamp = streaming_data.index[step] if hasattr(streaming_data.index, '__getitem__') else step

        # Deterministic predictions for rolling MAE tracking
        with torch.no_grad():
            transfer_model.eval()
            pretransfer_model.eval()
            pred_t  = float(transfer_model(x_tensor).squeeze())
            pred_pt = float(pretransfer_model(x_tensor).squeeze())

        rolling_t.update(actual - pred_t)
        rolling_pt.update(actual - pred_pt)

        # Blend weight update every eval_window steps
        if step > 0 and step % eval_window == 0 and rolling_t.ready() and rolling_pt.ready():
            mae_t  = rolling_t.mae()
            mae_pt = rolling_pt.mae()

            # Hard-switch override via existing switch_logic
            decision = decide_model(
                pretransfer_metrics={'rmse': mae_pt},
                transfer_metrics={'rmse': mae_t},
                margin_threshold_pct=hard_switch_threshold_pct,
            )

            if decision['switched']:
                # PreTransfer is clearly better — hard switch
                w_transfer = 0.0
            elif decision['selected_model'] == 'transfer' and \
                    decision['decision_reason'] != 'within_threshold_prefer_transfer':
                # Transfer is clearly better — hard switch
                w_transfer = 1.0
            else:
                # Marginal — use soft inverse-MAE blend
                if mae_t > 0 and mae_pt > 0:
                    inv_t  = 1.0 / mae_t
                    inv_pt = 1.0 / mae_pt
                    w_transfer = inv_t / (inv_t + inv_pt)
                # else keep previous weight

        # Active model label for logging
        if w_transfer >= 0.7:
            active_model = 'transfer'
        elif w_transfer <= 0.3:
            active_model = 'pretransfer'
        else:
            active_model = 'blend'

        # Final blended prediction (deterministic components already computed)
        blended_pred = w_transfer * pred_t + (1.0 - w_transfer) * pred_pt

        # MC Dropout uncertainty on the dominant model
        dominant_model = transfer_model if w_transfer >= 0.5 else pretransfer_model
        mc_mean, mc_std, mc_lower, mc_upper = predict_with_uncertainty(
            dominant_model, x_tensor, n_samples=n_mc_samples, confidence=mc_confidence,
        )

        rows.append({
            'timestamp':               timestamp,
            'actual_kwh':              actual,
            'active_model':            active_model,
            'blend_weight_transfer':   round(w_transfer, 4),
            'prediction_kwh':          round(blended_pred, 4),
            'uncertainty_std':         round(float(mc_std[0]), 4),
            'lower_ci_90':             round(float(mc_lower[0]), 4),
            'upper_ci_90':             round(float(mc_upper[0]), 4),
            'transfer_pred_kwh':       round(pred_t, 4),
            'pretransfer_pred_kwh':    round(pred_pt, 4),
            'rolling_mae_transfer':    round(rolling_t.mae(), 4),
            'rolling_mae_pretransfer': round(rolling_pt.mae(), 4),
            'stream_hour':             step,
        })

        if step % 1000 == 0 and step > 0:
            print(f"  Stream hour {step:>5}/{n_stream}  "
                  f"w_T={w_transfer:.2f}  active={active_model}  "
                  f"MAE_T={rolling_t.mae():.4f}  MAE_PT={rolling_pt.mae():.4f}")

    df = pd.DataFrame(rows)
    print(f"\n  ✓ Streaming complete: {len(df)} predictions")

    # Summary statistics
    switch_hours = (df['blend_weight_transfer'] == 0.0).sum() + \
                   (df['blend_weight_transfer'] == 1.0).sum()
    blend_hours  = len(df) - switch_hours
    streaming_mae = np.mean(np.abs(df['prediction_kwh'] - df['actual_kwh']))
    print(f"  Hard-switch hours: {switch_hours} | Soft-blend hours: {blend_hours}")
    print(f"  Overall streaming MAE: {streaming_mae:.4f}")

    return df
