"""
PRIME Experiment — Full End-to-End Pipeline

PRIME (Performance-weighted Robust Initialisation for Modelling Energy) is the
recommended production strategy that synthesises all 12 transfer learning
experiments into a single, automated pipeline:

  Discovery → Performance-Weighted Multi-Source Training → Evaluation
           → Soft-Blend Live Deployment with Uncertainty

What PRIME does better than any individual experiment:
  ✓ Multi-source (N=5) initialisation eliminates single-source collapse (Exp 7)
  ✓ Performance-weighted soup > uniform averaging (extends Exp 9)
  ✓ QuantileLoss(α=0.7) penalises underprediction 2.3× (asymmetric provisioning)
  ✓ Automatic multi-factor source ranking: completeness + type + site + profile
  ✓ Soft-blend streaming with MC Dropout uncertainty (production-ready output)
  ✓ Hard-switch override when one model is clearly dominant

Usage
-----
  python scripts/run_prime_experiment.py
  python scripts/run_prime_experiment.py --target-building Rat_education_Denise
  python scripts/run_prime_experiment.py --no-sweep --weeks 8 --skip-streaming

Outputs (results/prime/{target_building}_{weeks}week/)
------------------------------------------------------
  source_rankings.csv         — Ranked source buildings with composite scores
  source_weights.csv          — Val MAEs and inverse-MAE weights used in soup
  evaluation_comparison.csv   — PRIME_Transfer vs PreTransfer test metrics
  data_efficiency_prime.csv   — PRIME_Transfer metrics across all weeks sweeps
  live_inference.csv          — Per-hour streaming predictions + uncertainty
  figures/
    data_efficiency.png       — PRIME vs PreTransfer data efficiency curve
    blend_weights.png         — Blend weight w_transfer over streaming time
    uncertainty_bands.png     — Predictions ± 90% CI over streaming horizon
"""

import argparse
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import numpy as np
import pandas as pd

from src.train_prime import (
    train_prime_source,
    build_weighted_soup,
    train_prime_transfer,
    run_soft_blend_streaming,
)
from src.train_pretransfer import train_pretransfer

from scripts.discover_buildings import (
    compute_completeness,
    compute_building_profile,
    score_source_candidates,
    select_sources,
)
from src.data_loader import load_electricity_data


WEEKS_LIST = [1, 2, 4, 8, 16, 32, 64, 104]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _out_dir(project_root: str, target_building: str, weeks: int | None = None) -> str:
    suffix = f'_{weeks}week' if weeks is not None else '_sweep'
    d = os.path.join(project_root, 'results', 'prime', f'{target_building}{suffix}')
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, 'figures'), exist_ok=True)
    return d


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"  ✓ Saved: {os.path.relpath(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Source Discovery
# ─────────────────────────────────────────────────────────────────────────────

def phase_discovery(
    target_building: str,
    site_id: str | None,
    building_type: str | None,
    n_sources: int,
    out_dir: str,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Rank source buildings and save source_rankings.csv.

    Returns:
        (source_list, electricity_df, metadata_df)
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 1 — SOURCE DISCOVERY")
    print(f"  Target: {target_building}  |  N sources: {n_sources}")
    print(f"{'='*70}")

    # Always load ALL buildings for source ranking — PRIME's value comes from
    # cross-site diversity; restricting to the target's site/type would defeat that.
    electricity, metadata, valid = load_electricity_data(site_id=None, building_type=None)
    completeness = compute_completeness(electricity, min_weeks=16)

    if target_building not in completeness.index:
        raise ValueError(
            f"Target '{target_building}' does not meet minimum data quality threshold.\n"
            f"Available buildings: {completeness.index.tolist()}"
        )

    # Rank all remaining buildings as candidates
    candidates = [b for b in completeness.index if b != target_building]
    if not candidates:
        raise ValueError(f"No candidate source buildings found after excluding {target_building}.")

    scores = score_source_candidates(electricity, target_building, candidates, metadata, completeness)
    prime_sources = scores.index.tolist()[:n_sources]

    print(f"\n  Top-{n_sources} sources (multi-factor ranked):")
    for i, bid in enumerate(prime_sources, 1):
        tgt_row = metadata[metadata['building_id'] == target_building]
        src_row = metadata[metadata['building_id'] == bid]
        t_type = tgt_row['primaryspaceusage'].values[0] if len(tgt_row) else '?'
        s_type = src_row['primaryspaceusage'].values[0] if len(src_row) else '?'
        t_site = tgt_row['site_id'].values[0] if len(tgt_row) else '?'
        s_site = src_row['site_id'].values[0] if len(src_row) else '?'
        match_str = []
        if s_type == t_type:
            match_str.append('type✓')
        if s_site == t_site:
            match_str.append('site✓')
        print(f"    {i}. {bid}  score={scores[bid]:.1f}  completeness={completeness.get(bid, 0):.1f}%"
              f"  {' '.join(match_str)}")

    # Save source rankings
    ranking_rows = []
    for bid in scores.index:
        src_row = metadata[metadata['building_id'] == bid]
        ranking_rows.append({
            'rank':                len(ranking_rows) + 1,
            'building_id':         bid,
            'composite_score':     round(scores[bid], 2),
            'completeness_pct':    round(float(completeness.get(bid, 0)), 2),
            'site_id':             src_row['site_id'].values[0] if len(src_row) else '',
            'building_type':       src_row['primaryspaceusage'].values[0] if len(src_row) else '',
            'selected_as_source':  bid in prime_sources,
        })
    _save_csv(pd.DataFrame(ranking_rows), os.path.join(out_dir, 'source_rankings.csv'))

    return prime_sources, electricity, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Data efficiency sweep (sweep mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(
    args: argparse.Namespace,
    experiment_name: str,
    soup_state: dict,
    soup_hparams,
    out_dir: str,
    electricity_all: pd.DataFrame | None = None,
    metadata_all: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, object | None, object | None]:
    """Train PRIME_Transfer + PreTransfer at every weeks value in WEEKS_LIST.

    The soup is built ONCE (source models are trained on full 2-year data,
    independent of target data limit). Only the fine-tuning phase is repeated.
    This is significantly more efficient than running the experiment N times.

    Returns:
        (sweep_df, streaming_model, streaming_pt_model)
        streaming_model/pt_model are the models at args.weeks (for Phase 5).
    """
    efficiency_rows = []
    streaming_model    = None   # saved for streaming phase
    streaming_pt_model = None

    for weeks in WEEKS_LIST:
        print(f"\n{'─'*70}")
        print(f"  SWEEP: {weeks} week{'s' if weeks > 1 else ''}")
        print(f"{'─'*70}")

        # PRIME_Transfer
        try:
            prime_model, prime_results, _ = train_prime_transfer(
                target_building=args.target_building,
                soup_state_dict=soup_state,
                soup_hparams=soup_hparams,
                data_limit_weeks=weeks,
                site_id=args.site_id,
                building_type=args.building_type,
                experiment_name=f"{experiment_name}_{weeks}week",
                loss_alpha=args.loss_alpha,
                seq_length=24,
                epochs=50,
            )
            prime_r = prime_results[0] if prime_results else {}
        except Exception as e:
            print(f"  ✗ PRIME_Transfer ({weeks} weeks) failed: {e}")
            prime_model, prime_r = None, {}

        # PreTransfer baseline
        try:
            pt_model, pt_results = train_pretransfer(
                target_building=args.target_building,
                epochs=100,
                seq_length=24,
                data_limit_weeks=weeks,
                site_id=args.site_id,
                building_type=args.building_type,
                experiment_name=f"{experiment_name}_{weeks}week",
            )
            pt_r = pt_results[0] if pt_results else {}
        except Exception as e:
            print(f"  ✗ PreTransfer ({weeks} weeks) failed: {e}")
            pt_model, pt_r = None, {}

        # Save models for the streaming phase (at the user's chosen --weeks)
        if weeks == args.weeks:
            streaming_model    = prime_model
            streaming_pt_model = pt_model

        # PRIME Streaming — run soft-blend at this week level
        streaming_mae_val  = float('nan')
        streaming_rmse_val = float('nan')
        if (not args.skip_streaming
                and prime_model is not None
                and pt_model is not None
                and electricity_all is not None
                and metadata_all is not None):
            try:
                stream_out_dir = os.path.join(out_dir, 'streaming')
                os.makedirs(stream_out_dir, exist_ok=True)
                live_df = run_soft_blend_streaming(
                    transfer_model=prime_model,
                    pretransfer_model=pt_model,
                    electricity_df=electricity_all,
                    target_building=args.target_building,
                    data_limit_weeks=weeks,
                    metadata=metadata_all,
                    weather_df=weather_df,
                    eval_window=args.eval_window,
                    hard_switch_threshold_pct=args.threshold,
                    n_mc_samples=args.mc_samples,
                    mc_confidence=0.9,
                    seq_length=24,
                )
                _save_csv(live_df, os.path.join(stream_out_dir, f'live_inference_{weeks}week.csv'))
                errors = live_df['prediction_kwh'] - live_df['actual_kwh']
                streaming_mae_val  = round(float(errors.abs().mean()), 4)
                streaming_rmse_val = round(float((errors ** 2).mean() ** 0.5), 4)
                print(f"  \u2713 Streaming {weeks}w: MAE={streaming_mae_val:.4f}  RMSE={streaming_rmse_val:.4f}")
            except Exception as e:
                print(f"  \u2717 Streaming ({weeks} weeks) skipped: {e}")

        efficiency_rows.append({
            'weeks':                weeks,
            'prime_rmse':           round(prime_r.get('test_rmse', float('nan')), 4),
            'prime_mae':            round(prime_r.get('test_mae',  float('nan')), 4),
            'pretransfer_rmse':     round(pt_r.get('test_rmse',    float('nan')), 4),
            'pretransfer_mae':      round(pt_r.get('test_mae',     float('nan')), 4),
            'prime_streaming_rmse': streaming_rmse_val,
            'prime_streaming_mae':  streaming_mae_val,
        })

    df = pd.DataFrame(efficiency_rows)

    # Save combined sweep CSV
    _save_csv(df, os.path.join(out_dir, 'data_efficiency_prime.csv'))

    # Save separate per-model CSVs matching existing experiment structure
    # (columns: weeks, rmse, mae — same as data_efficiency_transfer.csv etc.)
    _save_csv(
        df[['weeks', 'prime_rmse', 'prime_mae']].rename(
            columns={'prime_rmse': 'rmse', 'prime_mae': 'mae'}),
        os.path.join(out_dir, 'data_efficiency_prime_transfer.csv'),
    )
    _save_csv(
        df[['weeks', 'pretransfer_rmse', 'pretransfer_mae']].rename(
            columns={'pretransfer_rmse': 'rmse', 'pretransfer_mae': 'mae'}),
        os.path.join(out_dir, 'data_efficiency_prime_pretransfer.csv'),
    )
    if 'prime_streaming_rmse' in df.columns:
        _save_csv(
            df[['weeks', 'prime_streaming_rmse', 'prime_streaming_mae']].rename(
                columns={'prime_streaming_rmse': 'rmse', 'prime_streaming_mae': 'mae'}),
            os.path.join(out_dir, 'data_efficiency_prime_streaming.csv'),
        )

    # Generate efficiency curve figure
    _generate_efficiency_figure(df, out_dir)

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — Data Efficiency Results")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    return df, streaming_model, streaming_pt_model


def _generate_efficiency_figure(df: pd.DataFrame, out_dir: str) -> None:
    """Generate PRIME vs PreTransfer data efficiency curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, label in [
        (axes[0], 'rmse', 'RMSE (kWh)'),
        (axes[1], 'mae',  'MAE (kWh)'),
    ]:
        prime_col = f'prime_{metric}'
        pt_col    = f'pretransfer_{metric}'
        if prime_col in df.columns:
            ax.plot(df['weeks'], df[prime_col],   'o-', color='steelblue',
                    label='PRIME_Transfer', linewidth=1.8, markersize=5)
        if pt_col in df.columns:
            ax.plot(df['weeks'], df[pt_col], 's--', color='darkorange',
                    label='PreTransfer', linewidth=1.5, markersize=5)
        stream_col = f'prime_streaming_{metric}'
        if stream_col in df.columns and df[stream_col].notna().any():
            ax.plot(df['weeks'], df[stream_col], '^-.', color='seagreen',
                    label='PRIME_Streaming', linewidth=1.5, markersize=5)
        ax.set_xscale('log')
        ax.set_xticks(df['weeks'])
        ax.set_xticklabels(df['weeks'])
        ax.set_xlabel('Training weeks')
        ax.set_ylabel(label)
        ax.set_title(f'PRIME Data Efficiency — {label}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('PRIME: Performance-Weighted Multi-Source Transfer vs PreTransfer',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figures', 'data_efficiency.png'), dpi=120)
    plt.close(fig)
    print("  ✓ Saved: figures/data_efficiency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Build performance-weighted soup
# ─────────────────────────────────────────────────────────────────────────────

def phase_soup(
    source_list: list[str],
    electricity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    site_id: str | None,
    building_type: str | None,
    experiment_name: str,
    loss_alpha: float,
    out_dir: str,
) -> tuple[dict, object]:
    """Train N source models and build the weighted soup.

    Returns:
        (soup_state_dict, soup_hparams)
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 2 — PERFORMANCE-WEIGHTED SOUP  (N={len(source_list)})")
    print(f"{'='*70}")

    checkpoint_paths = []
    val_maes = []
    reference_input_size = None   # set from first model; all subsequent models align to it

    for i, source in enumerate(source_list):
        print(f"\n  Training source {i+1}/{len(source_list)}: {source}")
        ckpt_path, val_mae = train_prime_source(
            source_building=source,
            reference_input_size=reference_input_size,
            epochs=50,
            seq_length=168,
            site_id=None,          # search all sites for source buildings
            building_type=None,    # search all types for source buildings
            experiment_name=experiment_name,
            loss_alpha=loss_alpha,
        )
        checkpoint_paths.append(ckpt_path)
        val_maes.append(val_mae)

        if reference_input_size is None:
            # Fix input_size from first source; subsequent sources align to this
            from src.models import EnergyLSTM
            m = EnergyLSTM.load_from_checkpoint(ckpt_path)
            reference_input_size = m.hparams.input_size
            print(f"  Reference input_size fixed at: {reference_input_size}")

    # Build weighted soup
    soup_state, soup_hparams = build_weighted_soup(checkpoint_paths, val_maes)

    # Save source weights report
    weights_rows = []
    total_inv = sum(1.0 / m for m in val_maes)
    for src, ckpt, mae in zip(source_list, checkpoint_paths, val_maes):
        weights_rows.append({
            'source_building':  src,
            'checkpoint':       os.path.basename(ckpt),
            'val_mae':          round(mae, 4),
            'soup_weight':      round((1.0 / mae) / total_inv, 4),
        })
    _save_csv(pd.DataFrame(weights_rows), os.path.join(out_dir, 'source_weights.csv'))

    return soup_state, soup_hparams


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — PRIME_Transfer + PreTransfer training
# ─────────────────────────────────────────────────────────────────────────────

def phase_training(
    target_building: str,
    soup_state: dict,
    soup_hparams,
    site_id: str | None,
    building_type: str | None,
    data_limit_weeks: int,
    experiment_name: str,
    loss_alpha: float,
) -> tuple[object, object, list, list]:
    """Fine-tune soup (PRIME_Transfer) and train PreTransfer baseline.

    Returns:
        (prime_model, pretransfer_model, prime_test_results, pt_test_results)
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 3 — FINE-TUNING  ({data_limit_weeks} weeks target data)")
    print(f"{'='*70}")

    # PRIME_Transfer (soup → fine-tune)
    prime_model, prime_results, _ = train_prime_transfer(
        target_building=target_building,
        soup_state_dict=soup_state,
        soup_hparams=soup_hparams,
        data_limit_weeks=data_limit_weeks,
        site_id=site_id,
        building_type=building_type,
        experiment_name=experiment_name,
        loss_alpha=loss_alpha,
        seq_length=24,
        epochs=50,
    )

    # PreTransfer baseline (from scratch)
    print(f"\n  Training PreTransfer baseline...")
    pretransfer_model, pt_results = train_pretransfer(
        target_building=target_building,
        epochs=100,
        seq_length=24,
        data_limit_weeks=data_limit_weeks,
        site_id=site_id,
        building_type=building_type,
        experiment_name=experiment_name,
    )

    return prime_model, pretransfer_model, prime_results, pt_results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def phase_evaluation(
    prime_results: list,
    pt_results: list,
    data_limit_weeks: int,
    out_dir: str,
) -> None:
    """Save evaluation_comparison.csv comparing PRIME_Transfer vs PreTransfer."""

    print(f"\n{'='*70}")
    print(f"  PHASE 4 — EVALUATION COMPARISON")
    print(f"{'='*70}")

    rows = []
    for label, results in [('PRIME_Transfer', prime_results), ('PreTransfer', pt_results)]:
        if results:
            r = results[0]
            rows.append({
                'model':            label,
                'data_limit_weeks': data_limit_weeks,
                'test_rmse':        round(r.get('test_rmse', float('nan')), 4),
                'test_mae':         round(r.get('test_mae',  float('nan')), 4),
                'test_loss':        round(r.get('test_loss', float('nan')), 4),
            })

    df = pd.DataFrame(rows)
    _save_csv(df, os.path.join(out_dir, 'evaluation_comparison.csv'))

    print(f"\n  Results at {data_limit_weeks} weeks of target data:")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Live streaming
# ─────────────────────────────────────────────────────────────────────────────

def phase_streaming(
    prime_model,
    pretransfer_model,
    target_building: str,
    site_id: str | None,
    building_type: str | None,
    data_limit_weeks: int,
    eval_window: int,
    threshold_pct: float,
    mc_samples: int,
    mc_confidence: float,
    out_dir: str,
) -> None:
    """Run soft-blend streaming and save live_inference.csv + figures."""

    print(f"\n{'='*70}")
    print(f"  PHASE 5 — SOFT-BLEND LIVE STREAMING")
    print(f"{'='*70}")

    electricity, metadata, _ = load_electricity_data(
        site_id=site_id, building_type=building_type
    )

    project_root = _root
    weather_df = None
    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv',
    )
    try:
        weather_df = pd.read_csv(weather_path)
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        weather_df = weather_df.set_index('timestamp')
    except Exception as e:
        print(f"  Warning: weather unavailable — {e}")

    # Need full electricity data for streaming (not filtered by site/type)
    electricity_all, metadata_all, _ = load_electricity_data(site_id=None, building_type=None)

    live_df = run_soft_blend_streaming(
        transfer_model=prime_model,
        pretransfer_model=pretransfer_model,
        electricity_df=electricity_all,
        target_building=target_building,
        data_limit_weeks=data_limit_weeks,
        metadata=metadata_all,
        weather_df=weather_df,
        eval_window=eval_window,
        hard_switch_threshold_pct=threshold_pct,
        n_mc_samples=mc_samples,
        mc_confidence=mc_confidence,
        seq_length=24,
    )

    _save_csv(live_df, os.path.join(out_dir, 'live_inference.csv'))

    # Generate figures
    _generate_figures(live_df, out_dir)


def _generate_figures(live_df: pd.DataFrame, out_dir: str) -> None:
    """Generate streaming analysis figures (optional — requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping figures (matplotlib not available).")
        return

    fig_dir = os.path.join(out_dir, 'figures')

    # Figure 1 — Blend weight over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(live_df['stream_hour'], live_df['blend_weight_transfer'],
            color='steelblue', linewidth=0.8)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.5, label='Equal blend')
    ax.axhline(1.0, color='green', linestyle=':', linewidth=0.5, label='Hard: Transfer')
    ax.axhline(0.0, color='red',   linestyle=':', linewidth=0.5, label='Hard: PreTransfer')
    ax.set_xlabel('Streaming hour')
    ax.set_ylabel('Blend weight (Transfer)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('PRIME: Soft-Blend Transfer Weight Over Streaming Horizon')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'blend_weights.png'), dpi=120)
    plt.close(fig)
    print("  ✓ Saved: figures/blend_weights.png")

    # Figure 2 — Predictions with uncertainty bands (first 720 hours = 30 days)
    sample = live_df.head(720)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(sample['stream_hour'], sample['lower_ci_90'], sample['upper_ci_90'],
                    alpha=0.25, color='steelblue', label='90% CI')
    ax.plot(sample['stream_hour'], sample['prediction_kwh'],
            color='steelblue', linewidth=1.0, label='PRIME prediction')
    ax.plot(sample['stream_hour'], sample['actual_kwh'],
            color='darkorange', linewidth=0.8, linestyle='--', label='Actual', alpha=0.85)
    ax.set_xlabel('Streaming hour')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('PRIME: Predictions with 90% Confidence Interval (first 30 streaming days)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'uncertainty_bands.png'), dpi=120)
    plt.close(fig)
    print("  ✓ Saved: figures/uncertainty_bands.png")

    # Figure 3 — Rolling MAE comparison
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(live_df['stream_hour'], live_df['rolling_mae_transfer'],
            color='steelblue', label='PRIME_Transfer', linewidth=0.9)
    ax.plot(live_df['stream_hour'], live_df['rolling_mae_pretransfer'],
            color='darkorange', label='PreTransfer', linewidth=0.9, linestyle='--')
    ax.set_xlabel('Streaming hour')
    ax.set_ylabel('Rolling MAE (kWh)')
    ax.set_title('PRIME: Rolling MAE comparison over Streaming Horizon')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'rolling_mae.png'), dpi=120)
    plt.close(fig)
    print("  ✓ Saved: figures/rolling_mae.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PRIME Experiment — Full end-to-end transfer learning pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--target-building', default='Eagle_education_Brooke',
                        help='Building ID of the target. Defaults to Eagle_education_Brooke '
                             '(the cross-site collapse case PRIME is designed to solve).')
    parser.add_argument('--site-id', default=None,
                        help='Site filter for data loading (e.g. Rat). None = any site.')
    parser.add_argument('--building-type', default=None,
                        help='Building type filter (e.g. Education). None = any type.')
    parser.add_argument('--weeks', type=int, default=8,
                        help='Weeks of target data for fine-tuning (1–104).')
    parser.add_argument('--n-sources', type=int, default=5,
                        help='Number of source buildings for the soup (ablation N=5 optimal).')
    parser.add_argument('--eval-window', type=int, default=168,
                        help='Hours between blend-weight updates in streaming (168 = 1 week).')
    parser.add_argument('--threshold', type=float, default=20.0,
                        help='Hard-switch margin threshold in percent (default 20.0%).')
    parser.add_argument('--mc-samples', type=int, default=50,
                        help='MC Dropout samples for uncertainty estimation.')
    parser.add_argument('--loss-alpha', type=float, default=0.7,
                        help='QuantileLoss alpha (0.7 = penalise underprediction 2.3×).')
    parser.add_argument('--no-sweep', action='store_true',
                        help='Skip the data efficiency sweep and train at --weeks only. '
                             'Useful for quick development runs.')
    parser.add_argument('--skip-streaming', action='store_true',
                        help='Skip Phase 5 (live streaming) — saves time during development.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = _root
    experiment_name = f"prime_{args.target_building[:20]}"
    sweep_mode = not args.no_sweep

    out_dir = _out_dir(project_root, args.target_building,
                       weeks=None if sweep_mode else args.weeks)

    print(f"\n{'='*70}")
    print(f"  PRIME EXPERIMENT{'  [SWEEP]' if sweep_mode else '  [SINGLE WEEK]'}")
    print(f"  Target:      {args.target_building}")
    print(f"  Mode:        {'Full sweep [1,2,4,8,16,32,64,104] weeks' if sweep_mode else f'{args.weeks} weeks'}")
    print(f"  N sources:   {args.n_sources}")
    print(f"  Loss alpha:  {args.loss_alpha}  (underprediction = {args.loss_alpha/(1-args.loss_alpha):.1f}× penalty)")
    print(f"  Output:      {os.path.relpath(out_dir)}")
    print(f"{'='*70}")

    # ── Phase 1: Discovery (always — built once) ──────────────────────────
    source_list, electricity, metadata = phase_discovery(
        target_building=args.target_building,
        site_id=args.site_id,
        building_type=args.building_type,
        n_sources=args.n_sources,
        out_dir=out_dir,
    )

    # ── Phase 2: Weighted soup (always — built once) ──────────────────────
    soup_state, soup_hparams = phase_soup(
        source_list=source_list,
        electricity_df=electricity,
        metadata_df=metadata,
        site_id=args.site_id,
        building_type=args.building_type,
        experiment_name=experiment_name,
        loss_alpha=args.loss_alpha,
        out_dir=out_dir,
    )

    # ── Load full electricity + weather for streaming (hoisted — needed by sweep) ──
    # phase_discovery already loaded all buildings; reuse those variables.
    electricity_all = electricity
    metadata_all = metadata
    project_root = _root
    weather_df = None
    weather_path = os.path.join(
        project_root, 'data', 'raw', 'building-data-genome-project-2',
        'data', 'weather', 'weather.csv',
    )
    try:
        weather_df = pd.read_csv(weather_path)
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        weather_df = weather_df.set_index('timestamp')
        print("  ✓ Weather data loaded")
    except Exception as e:
        print(f"  Warning: weather unavailable — {e}")

    if sweep_mode:
        # ── Sweep: fine-tune + evaluate at all weeks ──────────────────────
        _, prime_model, pretransfer_model = run_data_efficiency_sweep(
            args=args,
            experiment_name=experiment_name,
            soup_state=soup_state,
            soup_hparams=soup_hparams,
            out_dir=out_dir,
            electricity_all=electricity_all,
            metadata_all=metadata_all,
            weather_df=weather_df,
        )
        # Snapshot evaluation at args.weeks for the comparison CSV
        if prime_model is not None or pretransfer_model is not None:
            sweep_df = pd.read_csv(os.path.join(out_dir, 'data_efficiency_prime.csv'))
            matched = sweep_df[sweep_df['weeks'] == args.weeks]
            if len(matched) > 0:
                row = matched.iloc[0]
                snap = pd.DataFrame([
                    {'model': 'PRIME_Transfer', 'data_limit_weeks': args.weeks,
                     'test_rmse': row['prime_rmse'], 'test_mae': row['prime_mae']},
                    {'model': 'PreTransfer', 'data_limit_weeks': args.weeks,
                     'test_rmse': row['pretransfer_rmse'], 'test_mae': row['pretransfer_mae']},
                ])
                _save_csv(snap, os.path.join(out_dir, 'evaluation_comparison.csv'))
    else:
        # ── Single run ────────────────────────────────────────────────────
        prime_model, pretransfer_model, prime_results, pt_results = phase_training(
            target_building=args.target_building,
            soup_state=soup_state,
            soup_hparams=soup_hparams,
            site_id=args.site_id,
            building_type=args.building_type,
            data_limit_weeks=args.weeks,
            experiment_name=experiment_name,
            loss_alpha=args.loss_alpha,
        )
        phase_evaluation(
            prime_results=prime_results,
            pt_results=pt_results,
            data_limit_weeks=args.weeks,
            out_dir=out_dir,
        )

    # ── Phase 5: Streaming (both modes — uses args.weeks model) ──────────
    if not args.skip_streaming and prime_model is not None and pretransfer_model is not None:
        phase_streaming(
            prime_model=prime_model,
            pretransfer_model=pretransfer_model,
            target_building=args.target_building,
            site_id=args.site_id,
            building_type=args.building_type,
            data_limit_weeks=args.weeks,
            eval_window=args.eval_window,
            threshold_pct=args.threshold,
            mc_samples=args.mc_samples,
            mc_confidence=0.9,
            out_dir=out_dir,
        )
    elif args.skip_streaming:
        print("\n  [Streaming skipped via --skip-streaming]")
    else:
        print("\n  [Streaming skipped — model not available for streaming]")

    print(f"\n\n{'='*70}")
    print(f"  PRIME {'SWEEP ' if sweep_mode else ''}COMPLETE")
    print(f"  Output: {os.path.relpath(out_dir)}")
    for f in sorted(os.listdir(out_dir)):
        if os.path.isfile(os.path.join(out_dir, f)):
            print(f"    {f}")
    fig_dir = os.path.join(out_dir, 'figures')
    if os.path.isdir(fig_dir):
        for f in sorted(os.listdir(fig_dir)):
            print(f"    figures/{f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
