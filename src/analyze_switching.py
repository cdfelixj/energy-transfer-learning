"""
Switching Pattern Analysis

Analyses the output of the switch modelling experiment to surface insights about:
  - How often switching occurs (and in which data-amount range)
  - The RMSE benefit realised by switching vs always using one model
  - The oracle performance (best possible at every week count)

Input: data_efficiency_switched.csv produced by evaluate_all_models.evaluate_data_efficiency_with_switching()
Output: switch_summary.csv (key-value statistics)
"""

import os
import math
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _mean_rmse_strategy(df: pd.DataFrame, selected_col: str) -> float:
    """Mean RMSE of the selected model across all rows with valid data."""
    rmses = []
    for _, row in df.iterrows():
        if row[selected_col] == 'transfer' and _valid(row.get('transfer_rmse')):
            rmses.append(float(row['transfer_rmse']))
        elif row[selected_col] == 'pretransfer' and _valid(row.get('pretransfer_rmse')):
            rmses.append(float(row['pretransfer_rmse']))
    return float(np.mean(rmses)) if rmses else float('nan')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_switching_patterns(switched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute switching statistics from a data_efficiency_switched DataFrame.

    Args:
        switched_df: DataFrame produced by
                     evaluate_data_efficiency_with_switching(), with columns:
                       weeks, pretransfer_mae, pretransfer_rmse, pretransfer_r2,
                       transfer_mae, transfer_rmse, transfer_r2,
                       selected_model, rmse_margin_pct, switched,
                       decision_reason, confidence

    Returns:
        DataFrame with two columns ['metric', 'value'] suitable for CSV export.
    """
    df = switched_df.copy()
    metrics = {}

    total = len(df)
    metrics['total_weeks_evaluated'] = total

    # ── Switch counts ─────────────────────────────────────────────────────
    switched = df[df['switched'] == True]
    n_switched = len(switched)
    metrics['total_switches'] = n_switched
    metrics['switch_rate_pct'] = round(n_switched / total * 100, 2) if total else 0.0

    # Weeks where switching occurred
    switched_weeks = sorted(switched['weeks'].tolist())
    metrics['switched_at_weeks'] = str(switched_weeks) if switched_weeks else '[]'

    # Breakdown by data-amount tier
    tier_low  = df[df['weeks'].isin([1, 2, 4])]
    tier_mid  = df[df['weeks'].isin([8, 16, 32])]
    tier_high = df[df['weeks'].isin([64, 104])]
    metrics['switches_in_tier_1_4_weeks']    = int(tier_low['switched'].sum())
    metrics['switches_in_tier_8_32_weeks']   = int(tier_mid['switched'].sum())
    metrics['switches_in_tier_64_104_weeks'] = int(tier_high['switched'].sum())

    # ── Decision reason breakdown ─────────────────────────────────────────
    if 'decision_reason' in df.columns:
        for reason, count in df['decision_reason'].value_counts().items():
            metrics[f'reason_{reason}'] = int(count)

    # ── RMSE benefit from switching ───────────────────────────────────────
    switch_improvements = []
    for _, row in switched.iterrows():
        pt_rmse = row.get('pretransfer_rmse')
        tr_rmse = row.get('transfer_rmse')
        if _valid(pt_rmse) and _valid(tr_rmse) and row['selected_model'] == 'pretransfer':
            improvement_pct = (float(tr_rmse) - float(pt_rmse)) / float(tr_rmse) * 100
            switch_improvements.append(improvement_pct)

    if switch_improvements:
        metrics['avg_rmse_improvement_when_switched_pct'] = round(float(np.mean(switch_improvements)), 4)
        metrics['max_rmse_improvement_when_switched_pct'] = round(float(np.max(switch_improvements)), 4)
        best_idx = switched['rmse_margin_pct'].idxmax() if not switched.empty else None
        if best_idx is not None:
            metrics['best_week_to_switch'] = int(switched.loc[best_idx, 'weeks'])
            metrics['best_switch_margin_pct'] = round(float(switched.loc[best_idx, 'rmse_margin_pct']), 4)
    else:
        metrics['avg_rmse_improvement_when_switched_pct'] = float('nan')
        metrics['max_rmse_improvement_when_switched_pct'] = float('nan')
        metrics['best_week_to_switch'] = None
        metrics['best_switch_margin_pct'] = float('nan')

    # ── Oracle RMSE (always pick the best available model) ───────────────
    oracle_rmses = []
    for _, row in df.iterrows():
        pt = row.get('pretransfer_rmse')
        tr = row.get('transfer_rmse')
        candidates = [v for v in [pt, tr] if _valid(v)]
        if candidates:
            oracle_rmses.append(min(float(v) for v in candidates))
    metrics['oracle_mean_rmse'] = round(float(np.mean(oracle_rmses)), 4) if oracle_rmses else float('nan')

    # ── Strategy mean RMSEs ───────────────────────────────────────────────
    pt_rmses = [float(v) for v in df['pretransfer_rmse'].dropna() if _valid(v)]
    tr_rmses = [float(v) for v in df['transfer_rmse'].dropna()    if _valid(v)]
    sw_rmse  = _mean_rmse_strategy(df, 'selected_model')

    metrics['always_pretransfer_mean_rmse'] = round(float(np.mean(pt_rmses)), 4) if pt_rmses else float('nan')
    metrics['always_transfer_mean_rmse']    = round(float(np.mean(tr_rmses)), 4) if tr_rmses else float('nan')
    metrics['switched_strategy_mean_rmse']  = round(sw_rmse, 4) if _valid(sw_rmse) else float('nan')

    # RMSE delta: switch strategy vs always Transfer
    if _valid(metrics['switched_strategy_mean_rmse']) and _valid(metrics['always_transfer_mean_rmse']):
        delta = metrics['always_transfer_mean_rmse'] - metrics['switched_strategy_mean_rmse']
        metrics['switched_vs_always_transfer_rmse_delta'] = round(delta, 4)
        pct = delta / metrics['always_transfer_mean_rmse'] * 100
        metrics['switched_vs_always_transfer_rmse_delta_pct'] = round(pct, 4)
    else:
        metrics['switched_vs_always_transfer_rmse_delta'] = float('nan')
        metrics['switched_vs_always_transfer_rmse_delta_pct'] = float('nan')

    # ── Confidence distribution ───────────────────────────────────────────
    if 'confidence' in df.columns:
        for conf, count in df['confidence'].value_counts().items():
            metrics[f'confidence_{conf}'] = int(count)

    # ── Return as tidy key-value DataFrame ────────────────────────────────
    summary_df = pd.DataFrame([
        {'metric': k, 'value': v} for k, v in metrics.items()
    ])
    return summary_df


def print_switching_report(switched_df: pd.DataFrame, experiment_name: str = '') -> None:
    """Print a human-readable summary of switching decisions to stdout."""
    summary = analyze_switching_patterns(switched_df)
    kv = dict(zip(summary['metric'], summary['value']))

    header = f'  SWITCH MODELLING REPORT{f": {experiment_name}" if experiment_name else ""}'
    print(f"\n{'=' * 90}")
    print(header)
    print(f"{'=' * 90}")

    total = kv.get('total_weeks_evaluated', '?')
    n_sw  = kv.get('total_switches', '?')
    rate  = kv.get('switch_rate_pct', '?')
    print(f"\n  Switch rate      : {n_sw} / {total} week counts ({rate}%)")
    print(f"  Switched at weeks: {kv.get('switched_at_weeks', '[]')}")

    print(f"\n  Decision reasons:")
    for reason in ('transfer_better', 'pretransfer_better',
                   'within_threshold_prefer_transfer',
                   'only_transfer_available', 'only_pretransfer_available', 'no_data'):
        count = kv.get(f'reason_{reason}', 0)
        print(f"    {reason:<40} {count}")

    print(f"\n  RMSE summary (mean across all weeks):")
    print(f"    Always Transfer    : {kv.get('always_transfer_mean_rmse', 'N/A')}")
    print(f"    Always PreTransfer : {kv.get('always_pretransfer_mean_rmse', 'N/A')}")
    print(f"    Switched strategy  : {kv.get('switched_strategy_mean_rmse', 'N/A')}")
    print(f"    Oracle (best each) : {kv.get('oracle_mean_rmse', 'N/A')}")

    delta     = kv.get('switched_vs_always_transfer_rmse_delta', float('nan'))
    delta_pct = kv.get('switched_vs_always_transfer_rmse_delta_pct', float('nan'))
    if _valid(delta):
        print(f"\n  Switched vs always-Transfer : {delta:+.4f} RMSE ({delta_pct:+.2f}%)")
    else:
        print(f"\n  Switched vs always-Transfer : N/A")

    avg_imp = kv.get('avg_rmse_improvement_when_switched_pct', float('nan'))
    if _valid(avg_imp):
        print(f"  Avg RMSE improvement on switch events : {avg_imp:.2f}%")

    print(f"\n  Tier breakdown (switches):")
    print(f"    Weeks  1–4   : {kv.get('switches_in_tier_1_4_weeks',   0)}")
    print(f"    Weeks  8–32  : {kv.get('switches_in_tier_8_32_weeks',  0)}")
    print(f"    Weeks 64–104 : {kv.get('switches_in_tier_64_104_weeks',0)}")
    print(f"\n{'=' * 90}")


def load_and_analyse(csv_path: str, experiment_name: str = '') -> pd.DataFrame:
    """
    Convenience wrapper: load a switched CSV, print the report, return summary.

    Args:
        csv_path: Path to data_efficiency_switched.csv
        experiment_name: Optional label for the report header.

    Returns:
        Summary DataFrame from analyze_switching_patterns().
    """
    df = pd.read_csv(csv_path)
    print_switching_report(df, experiment_name)
    return analyze_switching_patterns(df)
