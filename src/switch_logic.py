"""
Switch Logic: Model Selection Decision Engine

Decides whether to select the Transfer or PreTransfer model for each week count
based on RMSE performance. Switching occurs when PreTransfer is meaningfully
better (i.e. RMSE improvement exceeds the significance threshold).

Decision rules:
  1. Both models valid: select the one with lower RMSE.
       - If margin >= threshold: select better model, mark switched=True when
         PreTransfer wins (because Transfer is the preferred default).
       - If margin < threshold: prefer Transfer (warm-start bias; assume
         Transfer generalises better when models are nearly equal).
  2. One model missing (NaN RMSE): select the available one.
  3. Both models missing: mark selected_model='no_data'.

Threshold: 2.0% (default). A difference of less than 2% is treated as
statistically insignificant given variability in limited-data training.
"""

from typing import Dict, Any
import math


# ---------------------------------------------------------------------------
# Reason tokens
# ---------------------------------------------------------------------------
REASON_TRANSFER_BETTER         = 'transfer_better'
REASON_PRETRANSFER_BETTER      = 'pretransfer_better'
REASON_WITHIN_THRESHOLD        = 'within_threshold_prefer_transfer'
REASON_ONLY_TRANSFER           = 'only_transfer_available'
REASON_ONLY_PRETRANSFER        = 'only_pretransfer_available'
REASON_NO_DATA                 = 'no_data'


def _is_valid(value: float) -> bool:
    """Return True if value is a finite non-NaN float."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def decide_model(
    pretransfer_metrics: Dict[str, Any],
    transfer_metrics: Dict[str, Any],
    margin_threshold_pct: float = 2.0,
) -> Dict[str, Any]:
    """
    Decide which model to select based on RMSE comparison.

    Args:
        pretransfer_metrics: Dict with at least 'rmse', 'mae', 'r2' keys.
        transfer_metrics: Dict with at least 'rmse', 'mae', 'r2' keys.
        margin_threshold_pct: Minimum RMSE improvement (%) required to switch
                              away from the Transfer default. Default: 2.0%.

    Returns:
        Dict with keys:
            selected_model    (str)   : 'transfer' | 'pretransfer' | 'no_data'
            rmse_margin_pct   (float) : (transfer_rmse - pretransfer_rmse) /
                                        pretransfer_rmse * 100. Positive means
                                        Transfer is worse; negative means Transfer
                                        is better.
            switched          (bool)  : True when PreTransfer is selected over
                                        the default Transfer.
            decision_reason   (str)   : Human-readable reason token.
            confidence        (str)   : 'high' | 'low' | 'none'
    """
    pt_rmse = pretransfer_metrics.get('rmse')
    tr_rmse = transfer_metrics.get('rmse')

    pt_valid = _is_valid(pt_rmse)
    tr_valid = _is_valid(tr_rmse)

    # ── Edge cases: missing data ──────────────────────────────────────────
    if not pt_valid and not tr_valid:
        return {
            'selected_model':  'no_data',
            'rmse_margin_pct': float('nan'),
            'switched':        False,
            'decision_reason': REASON_NO_DATA,
            'confidence':      'none',
        }

    if not tr_valid:
        return {
            'selected_model':  'pretransfer',
            'rmse_margin_pct': float('nan'),
            'switched':        True,
            'decision_reason': REASON_ONLY_PRETRANSFER,
            'confidence':      'none',
        }

    if not pt_valid:
        return {
            'selected_model':  'transfer',
            'rmse_margin_pct': float('nan'),
            'switched':        False,
            'decision_reason': REASON_ONLY_TRANSFER,
            'confidence':      'none',
        }

    # ── Both valid: compute margin ────────────────────────────────────────
    # Positive margin  → Transfer worse than PreTransfer (switch candidate)
    # Negative margin  → Transfer better than PreTransfer (keep Transfer)
    pt_rmse = float(pt_rmse)
    tr_rmse = float(tr_rmse)
    rmse_margin_pct = (tr_rmse - pt_rmse) / pt_rmse * 100

    abs_margin = abs(rmse_margin_pct)
    confidence = 'high' if abs_margin >= margin_threshold_pct else 'low'

    # Within threshold → prefer Transfer regardless of direction
    if abs_margin < margin_threshold_pct:
        return {
            'selected_model':  'transfer',
            'rmse_margin_pct': round(rmse_margin_pct, 4),
            'switched':        False,
            'decision_reason': REASON_WITHIN_THRESHOLD,
            'confidence':      confidence,
        }

    # Significant difference: select the better (lower RMSE) model
    if pt_rmse < tr_rmse:
        # PreTransfer wins — switch away from Transfer default
        return {
            'selected_model':  'pretransfer',
            'rmse_margin_pct': round(rmse_margin_pct, 4),
            'switched':        True,
            'decision_reason': REASON_PRETRANSFER_BETTER,
            'confidence':      confidence,
        }
    else:
        # Transfer wins (or exactly equal after threshold check, won't reach here)
        return {
            'selected_model':  'transfer',
            'rmse_margin_pct': round(rmse_margin_pct, 4),
            'switched':        False,
            'decision_reason': REASON_TRANSFER_BETTER,
            'confidence':      confidence,
        }


def apply_switching_to_df(pretransfer_df, transfer_df, margin_threshold_pct: float = 2.0):
    """
    Apply decide_model() row-by-row over two data efficiency DataFrames and
    return a merged DataFrame with switch decisions integrated.

    Args:
        pretransfer_df: DataFrame with columns [weeks, mae, rmse, r2, mape, median_ae]
        transfer_df:    DataFrame with columns [weeks, mae, rmse, r2, mape, median_ae]
        margin_threshold_pct: Significance threshold forwarded to decide_model().

    Returns:
        DataFrame with columns:
            weeks, pretransfer_mae, pretransfer_rmse, pretransfer_r2,
            transfer_mae, transfer_rmse, transfer_r2,
            selected_model, rmse_margin_pct, switched, decision_reason, confidence
    """
    import pandas as pd

    # Index by weeks for easy lookup
    pt = pretransfer_df.set_index('weeks')
    tr = transfer_df.set_index('weeks')

    all_weeks = sorted(set(pt.index.tolist()) | set(tr.index.tolist()))

    rows = []
    for weeks in all_weeks:
        pt_row = pt.loc[weeks] if weeks in pt.index else {}
        tr_row = tr.loc[weeks] if weeks in tr.index else {}

        pt_metrics = {
            'rmse': pt_row.get('rmse') if hasattr(pt_row, 'get') else None,
            'mae':  pt_row.get('mae')  if hasattr(pt_row, 'get') else None,
            'r2':   pt_row.get('r2')   if hasattr(pt_row, 'get') else None,
        }
        tr_metrics = {
            'rmse': tr_row.get('rmse') if hasattr(tr_row, 'get') else None,
            'mae':  tr_row.get('mae')  if hasattr(tr_row, 'get') else None,
            'r2':   tr_row.get('r2')   if hasattr(tr_row, 'get') else None,
        }

        decision = decide_model(pt_metrics, tr_metrics, margin_threshold_pct)

        rows.append({
            'weeks':              weeks,
            'pretransfer_mae':    pt_metrics.get('mae'),
            'pretransfer_rmse':   pt_metrics.get('rmse'),
            'pretransfer_r2':     pt_metrics.get('r2'),
            'transfer_mae':       tr_metrics.get('mae'),
            'transfer_rmse':      tr_metrics.get('rmse'),
            'transfer_r2':        tr_metrics.get('r2'),
            'selected_model':     decision['selected_model'],
            'rmse_margin_pct':    decision['rmse_margin_pct'],
            'switched':           decision['switched'],
            'decision_reason':    decision['decision_reason'],
            'confidence':         decision['confidence'],
        })

    return pd.DataFrame(rows)
