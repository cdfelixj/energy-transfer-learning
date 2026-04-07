"""
Switch Modelling Experiment Orchestrator

Compares PreTransfer and Transfer data-efficiency strategies, then applies
automatic model switching: for each week count, the model with significantly
better RMSE is selected (minimum 2 % improvement required to switch away from
the Transfer default; see src/switch_logic.py for full decision rules).

Experiment design
─────────────────
  Target building   : Rat_education_Denise  (Rat / Education)
  Single source     : Rat_education_Colin   (same site — fairest comparison)
  Weeks sweep       : [1, 2, 4, 8, 16, 32, 64, 104]
  Switch threshold  : 2.0 %  (tune via --threshold)

Starting with rat_education because it is the most mature experiment and
already has trained baselines (which this script reuses by default).

File outputs
────────────
  models/experiments/switch_modelling/
    baseline_rat_colin.ckpt              (copy / symlink from rat_education)
    data_efficiency/
      pretransfer_Rat_education_D_{N}week_*.ckpt
      transfer_Rat_education_D_{N}week_*.ckpt

  results/experiments/switch_modelling/
    data_efficiency_pretransfer.csv
    data_efficiency_transfer.csv
    data_efficiency_switched.csv          ← primary experiment output

  results/switch_modelling/
    {experiment_name}_switched.csv        ← centralized copy
    switch_summary.csv                    ← key-value statistics

Usage
─────
  python run_switch_modelling_experiment.py
  python run_switch_modelling_experiment.py --threshold 5.0
  python run_switch_modelling_experiment.py --target Rat_education_Denise
  python run_switch_modelling_experiment.py --skip-baseline
  python run_switch_modelling_experiment.py --eval-only
  python run_switch_modelling_experiment.py --eval-only --threshold 3.0
"""

import sys
import os
import argparse
import glob
import shutil

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

from src.train_baseline      import train_baseline
from src.train_pretransfer   import train_pretransfer
from src.train_transfer      import train_transfer
from src.analyze_switching   import print_switching_report, analyze_switching_patterns
from evaluate_all_models     import evaluate_data_efficiency_with_switching

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = 'switch_modelling'

DEFAULT_TARGET_BUILDING = 'Rat_education_Denise'
TARGET_SITE             = 'Rat'
TARGET_TYPE             = 'Education'

SOURCE_BUILDING         = 'Rat_education_Colin'
SOURCE_SITE             = 'Rat'
SOURCE_TYPE             = 'Education'

WEEKS                   = [1, 2, 4, 8, 16, 32, 64, 104]
DEFAULT_THRESHOLD_PCT   = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _latest(pattern: str):
    """Return the most recently modified file matching *pattern*, or None."""
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Baseline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_baseline(project_root: str) -> str:
    """Return path to baseline checkpoint, training only when needed."""
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Already exists for this experiment?
    existing = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
    if existing:
        print(f"\n[Baseline] Already exists — skipping training.")
        print(f"  Using: {os.path.basename(existing)}")
        return existing

    # 2. Reuse from rat_education (same source building)
    rat_ckpt = _latest(
        os.path.join(project_root, 'models', 'experiments', 'rat_education', 'baseline_*.ckpt')
    )
    if rat_ckpt:
        dest = os.path.join(exp_dir, 'baseline_rat_colin.ckpt')
        shutil.copy2(rat_ckpt, dest)
        print(f"\n[Baseline] Reused from rat_education experiment.")
        print(f"  → baseline_rat_colin.ckpt")
        return dest

    # 3. Train from scratch
    print(f"\n[Baseline] Training {SOURCE_BUILDING}...")
    train_baseline(
        building_ids=[SOURCE_BUILDING],
        epochs=50,
        seq_length=168,
        site_id=SOURCE_SITE,
        building_type=SOURCE_TYPE,
        experiment_name=EXPERIMENT_NAME,
    )
    ckpt = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
    if not ckpt:
        raise RuntimeError('Baseline training failed — no checkpoint saved.')
    dest = os.path.join(exp_dir, 'baseline_rat_colin.ckpt')
    os.rename(ckpt, dest)
    print(f"  → baseline_rat_colin.ckpt")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Data-efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(target_building: str, baseline_path: str, project_root: str) -> None:
    """Train PreTransfer and Transfer models for each week count, skipping done runs."""
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    de_dir  = os.path.join(exp_dir, 'data_efficiency')
    os.makedirs(de_dir, exist_ok=True)

    tgt = target_building[:15]

    for weeks in WEEKS:
        print(f"\n{'#' * 80}")
        print(f"  {weeks} WEEK(S) — {EXPERIMENT_NAME}")
        print(f"{'#' * 80}")

        # ── PreTransfer (scratch) ─────────────────────────────────────────
        if _latest(os.path.join(de_dir, f'pretransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[1/2] PreTransfer ({weeks} wks) — already trained, skipping.")
        else:
            print(f"\n[1/2] PreTransfer ({weeks} wks) — training from scratch...")
            try:
                train_pretransfer(
                    target_building=target_building,
                    epochs=100,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    architecture_match=baseline_path,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                fresh = _latest(os.path.join(exp_dir, f'pretransfer_{tgt}_*.ckpt'))
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'pretransfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
                else:
                    print(f"  ⚠ Checkpoint not found after PreTransfer training.")
            except Exception as exc:
                print(f"  ✗ PreTransfer ({weeks} wks) FAILED: {exc}")
                import traceback; traceback.print_exc()

        # ── Transfer (fine-tune from baseline) ────────────────────────────
        if _latest(os.path.join(de_dir, f'transfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[2/2] Transfer ({weeks} wks) — already trained, skipping.")
        else:
            print(f"\n[2/2] Transfer ({weeks} wks) — fine-tuning baseline...")
            try:
                train_transfer(
                    source_building=SOURCE_BUILDING,
                    target_building=target_building,
                    source_model_path=baseline_path,
                    epochs=50,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                src_prefix = SOURCE_BUILDING[:15]
                fresh = _latest(
                    os.path.join(exp_dir, f'transfer_{src_prefix}_{tgt}_*.ckpt')
                )
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'transfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
                else:
                    print(f"  ⚠ Checkpoint not found after Transfer training.")
            except Exception as exc:
                print(f"  ✗ Transfer ({weeks} wks) FAILED: {exc}")
                import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Evaluate & save outputs
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(
    target_building: str,
    project_root: str,
    margin_threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> None:
    """Evaluate PreTransfer, Transfer and the switched strategy; write CSVs."""
    import pandas as pd
    from evaluate_all_models import evaluate_data_efficiency

    results_dir     = os.path.join(project_root, 'results', 'experiments', EXPERIMENT_NAME)
    central_dir     = os.path.join(project_root, 'results', 'switch_modelling')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(central_dir, exist_ok=True)

    # ── Evaluate individual strategies ────────────────────────────────────
    print(f"\n  Evaluating PreTransfer...")
    pt_df = evaluate_data_efficiency(
        model_type='pretransfer',
        target_building=target_building,
        weeks_list=WEEKS,
        seq_length=24,
        experiment_name=EXPERIMENT_NAME,
        site_id=TARGET_SITE,
        building_type=TARGET_TYPE,
    )
    pt_path = os.path.join(results_dir, 'data_efficiency_pretransfer.csv')
    pt_df.to_csv(pt_path, index=False)
    print(f"  ✓ Saved: results/experiments/{EXPERIMENT_NAME}/data_efficiency_pretransfer.csv")

    print(f"\n  Evaluating Transfer...")
    tr_df = evaluate_data_efficiency(
        model_type='transfer',
        target_building=target_building,
        weeks_list=WEEKS,
        seq_length=24,
        experiment_name=EXPERIMENT_NAME,
        site_id=TARGET_SITE,
        building_type=TARGET_TYPE,
    )
    tr_path = os.path.join(results_dir, 'data_efficiency_transfer.csv')
    tr_df.to_csv(tr_path, index=False)
    print(f"  ✓ Saved: results/experiments/{EXPERIMENT_NAME}/data_efficiency_transfer.csv")

    # ── Apply switching ───────────────────────────────────────────────────
    print(f"\n  Applying switch logic (threshold={margin_threshold_pct}%)...")
    switched_df = evaluate_data_efficiency_with_switching(
        target_building=target_building,
        weeks_list=WEEKS,
        seq_length=24,
        experiment_name=EXPERIMENT_NAME,
        site_id=TARGET_SITE,
        building_type=TARGET_TYPE,
        margin_threshold_pct=margin_threshold_pct,
    )

    # Per-experiment output
    sw_path = os.path.join(results_dir, 'data_efficiency_switched.csv')
    switched_df.to_csv(sw_path, index=False)
    print(f"  ✓ Saved: results/experiments/{EXPERIMENT_NAME}/data_efficiency_switched.csv")

    # Centralized copy
    central_sw = os.path.join(central_dir, f'{EXPERIMENT_NAME}_switched.csv')
    switched_df.to_csv(central_sw, index=False)
    print(f"  ✓ Saved: results/switch_modelling/{EXPERIMENT_NAME}_switched.csv")

    # ── Switching analysis report ─────────────────────────────────────────
    print_switching_report(switched_df, EXPERIMENT_NAME)
    summary_df = analyze_switching_patterns(switched_df)
    summary_path = os.path.join(central_dir, 'switch_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ✓ Saved: results/switch_modelling/switch_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run the Switch Modelling data-efficiency experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--target',
        default=DEFAULT_TARGET_BUILDING,
        help=f'Target building (default: {DEFAULT_TARGET_BUILDING})',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help=f'Minimum RMSE %% improvement to trigger a switch (default: {DEFAULT_THRESHOLD_PCT})',
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline training (fail if checkpoint does not exist)',
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip all training; evaluate existing models and apply switching',
    )
    args = parser.parse_args()

    target_building       = args.target
    margin_threshold_pct  = args.threshold
    project_root          = get_project_root()

    print('\n' + '=' * 80)
    print('  SWITCH MODELLING EXPERIMENT')
    print('=' * 80)
    print(f'  Target building  : {target_building}')
    print(f'  Source building  : {SOURCE_BUILDING}')
    print(f'  Weeks sweep      : {WEEKS}')
    print(f'  Switch threshold : {margin_threshold_pct}%')
    print(f'  Experiment dir   : models/experiments/{EXPERIMENT_NAME}/')
    print('=' * 80)

    if args.eval_only:
        print('\n[--eval-only] Skipping all training.')
        evaluate_and_save(target_building, project_root, margin_threshold_pct)

    elif args.skip_baseline:
        exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
        baseline_path = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
        if not baseline_path:
            print('ERROR: --skip-baseline set but no baseline_*.ckpt found.')
            sys.exit(1)
        print(f'\n[--skip-baseline] Using existing baseline:')
        print(f'  {os.path.basename(baseline_path)}')
        print('\n' + '─' * 80)
        print('  STEP B — Data-Efficiency Sweep')
        print('─' * 80)
        run_data_efficiency_sweep(target_building, baseline_path, project_root)
        print('\n' + '─' * 80)
        print('  STEP C — Evaluation & Switching')
        print('─' * 80)
        evaluate_and_save(target_building, project_root, margin_threshold_pct)

    else:
        # Full pipeline
        print('\n' + '─' * 80)
        print('  STEP A — Baseline')
        print('─' * 80)
        baseline_path = prepare_baseline(project_root)

        print('\n' + '─' * 80)
        print('  STEP B — Data-Efficiency Sweep')
        print('─' * 80)
        run_data_efficiency_sweep(target_building, baseline_path, project_root)

        print('\n' + '─' * 80)
        print('  STEP C — Evaluation & Switching')
        print('─' * 80)
        evaluate_and_save(target_building, project_root, margin_threshold_pct)

    # ── Output summary ────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('  COMPLETE — Output files:')
    print('=' * 80)
    results_dir  = os.path.join(project_root, 'results', 'experiments', EXPERIMENT_NAME)
    central_dir  = os.path.join(project_root, 'results', 'switch_modelling')
    expected = [
        (results_dir,  'data_efficiency_pretransfer.csv'),
        (results_dir,  'data_efficiency_transfer.csv'),
        (results_dir,  'data_efficiency_switched.csv'),
        (central_dir,  f'{EXPERIMENT_NAME}_switched.csv'),
        (central_dir,  'switch_summary.csv'),
    ]
    for dirpath, fname in expected:
        full = os.path.join(dirpath, fname)
        status = '✓' if os.path.exists(full) else '✗ MISSING'
        rel = os.path.relpath(full, project_root).replace('\\', '/')
        print(f'  {status}  {rel}')
    print('=' * 80)


if __name__ == '__main__':
    main()
