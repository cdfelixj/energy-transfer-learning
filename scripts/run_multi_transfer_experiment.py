"""
Multi-Transfer Experiment Orchestrator

Compares three data-efficiency strategies for adapting to a target building
with limited data:

  1. Pre-Transfer   (scratch)          — train from scratch on N weeks of target data
  2. Transfer       (single-source)    — fine-tune from Eagle/Samantha-only baseline
  3. Multi-Transfer (multi-source)     — fine-tune from diverse 5-building baseline

Experiment design
─────────────────
  Target building  : Eagle_education_Brooke  (Eagle / Education, 99.78% complete)
  Single-source    : Eagle_education_Samantha (same site — fairest comparison)
  Multi-source pool: Rat/Colin + Eagle/Samantha + Lamb/Lucas + Hog/Miriam + Robin/Celia
                     (5 buildings, 3 sites, 3 types, 2 years each)

Eagle/Brooke is specifically chosen because the single-source Transfer model
collapses at <16 weeks there (see eagle_education experiment).  Multi-Transfer
is the main hypothesis for fixing that failure.

File outputs
────────────
  models/experiments/multi_transfer/
    baseline_single_eagle_samantha.ckpt
    baseline_multi_5buildings.ckpt
    data_efficiency/
      pretransfer_Eagle_educati_{N}week_epoch=...ckpt   (8 files)
      transfer_Eagle_educati_{N}week_epoch=...ckpt      (8 files)
      multitransfer_Eagle_educati_{N}week_epoch=...ckpt (8 files)

  results/experiments/multi_transfer/
    data_efficiency_pretransfer.csv
    data_efficiency_transfer.csv
    data_efficiency_multitransfer.csv

Usage
─────
  python run_multi_transfer_experiment.py
  python run_multi_transfer_experiment.py --target Eagle_education_Brooke
  python run_multi_transfer_experiment.py --skip-baselines    # re-use existing baselines
  python run_multi_transfer_experiment.py --eval-only         # skip training, evaluate only
"""

import sys
import os
import argparse
import glob
import shutil

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import pandas as pd

from src.train_baseline import train_baseline
from src.train_pretransfer import train_pretransfer
from src.train_transfer import train_transfer
from src.train_multi_transfer import train_multi_transfer
from evaluate_all_models import evaluate_data_efficiency

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = 'multi_transfer'

DEFAULT_TARGET_BUILDING = 'Eagle_education_Brooke'
TARGET_SITE             = 'Eagle'
TARGET_TYPE             = 'Education'

SINGLE_SOURCE_BUILDING = 'Eagle_education_Samantha'
SINGLE_SOURCE_SITE     = 'Eagle'
SINGLE_SOURCE_TYPE     = 'Education'

# 5 buildings — diverse sites and building types
MULTI_SOURCE_BUILDINGS = [
    'Rat_education_Colin',       # Rat   / Education
    'Eagle_education_Samantha',  # Eagle / Education
    'Lamb_education_Lucas',      # Lamb  / Education
    'Hog_office_Miriam',         # Hog   / Office
    'Robin_lodging_Celia',       # Robin / Lodging/residential
]

WEEKS = [1, 2, 4, 8, 16, 32, 64, 104]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _latest(pattern):
    """Return the most recently modified file matching glob pattern, or None."""
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Train / reuse baselines
# ─────────────────────────────────────────────────────────────────────────────

def prepare_single_source_baseline(project_root):
    """Return path to single-source baseline checkpoint, training only if needed."""
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Already prepared for this experiment?
    existing = _latest(os.path.join(exp_dir, 'baseline_single_*.ckpt'))
    if existing:
        print(f"\n[Single-source baseline] Already exists — skipping.")
        print(f"  Using: {os.path.basename(existing)}")
        return existing

    # 2. Reuse from eagle_education if available (same model)
    eagle_ckpt = _latest(
        os.path.join(project_root, 'models', 'experiments', 'eagle_education', 'baseline_*.ckpt')
    )
    if eagle_ckpt:
        dest = os.path.join(exp_dir, 'baseline_single_eagle_samantha.ckpt')
        shutil.copy2(eagle_ckpt, dest)
        print(f"\n[Single-source baseline] Reused from eagle_education experiment.")
        print(f"  → baseline_single_eagle_samantha.ckpt")
        return dest

    # 3. Train from scratch
    print(f"\n[Single-source baseline] Training {SINGLE_SOURCE_BUILDING}...")
    print(f"  Site: {SINGLE_SOURCE_SITE}  |  Type: {SINGLE_SOURCE_TYPE}")
    train_baseline(
        building_ids=[SINGLE_SOURCE_BUILDING],
        epochs=50,
        seq_length=168,
        site_id=SINGLE_SOURCE_SITE,
        building_type=SINGLE_SOURCE_TYPE,
        experiment_name=EXPERIMENT_NAME,
    )
    ckpt = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
    if not ckpt:
        raise RuntimeError(
            'Single-source baseline training failed — no checkpoint saved.'
        )
    dest = os.path.join(exp_dir, 'baseline_single_eagle_samantha.ckpt')
    os.rename(ckpt, dest)
    print(f"  → baseline_single_eagle_samantha.ckpt")
    return dest


def prepare_multi_source_baseline(project_root):
    """Return path to multi-source baseline checkpoint, training only if needed."""
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    existing = _latest(os.path.join(exp_dir, 'baseline_multi_*.ckpt'))
    if existing:
        print(f"\n[Multi-source baseline] Already exists — skipping.")
        print(f"  Using: {os.path.basename(existing)}")
        return existing

    print(f"\n[Multi-source baseline] Training on {len(MULTI_SOURCE_BUILDINGS)} buildings:")
    for b in MULTI_SOURCE_BUILDINGS:
        print(f"  • {b}")
    print(f"  site_id=None (all sites)  |  building_type=None (all types)")

    train_baseline(
        building_ids=MULTI_SOURCE_BUILDINGS,
        epochs=50,
        seq_length=168,
        site_id=None,        # span all sites
        building_type=None,  # span all types
        experiment_name=EXPERIMENT_NAME,
    )

    # The baseline script names the file with building_ids[0][:20] prefix
    ckpt = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
    if not ckpt:
        raise RuntimeError(
            'Multi-source baseline training failed — no checkpoint saved.'
        )
    dest = os.path.join(exp_dir, 'baseline_multi_5buildings.ckpt')
    os.rename(ckpt, dest)
    print(f"  → baseline_multi_5buildings.ckpt")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Data-efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(target_building, single_source_path,
                              multi_source_path, project_root):
    """Train all three strategies at each week count, skipping existing runs."""
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    de_dir  = os.path.join(exp_dir, 'data_efficiency')
    os.makedirs(de_dir, exist_ok=True)

    tgt = target_building[:15]

    for weeks in WEEKS:
        print(f"\n{'#' * 80}")
        print(f"  {weeks} WEEK(S) — {EXPERIMENT_NAME}")
        print(f"{'#' * 80}")

        # ── 1. Pre-Transfer (scratch) ─────────────────────────────────────
        if _latest(os.path.join(de_dir, f'pretransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[1/3] Pre-Transfer ({weeks} wks) — already trained, skipping.")
        else:
            print(f"\n[1/3] Pre-Transfer ({weeks} wks) — training from scratch...")
            try:
                train_pretransfer(
                    target_building=target_building,
                    epochs=100,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    architecture_match=single_source_path,
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
            except Exception as exc:
                print(f"  ✗ Pre-Transfer ({weeks} wks) FAILED: {exc}")
                import traceback; traceback.print_exc()

        # ── 2. Transfer (single-source) ───────────────────────────────────
        if _latest(os.path.join(de_dir, f'transfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[2/3] Transfer ({weeks} wks) — already trained, skipping.")
        else:
            print(f"\n[2/3] Transfer ({weeks} wks) — fine-tuning single-source baseline...")
            try:
                train_transfer(
                    source_building=SINGLE_SOURCE_BUILDING,
                    target_building=target_building,
                    source_model_path=single_source_path,
                    epochs=50,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                src_prefix = SINGLE_SOURCE_BUILDING[:15]
                fresh = _latest(
                    os.path.join(exp_dir, f'transfer_{src_prefix}_{tgt}_*.ckpt')
                )
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'transfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
            except Exception as exc:
                print(f"  ✗ Transfer ({weeks} wks) FAILED: {exc}")
                import traceback; traceback.print_exc()

        # ── 3. Multi-Transfer ─────────────────────────────────────────────
        if _latest(os.path.join(de_dir, f'multitransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[3/3] Multi-Transfer ({weeks} wks) — already trained, skipping.")
        else:
            print(f"\n[3/3] Multi-Transfer ({weeks} wks) — fine-tuning 5-building baseline...")
            try:
                train_multi_transfer(
                    target_building=target_building,
                    multi_baseline_model_path=multi_source_path,
                    epochs=50,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                fresh = _latest(
                    os.path.join(exp_dir, f'multitransfer_{tgt}_*.ckpt')
                )
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'multitransfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
            except Exception as exc:
                print(f"  ✗ Multi-Transfer ({weeks} wks) FAILED: {exc}")
                import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Evaluate & save CSVs
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(target_building, project_root):
    """Evaluate all three strategies and write data_efficiency_*.csv files."""
    results_dir = os.path.join(
        project_root, 'results', 'experiments', EXPERIMENT_NAME
    )
    os.makedirs(results_dir, exist_ok=True)

    for model_type in ('pretransfer', 'transfer', 'multitransfer'):
        print(f"\n  Evaluating {model_type}...")
        df = evaluate_data_efficiency(
            model_type=model_type,
            target_building=target_building,
            weeks_list=WEEKS,
            seq_length=24,
            experiment_name=EXPERIMENT_NAME,
            site_id=TARGET_SITE,
            building_type=TARGET_TYPE,
        )
        out_path = os.path.join(results_dir, f'data_efficiency_{model_type}.csv')
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved: results/experiments/{EXPERIMENT_NAME}/data_efficiency_{model_type}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run the Multi-Transfer data-efficiency experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--target',
        default=DEFAULT_TARGET_BUILDING,
        help=f'Target building (default: {DEFAULT_TARGET_BUILDING})',
    )
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline training (fail if checkpoints do not exist)',
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip all training and run evaluation only',
    )
    args = parser.parse_args()
    target_building = args.target

    project_root = get_project_root()

    print('\n' + '=' * 80)
    print('  MULTI-TRANSFER EXPERIMENT')
    print('=' * 80)
    print(f'  Target building : {target_building}')
    print(f'  Single source   : {SINGLE_SOURCE_BUILDING}')
    print(f'  Multi sources   :')
    for b in MULTI_SOURCE_BUILDINGS:
        print(f'    • {b}')
    print(f'  Weeks sweep     : {WEEKS}')
    print(f'  Experiment dir  : models/experiments/{EXPERIMENT_NAME}/')
    print('=' * 80)

    if args.eval_only:
        print('\n[--eval-only] Skipping all training.')
        evaluate_and_save(target_building, project_root)
    elif args.skip_baselines:
        exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
        single_source_path = _latest(os.path.join(exp_dir, 'baseline_single_*.ckpt'))
        multi_source_path  = _latest(os.path.join(exp_dir, 'baseline_multi_*.ckpt'))
        if not single_source_path:
            print('ERROR: --skip-baselines set but no baseline_single_*.ckpt found.')
            sys.exit(1)
        if not multi_source_path:
            print('ERROR: --skip-baselines set but no baseline_multi_*.ckpt found.')
            sys.exit(1)
        print(f'\n[--skip-baselines] Using existing baselines:')
        print(f'  Single: {os.path.basename(single_source_path)}')
        print(f'  Multi:  {os.path.basename(multi_source_path)}')
        run_data_efficiency_sweep(
            target_building, single_source_path, multi_source_path, project_root
        )
        evaluate_and_save(target_building, project_root)
    else:
        # Full pipeline
        print('\n' + '─' * 80)
        print('  STEP A — Baselines')
        print('─' * 80)
        single_source_path = prepare_single_source_baseline(project_root)
        multi_source_path  = prepare_multi_source_baseline(project_root)

        print('\n' + '─' * 80)
        print('  STEP B — Data-Efficiency Sweep')
        print('─' * 80)
        run_data_efficiency_sweep(
            target_building, single_source_path, multi_source_path, project_root
        )

        print('\n' + '─' * 80)
        print('  STEP C — Evaluation')
        print('─' * 80)
        evaluate_and_save(target_building, project_root)

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('  COMPLETE — Output files:')
    print('=' * 80)
    results_dir = os.path.join(
        project_root, 'results', 'experiments', EXPERIMENT_NAME
    )
    for fname in (
        'data_efficiency_pretransfer.csv',
        'data_efficiency_transfer.csv',
        'data_efficiency_multitransfer.csv',
    ):
        full_path = os.path.join(results_dir, fname)
        status = '✓' if os.path.exists(full_path) else '✗ MISSING'
        print(f'  {status}  results/experiments/{EXPERIMENT_NAME}/{fname}')
    print('=' * 80)


if __name__ == '__main__':
    main()
