"""
Multi-Transfer Generalisation Experiment

Answers: Does multi-source pre-training help on a target building where
single-source Transfer does NOT collapse?

In the eagle_education experiment, single-source Transfer collapses at <16
weeks (MAE spikes).  Multi-Transfer was designed to fix that.  But does it
also help on "easy" targets — or even hurt them?

This experiment replicates the multi_transfer protocol on a second target:
  Rat_education_Denise  (Rat / Education)

where rat_education experiments show single-source Transfer works fine.
The results establish whether multi-source pre-training is a generally
beneficial strategy or only rescues failure cases.

Experiment design
─────────────────
  Target building  : Rat_education_Denise    (Rat / Education)
  Single-source    : Rat_education_Colin     (same site — fairest comparison)
  Multi-source pool: same 5 buildings as multi_transfer experiment

  Strategies compared (same strategy keys as multi_transfer, different target):
    pretransfer   — scratch on N weeks Rat/Denise
    transfer      — fine-tune Rat/Colin baseline on N weeks Rat/Denise
    multitransfer — fine-tune 5-building baseline on N weeks Rat/Denise

Architecture note
─────────────────
  Rat/Denise target: input_size = 31
  Multi-source 5-building baseline: input_size = 29 (intersection)
  → train_multi_transfer truncates target features to 29 automatically.

File outputs
────────────
  models/experiments/multitransfer_generalisation/
    baseline_single_rat_colin.ckpt
    baseline_multi_5buildings.ckpt
    data_efficiency/
      pretransfer_Rat_education_D_{N}week_*.ckpt
      transfer_Rat_education_D_{N}week_*.ckpt
      multitransfer_Rat_education_D_{N}week_*.ckpt

  results/experiments/multitransfer_generalisation/
    data_efficiency_pretransfer.csv
    data_efficiency_transfer.csv
    data_efficiency_multitransfer.csv

Usage
─────
  python run_multitransfer_generalisation_experiment.py
  python run_multitransfer_generalisation_experiment.py --skip-baselines
  python run_multitransfer_generalisation_experiment.py --eval-only
"""

import sys
import os
import argparse
import glob
import shutil

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

from src.train_baseline import train_baseline
from src.train_pretransfer import train_pretransfer
from src.train_transfer import train_transfer
from src.train_multi_transfer import train_multi_transfer
from evaluate_all_models import evaluate_data_efficiency

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = 'multitransfer_generalisation'

DEFAULT_TARGET_BUILDING = 'Rat_education_Denise'
TARGET_SITE             = 'Rat'
TARGET_TYPE             = 'Education'

SINGLE_SOURCE_BUILDING  = 'Rat_education_Colin'
SINGLE_SOURCE_SITE      = 'Rat'
SINGLE_SOURCE_TYPE      = 'Education'

MULTI_SOURCE_BUILDINGS = [
    'Rat_education_Colin',
    'Eagle_education_Samantha',
    'Lamb_education_Lucas',
    'Hog_office_Miriam',
    'Robin_lodging_Celia',
]

WEEKS = [1, 2, 4, 8, 16, 32, 64, 104]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Baselines
# ─────────────────────────────────────────────────────────────────────────────

def prepare_single_source_baseline(project_root):
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    existing = _latest(os.path.join(exp_dir, 'baseline_single_*.ckpt'))
    if existing:
        print(f"\n[Single-source baseline] Already exists — skipping.")
        return existing

    # Reuse rat_education baseline (same building)
    rat_ckpt = _latest(os.path.join(
        project_root, 'models', 'experiments', 'rat_education', 'baseline_*.ckpt'
    ))
    if rat_ckpt:
        dest = os.path.join(exp_dir, 'baseline_single_rat_colin.ckpt')
        shutil.copy2(rat_ckpt, dest)
        print(f"\n[Single-source baseline] Reused from rat_education experiment.")
        return dest

    # Train from scratch
    print(f"\n[Single-source baseline] Training {SINGLE_SOURCE_BUILDING}...")
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
        raise RuntimeError('Single-source baseline training failed.')
    dest = os.path.join(exp_dir, 'baseline_single_rat_colin.ckpt')
    os.rename(ckpt, dest)
    return dest


def prepare_multi_source_baseline(project_root):
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    existing = _latest(os.path.join(exp_dir, 'baseline_multi_*.ckpt'))
    if existing:
        print(f"\n[Multi-source baseline] Already exists — skipping.")
        return existing

    # Reuse multi_transfer's 5-building baseline (identical pool)
    mt_multi = _latest(os.path.join(
        project_root, 'models', 'experiments', 'multi_transfer', 'baseline_multi_*.ckpt'
    ))
    if mt_multi:
        dest = os.path.join(exp_dir, 'baseline_multi_5buildings.ckpt')
        shutil.copy2(mt_multi, dest)
        print(f"\n[Multi-source baseline] Reused from multi_transfer experiment.")
        return dest

    # Train from scratch
    print(f"\n[Multi-source baseline] Training {len(MULTI_SOURCE_BUILDINGS)}-building pool...")
    train_baseline(
        building_ids=MULTI_SOURCE_BUILDINGS,
        epochs=50,
        seq_length=168,
        site_id=None,
        building_type=None,
        experiment_name=EXPERIMENT_NAME,
    )
    ckpt = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
    if not ckpt:
        raise RuntimeError('Multi-source baseline training failed.')
    dest = os.path.join(exp_dir, 'baseline_multi_5buildings.ckpt')
    os.rename(ckpt, dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Data-efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(target_building, single_source_path,
                              multi_source_path, project_root):
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    de_dir  = os.path.join(exp_dir, 'data_efficiency')
    os.makedirs(de_dir, exist_ok=True)

    tgt = target_building[:15]

    for weeks in WEEKS:
        print(f"\n{'#'*80}")
        print(f"  {weeks} WEEK(S) — {EXPERIMENT_NAME}")
        print(f"{'#'*80}")

        # ── 1. Pre-Transfer (scratch) ────────────────────────────────────
        if _latest(os.path.join(de_dir, f'pretransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[1/3] Pre-Transfer {weeks}w — already trained, skipping.")
        else:
            print(f"\n[1/3] Pre-Transfer {weeks}w — training from scratch...")
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
                    shutil.move(fresh, os.path.join(
                        de_dir, f'pretransfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    ))
                    print(f"  ✓ Saved")
            except Exception as exc:
                print(f"  ✗ FAILED: {exc}"); import traceback; traceback.print_exc()

        # ── 2. Transfer (single-source) ──────────────────────────────────
        if _latest(os.path.join(de_dir, f'transfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[2/3] Transfer {weeks}w — already trained, skipping.")
        else:
            print(f"\n[2/3] Transfer {weeks}w — fine-tuning Rat/Colin baseline...")
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
                fresh = _latest(os.path.join(exp_dir, f'transfer_{src_prefix}_{tgt}_*.ckpt'))
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    shutil.move(fresh, os.path.join(
                        de_dir, f'transfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    ))
                    print(f"  ✓ Saved")
            except Exception as exc:
                print(f"  ✗ FAILED: {exc}"); import traceback; traceback.print_exc()

        # ── 3. Multi-Transfer ────────────────────────────────────────────
        if _latest(os.path.join(de_dir, f'multitransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[3/3] Multi-Transfer {weeks}w — already trained, skipping.")
        else:
            print(f"\n[3/3] Multi-Transfer {weeks}w — fine-tuning 5-building baseline...")
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
                fresh = _latest(os.path.join(exp_dir, f'multitransfer_{tgt}_*.ckpt'))
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    shutil.move(fresh, os.path.join(
                        de_dir, f'multitransfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    ))
                    print(f"  ✓ Saved")
            except Exception as exc:
                print(f"  ✗ FAILED: {exc}"); import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Evaluate & save CSVs
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(target_building, project_root):
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
        description='Multi-Transfer Generalisation Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target', default=DEFAULT_TARGET_BUILDING)
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    target_building = args.target
    project_root = get_project_root()

    print('\n' + '=' * 80)
    print('  MULTI-TRANSFER GENERALISATION EXPERIMENT')
    print('=' * 80)
    print(f'  Target building  : {target_building}  (Rat / Education)')
    print(f'  Single source    : {SINGLE_SOURCE_BUILDING}')
    print(f'  Multi sources    : same 5-building pool as multi_transfer')
    print(f'  Weeks sweep      : {WEEKS}')
    print('=' * 80)

    if args.eval_only:
        evaluate_and_save(target_building, project_root)
        return

    if args.skip_baselines:
        exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
        single_source_path = _latest(os.path.join(exp_dir, 'baseline_single_*.ckpt'))
        multi_source_path  = _latest(os.path.join(exp_dir, 'baseline_multi_*.ckpt'))
        if not single_source_path or not multi_source_path:
            print('ERROR: baselines missing.')
            sys.exit(1)
    else:
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

    print('\n' + '=' * 80)
    print('  COMPLETE — Output files:')
    print('=' * 80)
    results_dir = os.path.join(project_root, 'results', 'experiments', EXPERIMENT_NAME)
    for fname in ('data_efficiency_pretransfer.csv', 'data_efficiency_transfer.csv',
                  'data_efficiency_multitransfer.csv'):
        status = '✓' if os.path.exists(os.path.join(results_dir, fname)) else '✗ MISSING'
        print(f'  {status}  results/experiments/{EXPERIMENT_NAME}/{fname}')
    print('=' * 80)


if __name__ == '__main__':
    main()
