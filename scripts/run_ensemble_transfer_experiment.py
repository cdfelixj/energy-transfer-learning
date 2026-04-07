"""
Ensemble Transfer Experiment Orchestrator

Compares three data-efficiency strategies for adapting to a target building
with limited data:

  1. Pre-Transfer      (scratch)            — train from scratch on N weeks of target data
  2. Transfer          (single-source FT)   — fine-tune from Eagle/Samantha baseline
  3. Ensemble Transfer (model-soup FT)      — fine-tune from weight-averaged 5-model soup

Experiment design
─────────────────
  Target building  : Eagle_education_Brooke
  Single-source    : Eagle_education_Samantha (same site)
  Multi-source pool: same 5 buildings as Multi-Transfer experiment

  The ensemble is built by:
    i.  Training each of the 5 source buildings *individually* with features
        truncated to the common-feature count (determined by the multi-source
        baseline that uses their column intersection).
    ii. Computing a weighted average of the 5 resulting state dicts (uniform
        by default — "model soup").
    iii.Fine-tuning that averaged initialisation on the target building.

Architecture alignment
──────────────────────
  Per-site raw input sizes: Rat/Eagle/Office=31, Robin=30, Lamb=29.
  The multi-source baseline uses the intersection → input_size=29.
  Every individual ensemble component model is therefore also trained with
  input_size=29, allowing their weights to be summed.

File outputs
────────────
  models/experiments/ensemble_transfer/
    baseline_single_eagle_samantha.ckpt          (symlink / copy from multi_transfer)
    baseline_feature_reference.ckpt              (copy of multi_transfer multi-source)
    individual/
      individual_{building}_{epoch}_{loss}.ckpt  (5 files)
    data_efficiency/
      pretransfer_Eagle_educati_{N}week_*.ckpt   (8 files)
      transfer_Eagle_educati_{N}week_*.ckpt       (8 files)
      ensembletransfer_Eagle_educati_{N}week_*.ckpt (8 files)

  results/experiments/ensemble_transfer/
    data_efficiency_pretransfer.csv
    data_efficiency_transfer.csv
    data_efficiency_ensembletransfer.csv

Usage
─────
  python scripts/run_ensemble_transfer_experiment.py
  python scripts/run_ensemble_transfer_experiment.py --target Eagle_education_Brooke
  python scripts/run_ensemble_transfer_experiment.py --skip-baselines      # re-use existing
  python scripts/run_ensemble_transfer_experiment.py --eval-only           # no training, evaluate
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
from src.train_ensemble_transfer import (
    train_individual_source,
    average_model_weights,
    train_ensemble_transfer,
)
from evaluate_all_models import evaluate_data_efficiency

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = 'ensemble_transfer'

DEFAULT_TARGET_BUILDING = 'Eagle_education_Brooke'
TARGET_SITE             = 'Eagle'
TARGET_TYPE             = 'Education'

SINGLE_SOURCE_BUILDING = 'Eagle_education_Samantha'
SINGLE_SOURCE_SITE     = 'Eagle'
SINGLE_SOURCE_TYPE     = 'Education'

# Same 5-building pool as the Multi-Transfer experiment
MULTI_SOURCE_BUILDINGS = [
    'Rat_education_Colin',
    'Eagle_education_Samantha',
    'Lamb_education_Lucas',
    'Hog_office_Miriam',
    'Robin_lodging_Celia',
]

# Building meta for individual training calls (site_id, building_type)
BUILDING_META = {
    'Rat_education_Colin':      ('Rat',   'Education'),
    'Eagle_education_Samantha': ('Eagle', 'Education'),
    'Lamb_education_Lucas':     ('Lamb',  'Education'),
    'Hog_office_Miriam':        ('Hog',   'Office'),
    'Robin_lodging_Celia':      ('Robin', 'Lodging/residential'),
}

WEEKS = [1, 2, 4, 8, 16, 32, 64, 104]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _latest(pattern):
    """Return the most recently modified file matching glob pattern, or None."""
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Prepare baselines and individual source models
# ─────────────────────────────────────────────────────────────────────────────

def prepare_feature_reference(project_root):
    """Return a checkpoint whose input_size defines the target feature count.

    Prefers the multi_transfer multi-source baseline (already trained).
    Falls back to training a new multi-source baseline under this experiment.
    """
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Already copied into this experiment directory?
    ref = _latest(os.path.join(exp_dir, 'baseline_feature_reference*.ckpt'))
    if ref:
        print(f"\n[Feature reference] Already exists — skipping.")
        print(f"  Using: {os.path.basename(ref)}")
        return ref

    # 2. Re-use multi_transfer's multi-source baseline if available
    mt_multi = _latest(
        os.path.join(
            project_root, 'models', 'experiments', 'multi_transfer',
            'baseline_multi_*.ckpt'
        )
    )
    if mt_multi:
        dest = os.path.join(exp_dir, 'baseline_feature_reference.ckpt')
        shutil.copy2(mt_multi, dest)
        print(f"\n[Feature reference] Copied from multi_transfer experiment.")
        print(f"  Source: {os.path.basename(mt_multi)}")
        print(f"  → baseline_feature_reference.ckpt")
        return dest

    # 3. Train a new multi-source baseline for alignment purposes
    print(f"\n[Feature reference] Training fresh 5-building multi-source baseline...")
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
        raise RuntimeError(
            'Feature reference training failed — no checkpoint saved.'
        )
    dest = os.path.join(exp_dir, 'baseline_feature_reference.ckpt')
    os.rename(ckpt, dest)
    print(f"  → baseline_feature_reference.ckpt")
    return dest


def prepare_single_source_baseline(project_root):
    """Return path to single-source (Eagle/Samantha) baseline, training if needed."""
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Already in this experiment?
    existing = _latest(os.path.join(exp_dir, 'baseline_single_*.ckpt'))
    if existing:
        print(f"\n[Single-source baseline] Already exists — skipping.")
        print(f"  Using: {os.path.basename(existing)}")
        return existing

    # 2. Reuse from multi_transfer
    mt_single = _latest(
        os.path.join(
            project_root, 'models', 'experiments', 'multi_transfer',
            'baseline_single_*.ckpt'
        )
    )
    if mt_single:
        dest = os.path.join(exp_dir, 'baseline_single_eagle_samantha.ckpt')
        shutil.copy2(mt_single, dest)
        print(f"\n[Single-source baseline] Copied from multi_transfer experiment.")
        return dest

    # 3. Reuse from eagle_education
    eagle_ckpt = _latest(
        os.path.join(
            project_root, 'models', 'experiments', 'eagle_education',
            'baseline_*.ckpt'
        )
    )
    if eagle_ckpt:
        dest = os.path.join(exp_dir, 'baseline_single_eagle_samantha.ckpt')
        shutil.copy2(eagle_ckpt, dest)
        print(f"\n[Single-source baseline] Reused from eagle_education experiment.")
        return dest

    # 4. Train from scratch
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
    dest = os.path.join(exp_dir, 'baseline_single_eagle_samantha.ckpt')
    os.rename(ckpt, dest)
    return dest


def prepare_individual_source_models(feature_reference_path, project_root):
    """Train (or reuse) one feature-aligned baseline per source building.

    Returns a list of 5 checkpoint paths in the same order as MULTI_SOURCE_BUILDINGS.
    """
    ind_dir = os.path.join(
        project_root, 'models', 'experiments', EXPERIMENT_NAME, 'individual'
    )
    os.makedirs(ind_dir, exist_ok=True)

    model_paths = []

    for building in MULTI_SOURCE_BUILDINGS:
        # Skip if an individual checkpoint already exists for this building
        tag = building[:20]
        existing = _latest(os.path.join(ind_dir, f'individual_{tag}*.ckpt'))
        if existing:
            print(f"\n[Individual] {building} — already trained, skipping.")
            print(f"  Using: {os.path.basename(existing)}")
            model_paths.append(existing)
            continue

        print(f"\n[Individual] Training: {building}")
        site_id, building_type = BUILDING_META[building]

        # train_individual_source saves into EXPERIMENT_NAME exp_dir
        # We need to move it into ind_dir afterwards
        train_individual_source(
            source_building=building,
            epochs=50,
            seq_length=168,
            feature_reference_path=feature_reference_path,
            site_id=site_id,
            building_type=building_type,
            experiment_name=EXPERIMENT_NAME,
        )

        # Find the checkpoint just saved in exp_dir (not ind_dir)
        exp_base = os.path.join(
            project_root, 'models', 'experiments', EXPERIMENT_NAME
        )
        fresh = _latest(os.path.join(exp_base, f'individual_{tag}*.ckpt'))
        if not fresh:
            raise RuntimeError(
                f"No checkpoint found after training individual model for {building}."
            )

        # Move into ind_dir for organisation
        dest = os.path.join(ind_dir, os.path.basename(fresh))
        shutil.move(fresh, dest)
        print(f"  → individual/{os.path.basename(dest)}")
        model_paths.append(dest)

    print(f"\n[Individual models] All {len(model_paths)} ready:")
    for p in model_paths:
        print(f"  • {os.path.relpath(p, project_root)}")

    return model_paths


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Data-efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(target_building, single_source_path,
                              individual_model_paths, project_root):
    """Train pretransfer / transfer / ensembletransfer at each week count."""
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

        # ── 2. Transfer (single-source fine-tune) ─────────────────────────
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

        # ── 3. Ensemble Transfer ──────────────────────────────────────────
        if _latest(os.path.join(de_dir, f'ensembletransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[3/3] Ensemble Transfer ({weeks} wks) — already trained, skipping.")
        else:
            print(f"\n[3/3] Ensemble Transfer ({weeks} wks) — model soup fine-tune...")
            try:
                train_ensemble_transfer(
                    target_building=target_building,
                    source_model_paths=individual_model_paths,
                    model_weights=None,   # uniform averaging
                    epochs=50,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                fresh = _latest(
                    os.path.join(exp_dir, f'ensembletransfer_{tgt}_*.ckpt')
                )
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'ensembletransfer_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
            except Exception as exc:
                print(f"  ✗ Ensemble Transfer ({weeks} wks) FAILED: {exc}")
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

    for model_type in ('pretransfer', 'transfer', 'ensembletransfer'):
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
        print(
            f"  ✓ Saved: results/experiments/{EXPERIMENT_NAME}/"
            f"data_efficiency_{model_type}.csv"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run the Ensemble Transfer data-efficiency experiment',
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
        help='Skip baseline + individual model training; fail if checkpoints missing.',
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip all training and run evaluation only.',
    )
    args = parser.parse_args()
    target_building = args.target

    project_root = get_project_root()

    print('\n' + '=' * 80)
    print('  ENSEMBLE TRANSFER EXPERIMENT')
    print('=' * 80)
    print(f'  Target building  : {target_building}')
    print(f'  Single source    : {SINGLE_SOURCE_BUILDING}')
    print(f'  Ensemble sources :')
    for b in MULTI_SOURCE_BUILDINGS:
        print(f'    • {b}')
    print(f'  Weeks sweep      : {WEEKS}')
    print(f'  Experiment dir   : models/experiments/{EXPERIMENT_NAME}/')
    print('=' * 80)

    if args.eval_only:
        print('\n[--eval-only] Skipping all training.')
        evaluate_and_save(target_building, project_root)
        return

    if args.skip_baselines:
        exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
        single_source_path = _latest(os.path.join(exp_dir, 'baseline_single_*.ckpt'))
        ind_paths = sorted(
            glob.glob(os.path.join(exp_dir, 'individual', 'individual_*.ckpt'))
        )
        if not single_source_path:
            print('ERROR: --skip-baselines set but no baseline_single_*.ckpt found.')
            sys.exit(1)
        if len(ind_paths) < len(MULTI_SOURCE_BUILDINGS):
            print(
                f'ERROR: --skip-baselines set but only {len(ind_paths)} individual '
                f'model(s) found (need {len(MULTI_SOURCE_BUILDINGS)}).'
            )
            sys.exit(1)
        print(f'\n[--skip-baselines] Using existing models:')
        print(f'  Single-source: {os.path.basename(single_source_path)}')
        for p in ind_paths:
            print(f'  Individual:    {os.path.basename(p)}')
    else:
        # Full pipeline: Step A
        print('\n' + '─' * 80)
        print('  STEP A-1 — Feature reference baseline')
        print('─' * 80)
        feature_reference_path = prepare_feature_reference(project_root)

        print('\n' + '─' * 80)
        print('  STEP A-2 — Single-source baseline')
        print('─' * 80)
        single_source_path = prepare_single_source_baseline(project_root)

        print('\n' + '─' * 80)
        print('  STEP A-3 — Individual aligned source models')
        print('─' * 80)
        ind_paths = prepare_individual_source_models(
            feature_reference_path, project_root
        )

    # Step B
    print('\n' + '─' * 80)
    print('  STEP B — Data-Efficiency Sweep')
    print('─' * 80)
    run_data_efficiency_sweep(
        target_building, single_source_path, ind_paths, project_root
    )

    # Step C
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
        'data_efficiency_ensembletransfer.csv',
    ):
        full_path = os.path.join(results_dir, fname)
        status = '✓' if os.path.exists(full_path) else '✗ MISSING'
        print(f'  {status}  results/experiments/{EXPERIMENT_NAME}/{fname}')
    print('=' * 80)


if __name__ == '__main__':
    main()
