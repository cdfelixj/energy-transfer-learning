"""
Multi-Transfer N-Source Ablation Experiment

Answers: How does multi-transfer performance scale with the number and
diversity of source buildings?  Is there a point of diminishing returns?

Experiment design
─────────────────
  Target building: Eagle_education_Brooke  (Eagle / Education)
  Fixed data limit: 8 weeks  (SNAPSHOT_WEEK for the scaling curve plot)

  Pool builds progressively — each N adds one more diverse building:
    N=1  Eagle/Samantha                              (same site + type)
    N=2  + Rat/Colin                                 (diff site, same type)
    N=3  + Lamb/Lucas                                (diff site, same type)
    N=4  + Hog/Miriam                                (diff site + type: Office)
    N=5  + Robin/Celia                               (diff site + type: Lodging)

  For each N a separate multi-source baseline is trained (or reused) and then
  fine-tuned on the target using the standard Multi-Transfer protocol.

Architecture notes
──────────────────
  N=1,2 pool ∩ features = 31  (no Lamb)
  N=3,4,5 pool ∩ features = 29  (Lamb constrains to 29)
  train_multi_transfer handles the target-side truncation automatically.

File outputs
────────────
  models/experiments/multitransfer_ablation/
    baseline_n1_{...}.ckpt
    baseline_n2_{...}.ckpt
    baseline_n3_{...}.ckpt
    baseline_n4_{...}.ckpt
    baseline_n5_{...}.ckpt          (copy of multi_transfer baseline_multi_5buildings)
    data_efficiency/
      pretransfer_Eagle_educati_{N}week_*.ckpt   (8 files — reused from multi_transfer)
      multitransfer_n1_Eagle_educati_{N}week_*.ckpt  …n5…

  results/experiments/multitransfer_ablation/
    data_efficiency_pretransfer.csv
    data_efficiency_multitransfer_n1.csv  … n5 …

Usage
─────
  python run_multitransfer_ablation_experiment.py
  python run_multitransfer_ablation_experiment.py --skip-baselines
  python run_multitransfer_ablation_experiment.py --eval-only
"""

import sys
import os
import argparse
import glob
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train_baseline import train_baseline
from src.train_pretransfer import train_pretransfer
from src.train_multi_transfer import train_multi_transfer
from evaluate_all_models import evaluate_data_efficiency

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = 'multitransfer_ablation'

DEFAULT_TARGET_BUILDING = 'Eagle_education_Brooke'
TARGET_SITE             = 'Eagle'
TARGET_TYPE             = 'Education'

# The single-source building (used for pretransfer architecture alignment)
SINGLE_SOURCE_BUILDING  = 'Eagle_education_Samantha'

WEEKS = [1, 2, 4, 8, 16, 32, 64, 104]

# Progressive multi-source pools: each N adds one more diverse building.
# Keys are the strategy name suffix used in checkpoint/CSV filenames.
N_POOLS = {
    1:  ['Eagle_education_Samantha'],
    2:  ['Eagle_education_Samantha', 'Rat_education_Colin'],
    3:  ['Eagle_education_Samantha', 'Rat_education_Colin', 'Lamb_education_Lucas'],
    4:  ['Eagle_education_Samantha', 'Rat_education_Colin', 'Lamb_education_Lucas',
         'Hog_office_Miriam'],
    5:  ['Eagle_education_Samantha', 'Rat_education_Colin', 'Lamb_education_Lucas',
         'Hog_office_Miriam', 'Robin_lodging_Celia'],
    # N=10,15: add more Eagle/Rat buildings (same sites as N=5 pool) so that the
    # feature intersection stays fixed at 29 (Lamb remains the constraining site).
    10: ['Eagle_education_Samantha', 'Rat_education_Colin', 'Lamb_education_Lucas',
         'Hog_office_Miriam', 'Robin_lodging_Celia',
         'Eagle_education_Luther', 'Eagle_education_Lino', 'Eagle_education_Jewell',
         'Rat_education_Theo',   'Eagle_education_Shanna'],
    15: ['Eagle_education_Samantha', 'Rat_education_Colin', 'Lamb_education_Lucas',
         'Hog_office_Miriam', 'Robin_lodging_Celia',
         'Eagle_education_Luther', 'Eagle_education_Lino', 'Eagle_education_Jewell',
         'Rat_education_Theo',   'Eagle_education_Shanna',
         'Eagle_education_Wesley', 'Eagle_education_Raul', 'Eagle_education_Shante',
         'Eagle_education_Sherrill', 'Eagle_education_Peter'],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Prepare N baselines
# ─────────────────────────────────────────────────────────────────────────────

def prepare_baselines(project_root):
    """Train N=2,3,4 baselines; reuse existing for N=1 and N=5.

    Returns a dict {n: checkpoint_path}.
    """
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    baseline_paths = {}

    for n, pool in N_POOLS.items():
        dest = os.path.join(exp_dir, f'baseline_n{n}.ckpt')

        # Already prepared?
        if os.path.exists(dest):
            print(f"\n[Baseline N={n}] Already exists — skipping.")
            baseline_paths[n] = dest
            continue

        if n == 1:
            # Reuse Eagle/Samantha single-source from multi_transfer or eagle_education
            src = _latest(os.path.join(
                project_root, 'models', 'experiments', 'multi_transfer',
                'baseline_single_*.ckpt'
            )) or _latest(os.path.join(
                project_root, 'models', 'experiments', 'eagle_education',
                'baseline_*.ckpt'
            ))
            if src:
                shutil.copy2(src, dest)
                print(f"\n[Baseline N=1] Reused from existing experiment.")
                print(f"  → baseline_n1.ckpt")
            else:
                print(f"\n[Baseline N=1] Training {pool[0]}...")
                train_baseline(
                    building_ids=pool,
                    epochs=50,
                    seq_length=168,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                fresh = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
                if not fresh:
                    raise RuntimeError(f'Baseline N=1 training failed.')
                os.rename(fresh, dest)
                print(f"  → baseline_n1.ckpt")

        elif n == 5:
            # Reuse multi_transfer 5-building baseline
            src = _latest(os.path.join(
                project_root, 'models', 'experiments', 'multi_transfer',
                'baseline_multi_*.ckpt'
            ))
            if src:
                shutil.copy2(src, dest)
                print(f"\n[Baseline N=5] Reused from multi_transfer experiment.")
                print(f"  → baseline_n5.ckpt")
            else:
                print(f"\n[Baseline N=5] Training 5-building pool...")
                train_baseline(
                    building_ids=pool,
                    epochs=50,
                    seq_length=168,
                    site_id=None,
                    building_type=None,
                    experiment_name=EXPERIMENT_NAME,
                )
                fresh = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
                if not fresh:
                    raise RuntimeError('Baseline N=5 training failed.')
                os.rename(fresh, dest)
                print(f"  → baseline_n5.ckpt")

        else:
            # Train new N=2, N=3, N=4 baselines
            print(f"\n[Baseline N={n}] Training on {len(pool)} buildings:")
            for b in pool:
                print(f"  • {b}")
            train_baseline(
                building_ids=pool,
                epochs=50,
                seq_length=168,
                site_id=None,
                building_type=None,
                experiment_name=EXPERIMENT_NAME,
            )
            fresh = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
            if not fresh:
                raise RuntimeError(f'Baseline N={n} training failed.')
            os.rename(fresh, dest)
            print(f"  → baseline_n{n}.ckpt")

        baseline_paths[n] = dest

    return baseline_paths


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Data-efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(target_building, baseline_paths, project_root):
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    de_dir  = os.path.join(exp_dir, 'data_efficiency')
    os.makedirs(de_dir, exist_ok=True)

    tgt = target_building[:15]

    # ── Pretransfer (scratch) — shared across all N, feature-aligned to N=1 ─────
    for weeks in WEEKS:
        if _latest(os.path.join(de_dir, f'pretransfer_{tgt}_{weeks}week_*.ckpt')):
            print(f"\n[Pre-Transfer {weeks}w] Already trained, skipping.")
            continue

        # Try to reuse pretransfer from multi_transfer experiment
        mt_pt = _latest(os.path.join(
            project_root, 'models', 'experiments', 'multi_transfer',
            'data_efficiency', f'pretransfer_{tgt}_{weeks}week_*.ckpt'
        ))
        if mt_pt:
            shutil.copy2(mt_pt, os.path.join(de_dir, os.path.basename(mt_pt)))
            print(f"\n[Pre-Transfer {weeks}w] Reused from multi_transfer.")
            continue

        print(f"\n[Pre-Transfer {weeks}w] Training from scratch...")
        try:
            train_pretransfer(
                target_building=target_building,
                epochs=100,
                seq_length=24,
                data_limit_weeks=weeks,
                architecture_match=baseline_paths[1],
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
            print(f"  ✗ Pre-Transfer {weeks}w FAILED: {exc}")
            import traceback; traceback.print_exc()

    # ── Multitransfer_nN for each N ──────────────────────────────────────────
    for n, baseline_path in baseline_paths.items():
        strategy_key = f'multitransfer_n{n}'
        pool = N_POOLS[n]

        for weeks in WEEKS:
            ckpt_pattern = os.path.join(de_dir, f'{strategy_key}_{tgt}_{weeks}week_*.ckpt')
            if _latest(ckpt_pattern):
                print(f"\n[{strategy_key} {weeks}w] Already trained, skipping.")
                continue

            print(f"\n{'#'*60}")
            print(f"  {strategy_key} | {weeks} wk(s) | N={n} sources: {', '.join(b[:18] for b in pool)}")
            print(f"{'#'*60}")

            try:
                train_multi_transfer(
                    target_building=target_building,
                    multi_baseline_model_path=baseline_path,
                    epochs=50,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    site_id=TARGET_SITE,
                    building_type=TARGET_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                # train_multi_transfer saves as multitransfer_{tgt[:15]}_epoch=...
                fresh = _latest(os.path.join(exp_dir, f'multitransfer_{tgt}_*.ckpt'))
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'{strategy_key}_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
            except Exception as exc:
                print(f"  ✗ {strategy_key} {weeks}w FAILED: {exc}")
                import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Evaluate & save CSVs
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(target_building, project_root):
    results_dir = os.path.join(
        project_root, 'results', 'experiments', EXPERIMENT_NAME
    )
    os.makedirs(results_dir, exist_ok=True)

    strategies = ['pretransfer'] + [f'multitransfer_n{n}' for n in sorted(N_POOLS.keys())]

    for model_type in strategies:
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
        description='N-Source Multi-Transfer Ablation Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target', default=DEFAULT_TARGET_BUILDING)
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Fail if baseline_n{1..5}.ckpt not found')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip all training, evaluate only')
    args = parser.parse_args()
    target_building = args.target
    project_root = get_project_root()

    print('\n' + '=' * 80)
    print('  MULTI-TRANSFER N-SOURCE ABLATION EXPERIMENT')
    print('=' * 80)
    print(f'  Target: {target_building}')
    for n, pool in N_POOLS.items():
        print(f'  N={n}: {", ".join(b[:22] for b in pool)}')
    print(f'  Weeks: {WEEKS}')
    print('=' * 80)

    if args.eval_only:
        print('\n[--eval-only]')
        evaluate_and_save(target_building, project_root)
        return

    if args.skip_baselines:
        exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
        baseline_paths = {}
        for n in N_POOLS:
            path = os.path.join(exp_dir, f'baseline_n{n}.ckpt')
            if not os.path.exists(path):
                print(f'ERROR: --skip-baselines set but {path} not found.')
                sys.exit(1)
            baseline_paths[n] = path
        print(f'\n[--skip-baselines] Using existing baselines.')
    else:
        print('\n' + '─' * 80)
        print('  STEP A — N-Source Baselines')
        print('─' * 80)
        baseline_paths = prepare_baselines(project_root)

    print('\n' + '─' * 80)
    print('  STEP B — Data-Efficiency Sweep')
    print('─' * 80)
    run_data_efficiency_sweep(target_building, baseline_paths, project_root)

    print('\n' + '─' * 80)
    print('  STEP C — Evaluation')
    print('─' * 80)
    evaluate_and_save(target_building, project_root)

    print('\n' + '=' * 80)
    print('  COMPLETE — Output files:')
    print('=' * 80)
    results_dir = os.path.join(project_root, 'results', 'experiments', EXPERIMENT_NAME)
    for key in ['pretransfer'] + [f'multitransfer_n{n}' for n in sorted(N_POOLS.keys())]:
        fpath = os.path.join(results_dir, f'data_efficiency_{key}.csv')
        status = '✓' if os.path.exists(fpath) else '✗ MISSING'
        print(f'  {status}  results/experiments/{EXPERIMENT_NAME}/data_efficiency_{key}.csv')
    print('=' * 80)


if __name__ == '__main__':
    main()
