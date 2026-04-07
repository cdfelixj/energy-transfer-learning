"""
Cross-Type Transfer Experiment

Answers: Does the *type* of the source building matter for transfer learning?
Is a same-site, same-type source always the best, or can cross-type sources
still provide useful initialisations?

Three transfer variants are compared against training from scratch:

  transfer_samesite   : Eagle/Samantha → Eagle/Brooke  (same site, same type — tightest match)
  transfer_sametype   : Rat/Colin      → Eagle/Brooke  (diff site, same type — type match only)
  transfer_crosstype  : Hog/Miriam     → Eagle/Brooke  (diff site, diff type — no match)

Architecture note
─────────────────
  All three source buildings have input_size = 31.
  Target Eagle/Brooke also has input_size = 31.
  No feature truncation needed — weights transfer directly.

Baseline reuse
──────────────
  transfer_samesite  → multi_transfer/baseline_single_eagle_samantha.ckpt
  transfer_sametype  → rat_education/baseline_Rat_education_Colin*.ckpt
  transfer_crosstype → office_any/baseline_Hog_office_Miriam*.ckpt

  For transfer_sametype and transfer_crosstype the source building is from a
  different site/type than the target.  train_transfer is called with
  site_id=None, building_type=None so the validation step finds the source
  building (it is not filtered out).

File outputs
────────────
  models/experiments/cross_type_transfer/
    baseline_samesite_eagle_samantha.ckpt
    baseline_sametype_rat_colin.ckpt
    baseline_crosstype_hog_miriam.ckpt
    data_efficiency/
      pretransfer_Eagle_educati_{N}week_*.ckpt
      transfer_samesite_Eagle_educati_{N}week_*.ckpt
      transfer_sametype_Eagle_educati_{N}week_*.ckpt
      transfer_crosstype_Eagle_educati_{N}week_*.ckpt

  results/experiments/cross_type_transfer/
    data_efficiency_pretransfer.csv
    data_efficiency_transfer_samesite.csv
    data_efficiency_transfer_sametype.csv
    data_efficiency_transfer_crosstype.csv

Usage
─────
  python run_cross_type_experiment.py
  python run_cross_type_experiment.py --skip-baselines
  python run_cross_type_experiment.py --eval-only
"""

import sys
import os
import argparse
import glob
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train_baseline import train_baseline
from src.train_pretransfer import train_pretransfer
from src.train_transfer import train_transfer
from evaluate_all_models import evaluate_data_efficiency

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = 'cross_type_transfer'

DEFAULT_TARGET_BUILDING = 'Eagle_education_Brooke'
TARGET_SITE             = 'Eagle'
TARGET_TYPE             = 'Education'

WEEKS = [1, 2, 4, 8, 16, 32, 64, 104]

# Three source buildings and their transfer strategy keys
TRANSFER_VARIANTS = {
    'transfer_samesite':  {
        'source_building': 'Eagle_education_Samantha',
        'source_site':     'Eagle',
        'source_type':     'Education',
        'load_site':       'Eagle',    # site_id passed to train_transfer
        'load_type':       'Education',
        'description':     'Same site + same type (tightest match)',
        'baseline_file':   'baseline_samesite_eagle_samantha.ckpt',
    },
    'transfer_sametype': {
        'source_building': 'Rat_education_Colin',
        'source_site':     'Rat',
        'source_type':     'Education',
        'load_site':       None,       # must load all buildings to find Colin + Brooke
        'load_type':       None,
        'description':     'Different site, same type',
        'baseline_file':   'baseline_sametype_rat_colin.ckpt',
    },
    'transfer_crosstype': {
        'source_building': 'Hog_office_Miriam',
        'source_site':     'Hog',
        'source_type':     'Office',
        'load_site':       None,       # must load all buildings to find Miriam + Brooke
        'load_type':       None,
        'description':     'Different site + different type',
        'baseline_file':   'baseline_crosstype_hog_miriam.ckpt',
    },
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
# Step A: Prepare source baselines
# ─────────────────────────────────────────────────────────────────────────────

def prepare_baselines(project_root):
    """Collect or train the three source baselines.

    Returns dict {strategy_key: checkpoint_path}.
    """
    exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    baseline_paths = {}

    # ── same-site: reuse multi_transfer or eagle_education ───────────────
    key = 'transfer_samesite'
    meta = TRANSFER_VARIANTS[key]
    dest = os.path.join(exp_dir, meta['baseline_file'])
    if os.path.exists(dest):
        print(f"\n[{key}] Baseline already exists — skipping.")
    else:
        src = _latest(os.path.join(
            project_root, 'models', 'experiments', 'multi_transfer',
            'baseline_single_*.ckpt'
        )) or _latest(os.path.join(
            project_root, 'models', 'experiments', 'eagle_education',
            'baseline_*.ckpt'
        ))
        if src:
            shutil.copy2(src, dest)
            print(f"\n[{key}] Reused Eagle/Samantha baseline from existing experiment.")
        else:
            print(f"\n[{key}] Training {meta['source_building']}...")
            train_baseline(
                building_ids=[meta['source_building']],
                epochs=50, seq_length=168,
                site_id=meta['source_site'],
                building_type=meta['source_type'],
                experiment_name=EXPERIMENT_NAME,
            )
            fresh = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
            if not fresh:
                raise RuntimeError(f'{key} baseline training failed.')
            os.rename(fresh, dest)
    baseline_paths[key] = dest

    # ── same-type: reuse rat_education baseline ───────────────────────────
    key = 'transfer_sametype'
    meta = TRANSFER_VARIANTS[key]
    dest = os.path.join(exp_dir, meta['baseline_file'])
    if os.path.exists(dest):
        print(f"\n[{key}] Baseline already exists — skipping.")
    else:
        src = _latest(os.path.join(
            project_root, 'models', 'experiments', 'rat_education',
            'baseline_*.ckpt'
        ))
        if src:
            shutil.copy2(src, dest)
            print(f"\n[{key}] Reused Rat/Colin baseline from rat_education experiment.")
        else:
            print(f"\n[{key}] Training {meta['source_building']}...")
            train_baseline(
                building_ids=[meta['source_building']],
                epochs=50, seq_length=168,
                site_id=meta['source_site'],
                building_type=meta['source_type'],
                experiment_name=EXPERIMENT_NAME,
            )
            fresh = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
            if not fresh:
                raise RuntimeError(f'{key} baseline training failed.')
            os.rename(fresh, dest)
    baseline_paths[key] = dest

    # ── cross-type: reuse office_any baseline ─────────────────────────────
    key = 'transfer_crosstype'
    meta = TRANSFER_VARIANTS[key]
    dest = os.path.join(exp_dir, meta['baseline_file'])
    if os.path.exists(dest):
        print(f"\n[{key}] Baseline already exists — skipping.")
    else:
        src = _latest(os.path.join(
            project_root, 'models', 'experiments', 'office_any',
            'baseline_Hog_office_Miriam*.ckpt'
        ))
        if src:
            shutil.copy2(src, dest)
            print(f"\n[{key}] Reused Hog/Miriam baseline from office_any experiment.")
        else:
            print(f"\n[{key}] Training {meta['source_building']}...")
            train_baseline(
                building_ids=[meta['source_building']],
                epochs=50, seq_length=168,
                site_id=meta['source_site'],
                building_type=meta['source_type'],
                experiment_name=EXPERIMENT_NAME,
            )
            fresh = _latest(os.path.join(exp_dir, 'baseline_*.ckpt'))
            if not fresh:
                raise RuntimeError(f'{key} baseline training failed.')
            os.rename(fresh, dest)
    baseline_paths[key] = dest

    return baseline_paths


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Data-efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency_sweep(target_building, baseline_paths, project_root):
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
            print(f"\n[1/4] Pre-Transfer {weeks}w — already trained, skipping.")
        else:
            # Use samesite baseline for architecture alignment (input_size=31)
            arch_ref = baseline_paths['transfer_samesite']

            # Reuse from multi_transfer if available (same target + same architecture)
            mt_pt = _latest(os.path.join(
                project_root, 'models', 'experiments', 'multi_transfer',
                'data_efficiency', f'pretransfer_{tgt}_{weeks}week_*.ckpt'
            ))
            if mt_pt:
                shutil.copy2(mt_pt, os.path.join(de_dir, os.path.basename(mt_pt)))
                print(f"\n[1/4] Pre-Transfer {weeks}w — reused from multi_transfer.")
            else:
                print(f"\n[1/4] Pre-Transfer {weeks}w — training from scratch...")
                try:
                    train_pretransfer(
                        target_building=target_building,
                        epochs=100,
                        seq_length=24,
                        data_limit_weeks=weeks,
                        architecture_match=arch_ref,
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

        # ── 2/3/4. Each transfer variant ─────────────────────────────────
        for idx, (strategy_key, meta) in enumerate(TRANSFER_VARIANTS.items(), start=2):
            source_building = meta['source_building']
            baseline_path   = baseline_paths[strategy_key]

            if _latest(os.path.join(de_dir, f'{strategy_key}_{tgt}_{weeks}week_*.ckpt')):
                print(f"\n[{idx}/4] {strategy_key} {weeks}w — already trained, skipping.")
                continue

            print(f"\n[{idx}/4] {strategy_key} {weeks}w — {meta['description']}")
            print(f"  Source: {source_building}")
            try:
                train_transfer(
                    source_building=source_building,
                    target_building=target_building,
                    source_model_path=baseline_path,
                    epochs=50,
                    seq_length=24,
                    data_limit_weeks=weeks,
                    site_id=meta['load_site'],
                    building_type=meta['load_type'],
                    experiment_name=EXPERIMENT_NAME,
                )
                # train_transfer saves as transfer_{src[:15]}_{tgt[:15]}_epoch=...
                src_prefix = source_building[:15]
                fresh = _latest(os.path.join(
                    exp_dir, f'transfer_{src_prefix}_{tgt}_*.ckpt'
                ))
                if fresh:
                    epoch_part = os.path.basename(fresh).split('epoch=')[1]
                    dest_name = f'{strategy_key}_{tgt}_{weeks}week_epoch={epoch_part}'
                    shutil.move(fresh, os.path.join(de_dir, dest_name))
                    print(f"  ✓ Saved: {dest_name}")
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

    strategies = ['pretransfer', 'transfer_samesite', 'transfer_sametype', 'transfer_crosstype']

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
        description='Cross-Type Transfer Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target', default=DEFAULT_TARGET_BUILDING)
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    target_building = args.target
    project_root = get_project_root()

    print('\n' + '=' * 80)
    print('  CROSS-TYPE TRANSFER EXPERIMENT')
    print('=' * 80)
    print(f'  Target: {target_building}  (Eagle / Education)')
    for key, meta in TRANSFER_VARIANTS.items():
        print(f'  {key:22s}: {meta["source_building"]:30s}  [{meta["description"]}]')
    print(f'  Weeks: {WEEKS}')
    print('=' * 80)

    if args.eval_only:
        evaluate_and_save(target_building, project_root)
        return

    if args.skip_baselines:
        exp_dir = os.path.join(project_root, 'models', 'experiments', EXPERIMENT_NAME)
        baseline_paths = {}
        for key, meta in TRANSFER_VARIANTS.items():
            path = os.path.join(exp_dir, meta['baseline_file'])
            if not os.path.exists(path):
                print(f'ERROR: {path} not found.')
                sys.exit(1)
            baseline_paths[key] = path
    else:
        print('\n' + '─' * 80)
        print('  STEP A — Source Baselines')
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
    for key in ['pretransfer', 'transfer_samesite', 'transfer_sametype', 'transfer_crosstype']:
        fpath = os.path.join(results_dir, f'data_efficiency_{key}.csv')
        status = '✓' if os.path.exists(fpath) else '✗ MISSING'
        print(f'  {status}  results/experiments/{EXPERIMENT_NAME}/data_efficiency_{key}.csv')
    print('=' * 80)


if __name__ == '__main__':
    main()
