"""
Experiment Suite Orchestrator

Runs all 6 data-efficiency experiments end-to-end:

  1. rat_education     - Rat / Education  (Colin → Denise)  [skip training, models exist]
  2. rat_education_new - Rat / Education  (auto-selected pair, excl. Colin/Denise)
  3. eagle_education   - Eagle / Education
  4. lamb_education    - Lamb / Education
  5. office_any        - Any site / Office
  6. lodging_any       - Any site / Lodging/residential

For each experiment:
  Step 1 – Train baseline on source building (full data)
  Step 2 – Run data efficiency sweep (pre-transfer + transfer, 1–104 weeks)
  Step 3 – Run 4-model evaluation and save results

Prerequisites:
  1. Run discover_buildings.py to generate results/experiments/building_selections.csv
  2. Optionally set SKIP_EXISTING=True below to avoid re-training rat_education

Usage:
    python run_experiment_suite.py
    python run_experiment_suite.py --experiment eagle_education
"""

import sys
import os
import argparse
import glob
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train_baseline import train_baseline
from train_data_efficiency import ExperimentConfig, train_data_efficiency
from evaluate_all_models import evaluate_experiment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEEKS_TO_TRAIN = [1, 2, 4, 8, 16, 32, 64, 104]

# Set to True to skip training for experiments that already have a baseline
# e.g. rat_education already has a trained baseline in models/experiments/rat_education/
SKIP_EXISTING_BASELINE = True
SKIP_EXISTING_DATA_EFFICIENCY = True


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def load_selections(project_root):
    path = os.path.join(project_root, 'results', 'experiments', 'building_selections.csv')
    if not os.path.exists(path):
        print(
            "ERROR: building_selections.csv not found.\n"
            "Run discover_buildings.py first:\n"
            "  python discover_buildings.py"
        )
        sys.exit(1)
    return pd.read_csv(path)


def run_experiment(row, project_root):
    """Run a full experiment: baseline training + data efficiency + evaluation."""
    exp_name = row['experiment_name']
    source = row['source_building']
    target = row['target_building']
    site_val = str(row['site_id'])
    type_val = str(row['building_type'])
    site_id = None if site_val in ('Any', 'nan') else site_val
    building_type = None if type_val in ('Any', 'nan') else type_val

    exp_dir = os.path.join(project_root, 'models', 'experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'#' * 80}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Source: {source}  →  Target: {target}")
    print(f"  Site: {site_id or 'Any'}  |  Type: {building_type or 'Any'}")
    print(f"{'#' * 80}")

    # ------------------------------------------------------------------ #
    # Step 1: Train Baseline
    # ------------------------------------------------------------------ #
    existing_baselines = glob.glob(os.path.join(exp_dir, 'baseline_*.ckpt'))

    if existing_baselines and SKIP_EXISTING_BASELINE:
        baseline_path = max(existing_baselines, key=os.path.getmtime)
        print(f"\n[Step 1/3] Baseline already exists — skipping training.")
        print(f"  Using: {os.path.basename(baseline_path)}")
    else:
        print(f"\n[Step 1/3] Training baseline on {source} ...")
        _, _ = train_baseline(
            building_ids=[source],
            epochs=50,
            seq_length=168,
            site_id=site_id,
            building_type=building_type,
            experiment_name=exp_name,
        )
        existing_baselines = glob.glob(os.path.join(exp_dir, 'baseline_*.ckpt'))
        if not existing_baselines:
            print(f"  ERROR: Baseline checkpoint not found after training. Aborting {exp_name}.")
            return
        baseline_path = max(existing_baselines, key=os.path.getmtime)
        print(f"  ✓ Baseline saved: {os.path.basename(baseline_path)}")

    # ------------------------------------------------------------------ #
    # Step 2: Data Efficiency Sweep
    # ------------------------------------------------------------------ #
    de_dir = os.path.join(exp_dir, 'data_efficiency')
    existing_de = glob.glob(os.path.join(de_dir, '*.ckpt'))

    # Count unique weeks already trained for both model types
    trained_weeks = set()
    for f in existing_de:
        fname = os.path.basename(f)
        for part in fname.split('_'):
            if part.endswith('week'):
                try:
                    trained_weeks.add(int(part.replace('week', '')))
                except ValueError:
                    pass

    weeks_needed = [w for w in WEEKS_TO_TRAIN if w not in trained_weeks]

    if not weeks_needed and SKIP_EXISTING_DATA_EFFICIENCY:
        print(f"\n[Step 2/3] Data efficiency models already complete — skipping.")
    else:
        if SKIP_EXISTING_DATA_EFFICIENCY and trained_weeks:
            print(f"\n[Step 2/3] Partially complete. Already trained: {sorted(trained_weeks)} weeks.")
            print(f"  Training remaining: {weeks_needed} weeks...")
        else:
            print(f"\n[Step 2/3] Running data efficiency sweep ({WEEKS_TO_TRAIN} weeks)...")

        cfg = ExperimentConfig(
            name=exp_name,
            source_building=source,
            target_building=target,
            site_id=site_id,
            building_type=building_type,
            weeks_to_train=weeks_needed if SKIP_EXISTING_DATA_EFFICIENCY else WEEKS_TO_TRAIN,
        )
        train_data_efficiency(cfg, baseline_path)

    # ------------------------------------------------------------------ #
    # Step 3: Evaluation
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3/3] Evaluating all 4 model types for {exp_name} ...")
    evaluate_experiment(
        experiment_name=exp_name,
        source_building=source,
        target_building=target,
        site_id=site_id,
        building_type=building_type,
        baseline_model_path=baseline_path,
        weeks_list=WEEKS_TO_TRAIN,
    )

    print(f"\n✓ Experiment {exp_name} complete.")
    print(f"  Results: results/experiments/{exp_name}/")


def main():
    parser = argparse.ArgumentParser(description='Run the full experiment suite.')
    parser.add_argument(
        '--experiment', '-e',
        default=None,
        help='Run only a specific experiment by name (e.g. eagle_education). '
             'If omitted, all experiments are run.'
    )
    args = parser.parse_args()

    project_root = get_project_root()
    selections = load_selections(project_root)

    if args.experiment:
        mask = selections['experiment_name'] == args.experiment
        if not mask.any():
            print(f"ERROR: Experiment '{args.experiment}' not found in building_selections.csv")
            print(f"Available: {selections['experiment_name'].tolist()}")
            sys.exit(1)
        selections = selections[mask]

    print(f"\n{'=' * 80}")
    print(f"  RUNNING {len(selections)} EXPERIMENT(S)")
    print(f"{'=' * 80}")
    for _, row in selections.iterrows():
        print(f"  • {row['experiment_name']}: {row['source_building']} → {row['target_building']}")
    print(f"{'=' * 80}")

    for _, row in selections.iterrows():
        try:
            run_experiment(row, project_root)
        except Exception as e:
            import traceback
            print(f"\n✗ Experiment {row['experiment_name']} FAILED: {e}")
            traceback.print_exc()
            print("  Continuing with next experiment...")

    print(f"\n{'=' * 80}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 80}")
    print("Results saved under:")
    for _, row in selections.iterrows():
        print(f"  results/experiments/{row['experiment_name']}/")


if __name__ == '__main__':
    main()
