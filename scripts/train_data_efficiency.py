"""
Generalised Data Efficiency Training Script

Trains pre-transfer and transfer models with varying amounts of data for a given
ExperimentConfig. Supports all 6 experiment categories:

  rat_education, rat_education_new, eagle_education, lamb_education,
  office_any, lodging_any

Models are saved under:
  models/experiments/{experiment_name}/data_efficiency/

Usage (programmatic):
    from train_data_efficiency import ExperimentConfig, train_data_efficiency
    cfg = ExperimentConfig(
        name='eagle_education',
        source_building='Eagle_education_Raul',
        target_building='Eagle_education_XYZ',
        site_id='Eagle',
        building_type='Education',
    )
    train_data_efficiency(cfg, baseline_model_path)

Usage (CLI – runs all experiments from building_selections.csv):
    python train_data_efficiency.py
"""

import sys
import os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import glob
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

from src.train_pretransfer import train_pretransfer
from src.train_transfer import train_transfer
from src.train_frozen_backbone import train_frozen_backbone
from src.train_adapter import train_adapter


@dataclass
class ExperimentConfig:
    """Configuration for a single data-efficiency experiment."""
    name: str                          # e.g. 'eagle_education'
    source_building: str               # Baseline was trained on this building
    target_building: str               # Data efficiency sweep target
    site_id: Optional[str]            # e.g. 'Eagle'; None means any site
    building_type: Optional[str]      # e.g. 'Education'; None means any type
    weeks_to_train: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 104]
    )


def train_data_efficiency(config: ExperimentConfig, baseline_model_path: str):
    """Train pre-transfer and transfer models with varying data amounts.

    Args:
        config: ExperimentConfig describing this experiment.
        baseline_model_path: Absolute path to the baseline checkpoint to use
                             for transfer learning.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    data_efficiency_dir = os.path.join(
        project_root, 'models', 'experiments', config.name, 'data_efficiency'
    )
    os.makedirs(data_efficiency_dir, exist_ok=True)

    # Staging dir where train_pretransfer/train_transfer save checkpoints
    staging_dir = os.path.join(project_root, 'models', 'experiments', config.name)

    print("\n" + "=" * 80)
    print("  DATA EFFICIENCY TRAINING")
    print("=" * 80)
    print(f"  Experiment   : {config.name}")
    print(f"  Source       : {config.source_building}")
    print(f"  Target       : {config.target_building}")
    print(f"  Baseline     : {os.path.basename(baseline_model_path)}")
    print(f"  Data amounts : {config.weeks_to_train} weeks")
    print("=" * 80)

    for weeks in config.weeks_to_train:
        print(f"\n{'#' * 80}")
        print(f"  TRAINING WITH {weeks} WEEK(S) OF DATA  [{config.name}]")
        print(f"{'#' * 80}")

        # ---------------------------------------------------------- #
        # 1. Pre-Transfer (train from scratch)
        # ---------------------------------------------------------- #
        print(f"\n[1/2] Pre-Transfer ({weeks} weeks, from scratch)...")
        try:
            _, pretransfer_results = train_pretransfer(
                target_building=config.target_building,
                epochs=100,
                seq_length=24,
                data_limit_weeks=weeks,
                architecture_match=baseline_model_path,
                site_id=config.site_id,
                building_type=config.building_type,
                experiment_name=config.name,
            )

            # Rename freshly saved checkpoint to include week count and move to data_efficiency/
            pattern = os.path.join(
                staging_dir, f'pretransfer_{config.target_building[:15]}_*.ckpt'
            )
            fresh = glob.glob(pattern)
            if fresh:
                latest = max(fresh, key=os.path.getmtime)
                epoch_part = os.path.basename(latest).split('epoch=')[1]
                new_name = (
                    f'pretransfer_{config.target_building[:15]}'
                    f'_{weeks}week_epoch={epoch_part}'
                )
                shutil.move(latest, os.path.join(data_efficiency_dir, new_name))
                print(f"  ✓ Saved: {new_name}")

            print(f"  ✓ Pre-Transfer ({weeks} weeks) complete")
            print(f"    Test RMSE: {pretransfer_results[0]['test_rmse']:.4f}")
            print(f"    Test MAE:  {pretransfer_results[0]['test_mae']:.4f}")

        except Exception as e:
            print(f"  ✗ Pre-Transfer ({weeks} weeks) FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

        # ---------------------------------------------------------- #
        # 2. Transfer / Full Fine-Tuning (fine-tune from baseline)
        # ---------------------------------------------------------- #
        print(f"\n[2/4] Full Fine-Tuning ({weeks} weeks, all params trainable)...")
        try:
            _, transfer_results = train_transfer(
                source_building=config.source_building,
                target_building=config.target_building,
                source_model_path=baseline_model_path,
                epochs=50,
                seq_length=24,
                data_limit_weeks=weeks,
                site_id=config.site_id,
                building_type=config.building_type,
                experiment_name=config.name,
            )

            # Rename freshly saved checkpoint and move to data_efficiency/
            pattern = os.path.join(
                staging_dir,
                f'transfer_{config.source_building[:15]}_{config.target_building[:15]}_*.ckpt'
            )
            fresh = glob.glob(pattern)
            if fresh:
                latest = max(fresh, key=os.path.getmtime)
                epoch_part = os.path.basename(latest).split('epoch=')[1]
                new_name = (
                    f'transfer_{config.target_building[:15]}'
                    f'_{weeks}week_epoch={epoch_part}'
                )
                shutil.move(latest, os.path.join(data_efficiency_dir, new_name))
                print(f"  ✓ Saved: {new_name}")

            print(f"  ✓ Full Fine-Tuning ({weeks} weeks) complete")
            print(f"    Test RMSE: {transfer_results[0]['test_rmse']:.4f}")
            print(f"    Test MAE:  {transfer_results[0]['test_mae']:.4f}")

        except Exception as e:
            print(f"  ✗ Full Fine-Tuning ({weeks} weeks) FAILED: {e}")
            import traceback; traceback.print_exc()

        # ---------------------------------------------------------- #
        # 3. Frozen Backbone
        # ---------------------------------------------------------- #
        print(f"\n[3/4] Frozen Backbone ({weeks} weeks, encoder locked)...")
        try:
            _, frozen_results = train_frozen_backbone(
                source_building=config.source_building,
                target_building=config.target_building,
                source_model_path=baseline_model_path,
                epochs=50,
                seq_length=24,
                data_limit_weeks=weeks,
                site_id=config.site_id,
                building_type=config.building_type,
                experiment_name=config.name,
            )

            pattern = os.path.join(
                staging_dir,
                f'frozen_{config.source_building[:15]}_{config.target_building[:15]}_*.ckpt'
            )
            fresh = glob.glob(pattern)
            if fresh:
                latest = max(fresh, key=os.path.getmtime)
                epoch_part = os.path.basename(latest).split('epoch=')[1]
                new_name = (
                    f'frozen_{config.target_building[:15]}'
                    f'_{weeks}week_epoch={epoch_part}'
                )
                shutil.move(latest, os.path.join(data_efficiency_dir, new_name))
                print(f"  ✓ Saved: {new_name}")

            print(f"  ✓ Frozen Backbone ({weeks} weeks) complete")
            print(f"    Test RMSE: {frozen_results[0]['test_rmse']:.4f}")
            print(f"    Test MAE:  {frozen_results[0]['test_mae']:.4f}")

        except Exception as e:
            print(f"  ✗ Frozen Backbone ({weeks} weeks) FAILED: {e}")
            import traceback; traceback.print_exc()

        # ---------------------------------------------------------- #
        # 4. Adapter Layers
        # ---------------------------------------------------------- #
        print(f"\n[4/4] Adapter Layers ({weeks} weeks, bottleneck=32)...")
        try:
            _, adapter_results = train_adapter(
                source_building=config.source_building,
                target_building=config.target_building,
                source_model_path=baseline_model_path,
                epochs=50,
                seq_length=24,
                data_limit_weeks=weeks,
                site_id=config.site_id,
                building_type=config.building_type,
                experiment_name=config.name,
                adapter_bottleneck=32,
            )

            pattern = os.path.join(
                staging_dir,
                f'adapter_{config.source_building[:15]}_{config.target_building[:15]}_*.ckpt'
            )
            fresh = glob.glob(pattern)
            if fresh:
                latest = max(fresh, key=os.path.getmtime)
                epoch_part = os.path.basename(latest).split('epoch=')[1]
                new_name = (
                    f'adapter_{config.target_building[:15]}'
                    f'_{weeks}week_epoch={epoch_part}'
                )
                shutil.move(latest, os.path.join(data_efficiency_dir, new_name))
                print(f"  ✓ Saved: {new_name}")

            print(f"  ✓ Adapter Layers ({weeks} weeks) complete")
            print(f"    Test RMSE: {adapter_results[0]['test_rmse']:.4f}")
            print(f"    Test MAE:  {adapter_results[0]['test_mae']:.4f}")

        except Exception as e:
            print(f"  ✗ Adapter Layers ({weeks} weeks) FAILED: {e}")
            import traceback; traceback.print_exc()

        print(f"\n{'=' * 80}")
        print(f"  {weeks} WEEK(S) TRAINING COMPLETE  [{config.name}]")
        print(f"{'=' * 80}\n")

    print("\n" + "=" * 80)
    print(f"  ALL DATA EFFICIENCY TRAINING COMPLETE  [{config.name}]")
    print("=" * 80)
    print(f"  Models saved to: {data_efficiency_dir}")
    print("=" * 80)


# --------------------------------------------------------------------------- #
# Legacy wrapper kept for backward compatibility
# --------------------------------------------------------------------------- #
def train_all_data_efficiency_models():
    """Legacy entry point: runs the rat_education experiment (Colin→Denise)."""
    project_root = os.path.dirname(os.path.abspath(__file__))

    baseline_ckpts = glob.glob(
        os.path.join(project_root, 'models', 'experiments', 'rat_education', 'baseline_*.ckpt')
    )
    if not baseline_ckpts:
        print("ERROR: No baseline model found in models/experiments/rat_education/")
        print("Run run_experiment_suite.py or train src/train_baseline.py first.")
        return

    baseline_model_path = max(baseline_ckpts, key=os.path.getmtime)

    cfg = ExperimentConfig(
        name='rat_education',
        source_building='Rat_education_Colin',
        target_building='Rat_education_Denise',
        site_id='Rat',
        building_type='Education',
    )
    train_data_efficiency(cfg, baseline_model_path)


# --------------------------------------------------------------------------- #
# CLI: run all experiments from building_selections.csv
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import pandas as pd

    project_root = os.path.dirname(os.path.abspath(__file__))
    selections_path = os.path.join(
        project_root, 'results', 'experiments', 'building_selections.csv'
    )

    if not os.path.exists(selections_path):
        print(
            "ERROR: building_selections.csv not found.\n"
            "Run discover_buildings.py first:\n"
            "  python discover_buildings.py"
        )
        sys.exit(1)

    selections = pd.read_csv(selections_path)
    print(f"Loaded {len(selections)} experiment(s) from {selections_path}")

    for _, row in selections.iterrows():
        exp_name = row['experiment_name']

        # Locate the baseline model for this experiment
        exp_model_dir = os.path.join(
            project_root, 'models', 'experiments', exp_name
        )
        baseline_ckpts = glob.glob(os.path.join(exp_model_dir, 'baseline_*.ckpt'))

        if not baseline_ckpts:
            print(f"\n[SKIP] {exp_name}: No baseline model found in {exp_model_dir}")
            print("  Train the baseline first via run_experiment_suite.py")
            continue

        baseline_path = max(baseline_ckpts, key=os.path.getmtime)

        site_val = str(row['site_id'])
        type_val = str(row['building_type'])

        cfg = ExperimentConfig(
            name=exp_name,
            source_building=row['source_building'],
            target_building=row['target_building'],
            site_id=None if site_val in ('Any', 'nan') else site_val,
            building_type=None if type_val in ('Any', 'nan') else type_val,
        )

        train_data_efficiency(cfg, baseline_path)

