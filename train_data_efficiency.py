"""
Automated Data Efficiency Training Script

Trains pre-transfer and transfer models with varying amounts of data:
- 1 week, 2 weeks, 4 weeks, 8 weeks, 16 weeks

This allows evaluation of how model performance scales with training data availability.
Models are saved with week count in filename for easy identification.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import glob
from src.train_pretransfer import train_pretransfer
from src.train_transfer import train_transfer


def train_all_data_efficiency_models():
    """Train pre-transfer and transfer models with different data amounts"""
    
    # Configuration
    target_building = 'Rat_education_Denise'
    source_building = 'Rat_education_Colin'  # Baseline was trained on Colin
    weeks_to_train = [1, 2, 4, 8, 16, 32, 64, 104]  # Different data amounts to test (104 weeks = 2 years)
    
    # Get project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    # Find baseline model for transfer learning
    baseline_models = glob.glob(os.path.join(project_root, 'models', 'baseline_*.ckpt'))
    if not baseline_models:
        print("ERROR: No baseline model found!")
        print("Please run: python src/train_baseline.py")
        return
    
    baseline_model_path = max(baseline_models, key=os.path.getmtime)
    print(f"Using baseline model: {os.path.basename(baseline_model_path)}")
    
    # Create data_efficiency directory
    data_efficiency_dir = os.path.join(project_root, 'models', 'data_efficiency')
    os.makedirs(data_efficiency_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("  DATA EFFICIENCY TRAINING")
    print("="*80)
    print(f"\nTarget Building: {target_building}")
    print(f"Data amounts to train: {weeks_to_train} weeks")
    print(f"Models: Pre-Transfer (from scratch) + Transfer (fine-tuned)")
    print("="*80)
    
    # Train models for each data amount
    for weeks in weeks_to_train:
        print(f"\n{'#'*80}")
        print(f"  TRAINING WITH {weeks} WEEK(S) OF DATA")
        print(f"{'#'*80}")
        
        # 1. Train Pre-Transfer Model (from scratch)
        print(f"\n[1/2] Training Pre-Transfer model ({weeks} weeks, from scratch)...")
        try:
            pretransfer_model, pretransfer_results = train_pretransfer(
                target_building=target_building,
                epochs=100,
                seq_length=24,
                data_limit_weeks=weeks,
                architecture_match=baseline_model_path  # Match architecture for fair comparison
            )
            
            # Find the saved checkpoint and rename it with week count
            pretransfer_checkpoints = glob.glob(os.path.join(project_root, 'models', f'pretransfer_{target_building[:15]}_*.ckpt'))
            if pretransfer_checkpoints:
                latest_checkpoint = max(pretransfer_checkpoints, key=os.path.getmtime)
                new_name = f'pretransfer_{target_building[:15]}_{weeks}week_epoch={latest_checkpoint.split("epoch=")[1]}'
                new_path = os.path.join(data_efficiency_dir, new_name)
                
                # Move to data_efficiency folder
                import shutil
                shutil.move(latest_checkpoint, new_path)
                print(f"✓ Saved: {new_name}")
            
            print(f"✓ Pre-Transfer ({weeks} weeks) complete")
            print(f"  Test RMSE: {pretransfer_results[0]['test_rmse']:.4f}")
            print(f"  Test MAE: {pretransfer_results[0]['test_mae']:.4f}")
            
        except Exception as e:
            print(f"✗ Pre-Transfer ({weeks} weeks) FAILED: {e}")
            continue
        
        # 2. Train Transfer Model (fine-tuned from baseline)
        print(f"\n[2/2] Training Transfer model ({weeks} weeks, fine-tuned)...")
        try:
            transfer_model, transfer_results = train_transfer(
                source_building=source_building,
                target_building=target_building,
                source_model_path=baseline_model_path,
                epochs=50,  # Transfer learning typically needs fewer epochs
                seq_length=24,
                data_limit_weeks=weeks
            )
            
            # Find the saved checkpoint and rename it with week count
            transfer_checkpoints = glob.glob(os.path.join(project_root, 'models', f'transfer_*{target_building[:15]}_*.ckpt'))
            if transfer_checkpoints:
                latest_checkpoint = max(transfer_checkpoints, key=os.path.getmtime)
                new_name = f'transfer_{target_building[:15]}_{weeks}week_epoch={latest_checkpoint.split("epoch=")[1]}'
                new_path = os.path.join(data_efficiency_dir, new_name)
                
                # Move to data_efficiency folder
                import shutil
                shutil.move(latest_checkpoint, new_path)
                print(f"✓ Saved: {new_name}")
            
            print(f"✓ Transfer ({weeks} weeks) complete")
            print(f"  Test RMSE: {transfer_results[0]['test_rmse']:.4f}")
            print(f"  Test MAE: {transfer_results[0]['test_mae']:.4f}")
            
        except Exception as e:
            print(f"✗ Transfer ({weeks} weeks) FAILED: {e}")
            continue
        
        print(f"\n{'='*80}")
        print(f"  {weeks} WEEK(S) TRAINING COMPLETE")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("  ALL DATA EFFICIENCY TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {data_efficiency_dir}")
    print("\nNext step: Run evaluation")
    print("  python evaluate_all_models.py")
    print("="*80)


if __name__ == '__main__':
    train_all_data_efficiency_models()
