"""
QUICK START GUIDE: Three-Model Transfer Learning Pipeline
========================================================

This script provides a complete training pipeline for all three models.
Run this to train everything in sequence.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"  {description}")
    print("="*70)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  TRANSFER LEARNING PIPELINE: Three-Model Training                ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This pipeline will train three models:
    
    1. BASELINE MODEL
       - Trains on 2 years of data from 1 source building
       - Learns general energy consumption patterns
       
    2. PRE-TRANSFER MODEL  
       - Trains from scratch on 1 month of target data
       - Shows performance WITHOUT transfer learning
       
    3. TRANSFER MODEL
       - Fine-tunes baseline on 1 month of target data
       - Shows performance WITH transfer learning
       
    Expected duration: 1-3 hours (depends on hardware)
    """)
    
    response = input("Continue with training pipeline? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Ensure we're in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Step 1: Train Baseline
    if not run_command("python src/train_baseline.py", 
                      "STEP 1/3: Training Baseline Model"):
        print("\n✗ Pipeline stopped due to error in Step 1")
        return
    
    # Step 2: Train Pre-Transfer
    if not run_command("python src/train_pretransfer.py",
                      "STEP 2/3: Training Pre-Transfer Model"):
        print("\n✗ Pipeline stopped due to error in Step 2")
        return
    
    # Step 3: Train Transfer
    if not run_command("python src/train_transfer.py",
                      "STEP 3/3: Training Transfer Model"):
        print("\n✗ Pipeline stopped due to error in Step 3")
        return
    
    # Step 4: Evaluate
    print("\n" + "="*70)
    print("  All models trained successfully!")
    print("="*70)
    
    response = input("\nRun evaluation to compare all models? (y/n): ")
    if response.lower() == 'y':
        run_command("python evaluate_all_models.py",
                   "EVALUATION: Comparing All Three Models")
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)
    print("\nTrained models saved in: models/")
    print("Results saved in: results/")
    print("\nNext steps:")
    print("1. Review results/three_model_comparison.csv")
    print("2. Check results/model_comparison.png for visualization")
    print("3. Compare RMSE/MAE improvements between pre-transfer and transfer models")

if __name__ == '__main__':
    main()
