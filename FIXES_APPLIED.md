# Fixes Applied - Building Selection Issues

## Problems Identified

1. **Building Site Mismatch**: Scripts tried to use buildings from Eagle, Lamb, and Cockatoo sites, but `load_electricity_data()` only loads Rat site buildings
2. **Building Type Filter**: Only Education buildings from Rat site are available in the filtered dataset
3. **Path Issues**: `train_transfer.py` used relative paths that broke when run from project root

## Solutions Applied

### 1. Baseline Model (`src/train_baseline.py`)
**Changed:**
- ✅ Now uses only **1 Rat site Education building**
- ✅ Building: `Rat_education_Colin` (2 years of data: 2016-2017)
- ✅ Removed dependency on `selected_buildings.csv`

### 2. Pre-Transfer Model (`src/train_pretransfer.py`)
**Changed:**
- ✅ Target building: `Rat_education_Denise` (different from baseline buildings)
- ✅ Also a Rat site Education building
- ✅ Removed dependency on `selected_buildings.csv`

### 3. Transfer Model (`src/train_transfer.py`)
**Changed:**
- ✅ Source building: `Rat_education_Angelica` (used in baseline)
- ✅ Target building: `Rat_education_Denise` (same as pre-transfer)
- ✅ Fixed paths: Changed `'../models'` to `os.path.join(project_root, 'models')`
- ✅ Fixed paths: Changed `'../data/processed/...'` to proper absolute paths

### 4. Evaluation Script (`evaluate_all_models.py`)
**Changed:**
- ✅ Target building: `Rat_education_Denise` (matches training)

## New Training Setup

```
BASELINE MODEL (Source):
└─ Rat_education_Colin  (2 years data: 2016-2017, ~17,520 hours)

TARGET BUILDING (Transfer Learning):
└─ Rat_education_Denise    (1 month data, ~720 hours)
   ├─ Pre-Transfer Model (train from scratch)
   └─ Transfer Model (fine-tune baseline)
```

## Why This Works

1. **All buildings from same site (Rat)**: Ensures consistent data format and availability
2. **All buildings same type (Education)**: Similar usage patterns for better transfer
3. **Target building NOT in baseline**: Proper transfer learning evaluation
4. **Consistent paths**: Works when run from any directory

## Running the Pipeline

Now you can run:

```bash
# From project root
python run_training_pipeline.py

# Or individually
python src/train_baseline.py
python src/train_pretransfer.py
python src/train_transfer.py
python evaluate_all_models.py
```

All scripts will now find the correct buildings and paths!

## Available Rat Education Buildings

From the error message, these buildings are available:
- Rat_education_Angelica ✓ (used in baseline)
- Rat_education_Moises ✓ (used in baseline)
- Rat_education_Colin ✓ (used in baseline)
- Rat_education_Denise ✓ (used as target)
- Rat_education_Wilmer
- Rat_education_Conrad
- Rat_education_Rogelio
- Rat_education_Calvin
- Rat_education_Debra
- Rat_education_Mac
- ... and more

You can modify the scripts to use different combinations of these buildings for experiments.
