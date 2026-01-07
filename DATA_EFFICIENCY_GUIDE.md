# Data Efficiency Evaluation Guide

## Overview

The data efficiency evaluation system allows you to assess how pre-transfer and transfer learning models perform with varying amounts of training data (1, 2, 4, 8, 16, 32, 64 weeks, and 2 years).

## Files Created/Modified

1. **`train_data_efficiency.py`** (NEW)
   - Automated training script that trains all data efficiency models
   - Creates models with 1, 2, 4, 8, and 16 weeks of data
   - Saves checkpoints to `models/data_efficiency/` with week count in filename

2. **`evaluate_all_models.py`** (MODIFIED)
   - Added `evaluate_data_efficiency()` function
   - Added `compare_data_efficiency()` function
   - Extended `main()` to include data efficiency evaluation
   - Now outputs 3 comparison tables total

## Usage

### Step 1: Train Data Efficiency Models

```bash
python train_data_efficiency.py
```

This will:
- Train 8 pre-transfer models (1w, 2w, 4w, 8w, 16w, 32w, 64w, 104w) - from scratch
- Train 8 transfer models (1w, 2w, 4w, 8w, 16w, 32w, 64w, 104w) - fine-tuned from baseline
- Save all models to `models/data_efficiency/` folder
- Total: 16 models trained

**Note:** 104 weeks = 2 years (for comparison with baseline model training duration)

**Expected duration:** ~2-4 hours depending on your hardware

**Model naming convention:**
- Pre-transfer: `pretransfer_Rat_education_D_1week_epoch=XX_val_loss=YY.ckpt`
- Transfer: `transfer_Rat_education_D_1week_epoch=XX_val_loss=YY.ckpt`

### Step 2: Evaluate All Models

```bash
python evaluate_all_models.py
```

This will:
1. Run the original 3-model evaluation (Baseline vs Pre-Transfer vs Transfer with 2 months data)
2. **NEW:** Evaluate pre-transfer models with 1, 2, 4, 8, 16 weeks - display comparison table
3. **NEW:** Evaluate transfer models with 1, 2, 4, 8, 16 weeks - display comparison table

## Output

### Console Output

The evaluation script now produces:

1. **Original Metrics Table** - 4 models compared:
   - Baseline-Source (on Colin)
   - Baseline-Target (on Denise)
   - Pre-Transfer (2 months)
   - Transfer (2 months)

2. **Pre-Transfer Data Efficiency Table** - Shows MAE, RMSE, R², MAPE for:
   - 1 week, 2 weeks, 4 weeks, 8 weeks, 16 weeks, 32 weeks, 64 weeks, 2 years (104 weeks)

3. **Transfer Data Efficiency Table** - Shows MAE, RMSE, R², MAPE for:
   - 1 week, 2 weeks, 4 weeks, 8 weeks, 16 weeks, 32 weeks, 64 weeks, 2 years (104 weeks)

### Saved Files

Results are saved to the `results/` directory:

- `three_model_comparison.csv` - Original 3-model comparison
- `pretransfer_data_efficiency.csv` - Pre-transfer data efficiency results
- `transfer_data_efficiency.csv` - Transfer data efficiency results
- `model_comparison.png` - Visualization (unchanged)

## Handling Missing Models

If some models haven't been trained yet, the evaluation will:
- Display a warning message
- Skip that data amount
- Show "N/A" in the comparison table
- Continue with remaining evaluations

This allows partial evaluation if you only want to train subset of models.

## Example Output

```
==========================================================================================
  DATA EFFICIENCY ANALYSIS: Pre-Transfer Models
==========================================================================================

Comparison of Pre-Transfer model performance with varying amounts of training data:
(All models trained and evaluated on same building: Rat_education_Denise)

Metric                    1 Week          2 Weeks         4 Weeks         8 Weeks         16 Weeks        32 Weeks        64 Weeks        2 Years        
-----------------------------------------------------------------------------------------------------------------------------------------------------------
MAE (kWh)                       25.3421         18.7654         15.2341         12.8976         11.2345         10.1234          9.3456          8.7654  
RMSE (kWh)                      35.6789         26.4321         21.3456         18.9012         16.7890         15.2341         13.9876         12.8765  
R² Score                         0.3456          0.5678          0.6789          0.7234          0.7689          0.8123          0.8456          0.8789  
MAPE (%)                        42.3456         32.1234         26.7890         22.4567         19.8765         17.6543         15.8901         14.3210  
Median AE (kWh)                 21.2345         15.6789         12.3456         10.4567          9.1234          8.2345          7.4567          6.8901  
===========================================================================================================================================================

  IMPROVEMENT ANALYSIS (1 Week → 2 Years):
  • RMSE improved by 63.9%
  • MAE improved by 65.4%
  • R² improved by 154.3%
==========================================================================================
```

## Customization

To modify which data amounts are evaluated, edit the `weeks_list` parameter:

```python
# In evaluate_all_models.py, main() function
pretransfer_data_eff_results = evaluate_data_efficiency(
    model_type='pretransfer',
    target_building=target_building,
    weeks_list=[1, 2, 4, 8, 16, 32, 64, 104],  # <- Modify this list (104 = 2 years)
    seq_length=seq_length
)
```

## Troubleshooting

### No models found
**Error:** `⚠ WARNING: No pretransfer model found for X week(s)`

**Solution:** Run `python train_data_efficiency.py` to train all models

### Feature mismatch
**Error:** Model expects different number of features

**Solution:** Ensure all models use `architecture_match` parameter to maintain consistent architecture

### Memory issues
**Error:** Out of memory during training

**Solution:** Reduce batch size in `train_pretransfer.py` and `train_transfer.py` (default: 32)

## Architecture Details

All data efficiency models use:
- **Architecture:** Matches baseline (2 layers, 64 hidden units)
- **Sequence length:** 24 hours (1 day)
- **Batch size:** 32
- **Target building:** Rat_education_Denise
- **Evaluation:** Same test set for fair comparison

Pre-transfer models:
- Random initialization
- Train from scratch

Transfer models:
- Initialize from baseline (Rat_education_Colin)
- Fine-tune on target data
