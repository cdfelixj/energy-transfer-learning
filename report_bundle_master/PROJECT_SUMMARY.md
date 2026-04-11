# Transfer Learning for Building Energy Forecasting

## Project Overview

This project implements a **4-strategy transfer learning framework** for building energy consumption forecasting using LSTM neural networks, validated across 12 experiments covering 6 building-pair clusters.

## 🎯 Experimental Design

### Fine-Tuning Strategies

1. **Scratch / Pre-Transfer** (`src/train_pretransfer.py`)
   - **Purpose**: Control group — train from scratch on limited target data
   - **Architecture**: 2 layers × 64 hidden, seq=24h
   - **Trainable params**: ~88 K (all)
   - **Starting point**: Random initialisation

2. **Full Fine-Tuning / Transfer** (`src/train_transfer.py`)
   - **Purpose**: Warm-start fine-tuning on all parameters
   - **Architecture**: 2 layers × 64 hidden, seq=24h
   - **Trainable params**: ~620 K (all)
   - **Starting point**: Baseline weights
   - **LR**: 1e-4 (10× lower than Scratch)

3. **Frozen Backbone** (`src/models.py` — `EnergyLSTMFrozen`)
   - **Purpose**: Transfer with catastrophic-forgetting prevention
   - **Architecture**: LSTM frozen; only MLP head trainable
   - **Trainable params**: ~8 K
   - **Starting point**: Baseline weights

4. **Adapter Layers** (`src/models.py` — `EnergyLSTMAdapter`)
   - **Purpose**: Minimal-parameter transfer with added expressiveness
   - **Architecture**: Linear(128→32)+ReLU+Linear(32→128) bottleneck, LSTM frozen
   - **Trainable params**: ~16 K (adapter + head)
   - **Starting point**: Baseline weights

**Baseline** (`src/train_baseline.py`): 3-layer LSTM (128 hidden, seq=168h) trained on 2 years of source building data; produces the shared checkpoint from which all transfer strategies start.

**Data Efficiency Sweep** (`train_data_efficiency.py`): Train all 4 strategies at 1, 2, 4, 8, 16, 32, 64, 104 weeks of target data to quantify the data-scarce benefit of transfer learning.

### Key Design Decisions

✅ **Stratified Random Split**: Month-based shuffle split ensures train/val/test have similar energy distributions. Chronological split caused a 52% train→test mean shift (negative R²).

✅ **Adaptive architecture**: Limited-data models use 2 layers × 64 hidden (not 3 × 128) to prevent overfitting with small datasets.

✅ **Same target data per comparison**: All 4 strategies are evaluated on identical target-building data to isolate the effect of pre-training and fine-tuning strategy.

## 📊 Research Questions

1. Does any transfer strategy beat Scratch at low data levels (1–8 weeks)?
2. Which fine-tuning strategy is most robust — Full, Frozen, or Adapter?
3. Can multi-source pre-training fix catastrophic collapse on hard targets?
4. How does source-to-target domain distance (site/type) affect TL benefit?
5. Does auto-switching between Scratch and Transfer always select the better one?

## 📋 Experiment Roster

| # | Name | Source → Target | Key question |
|---|---|---|---|
| 1 | `rat_education` | Rat/Colin → Rat/Denise | Does TL work at all? |
| 2 | `rat_education_new` | Rat/Theo → Rat/Lee | Replicability? |
| 3 | `eagle_education` | Eagle/Samantha → Eagle/Brooke | TL at different site? |
| 4 | `lamb_education` | Lamb/Lucas → Lamb/Mae | TL at a third site? |
| 5 | `office_any` | Hog/Miriam → Hog/Denita | TL for office type? |
| 6 | `lodging_any` | Robin/Celia → Robin/Oliva | TL for lodging type? |
| 7 | `multi_transfer` | 5-building pool → Eagle/Brooke | Multi-source fixes collapse? |
| 8 | `cross_type_transfer` | Same-site / Same-type / Cross-type → Eagle/Brooke | Domain distance impact? |
| 9 | `ensemble_transfer` | Model soup (5 baselines) → Eagle/Brooke | Joint vs. averaging? |
| 10 | `multitransfer_ablation` | N=1…15 sources → Eagle/Brooke | How many sources needed? |
| 11 | `multitransfer_generalisation` | 5-building pool → Rat/Denise | Multi-source on easy targets? |
| 12 | `switch_modelling` | Colin → Denise | Auto-select beats either? |
| PRIME | `prime_experiment` | 5 Eagle/Edu sources → Eagle/Brooke | Does performance-weighted blending beat Scratch? |

## 🚀 Usage

```bash
python scripts/discover_buildings.py                          # Select building pairs
python scripts/run_experiment_suite.py                        # Experiments 1–6
python scripts/run_multi_transfer_experiment.py               # Experiment 7
python scripts/run_cross_type_experiment.py                   # Experiment 8
python scripts/run_ensemble_transfer_experiment.py            # Experiment 9
python scripts/run_multitransfer_ablation_experiment.py       # Experiment 10
python scripts/run_multitransfer_generalisation_experiment.py # Experiment 11
python scripts/run_switch_modelling_experiment.py             # Experiment 12
python scripts/run_prime_experiment.py                        # PRIME experiment
python scripts/evaluate_all_models.py                         # Final evaluation
```

## 📈 Expected Results

```
4-Strategy Comparison (per experiment, per data level):
──────────────────────────────────────────────────────────────────
Strategy       Data Source     MAE (kWh)   R²     vs Scratch
──────────────────────────────────────────────────────────────────
Scratch        8wk target      ~basis      ~0.4   control
Full FT        8wk + TL        <basis      >0.6   ↑ better
Frozen         8wk + TL        <basis      >0.6   ↑ most stable low-data
Adapter        8wk + TL        <basis      >0.6   ↑ competitive
──────────────────────────────────────────────────────────────────
```

**Success Criteria**: Any transfer strategy MAE < Scratch MAE proves transfer learning helps.
Benefit is reported as `(Scratch_MAE − Strategy_MAE) / Scratch_MAE × 100%`.

## 🔧 Technical Details

### Data Processing
- **Dataset**: Building Data Genome Project 2 (electricity meters, multiple sites and types)
- **Features**: Weather data (8) + temporal features (4 cyclical) = 31 features total (29 for Lamb site)
- **Normalisation**: StandardScaler on features, energy target left unscaled
- **Train/Val/Test**: 60/20/20 by month-based stratification

### Model Architecture
```python
# Baseline (abundant data)
LSTM: 3 layers × 128 hidden units
Sequence: 168 hours (1 week)
Dropout: 0.2
Learning rate: 5e-4

# Limited-data strategies
LSTM: 2 layers × 64 hidden units
Sequence: 24 hours (1 day)
Dropout: 0.2
Learning rate: 1e-3 (Scratch), 1e-4 (transfer strategies)
```

### Critical Fixes Applied

⚠️ **Distribution Mismatch**: Chronological split caused 52% mean shift. Fixed by stratified month-based random split.  
✅ **Early Stopping**: Patience increased from 10 → 20 to allow convergence.  
✅ **Architecture Scaling**: 64 hidden / 2 layers for limited-data (not 128/3).  
✅ **Sequence Length**: 168h for baseline, 24h for limited-data training.

## 📁 Project Structure

```
energy-transfer-learning/
├── src/
│   ├── data_loader.py                       # Data loading & preprocessing
│   ├── models.py                            # EnergyLSTM, EnergyLSTMFrozen, EnergyLSTMAdapter
│   ├── train_baseline.py                    # Train source building
│   ├── train_pretransfer.py                 # Train from scratch (Scratch strategy)
│   ├── train_transfer.py                    # Full fine-tuning strategy
│   └── switch_logic.py                      # Auto-selection logic (Experiment 12)
├── scripts/
│   ├── discover_buildings.py                    # Auto building-pair selection
│   ├── run_experiment_suite.py                  # Experiments 1–6
│   ├── run_multi_transfer_experiment.py         # Experiment 7
│   ├── run_cross_type_experiment.py             # Experiment 8
│   ├── run_ensemble_transfer_experiment.py      # Experiment 9
│   ├── run_multitransfer_ablation_experiment.py # Experiment 10
│   ├── run_multitransfer_generalisation_experiment.py  # Experiment 11
│   ├── run_switch_modelling_experiment.py       # Experiment 12
│   ├── evaluate_all_models.py                   # Comprehensive evaluation
│   └── train_data_efficiency.py                 # Data-sweep training helper
├── notebooks/comprehensive_analysis.ipynb      # Analysis notebook (14 sections, all experiments + PRIME)
├── models/experiments/                      # Saved checkpoints
├── results/experiments/                     # CSVs per experiment (Experiments 1–12)
└── results/prime/                           # PRIME experiment CSVs
```

## 📚 Key Findings from Development

1. **Stratified split is essential**: Education buildings have school-holiday patterns that cause a 52% train→test mean shift with chronological splits — fixed by month stratification.
2. **Same-site TL works**: Experiment 1 (Rat/Colin→Rat/Denise) shows 17% MAE improvement at 8 weeks; Transfer consistently beats Scratch in the same-site/same-type setting.
3. **Eagle/Brooke is a hard target**: Single-source Transfer collapses at <16 weeks (MAE up to 904 kWh, 30–45% worse than Scratch). Multi-source pre-training with N=5 eliminates collapse.
4. **Frozen Backbone is most reliable at 1–4 weeks** of target data; Full Fine-Tuning catches up at ≥32 weeks.
5. **N-source optimum is N=3**: Diminishing returns beyond 3 sources; pool diversity (site + type) matters more than raw quantity.
6. **Auto-switching (Switch Modelling, Exp 12)** achieves RMSE 22.72 vs oracle 22.70 vs always-transfer 25.45 — a 10.7% improvement with a trivial post-hoc rule.
7. **PRIME is a negative result at low data**: PRIME MAE @ 8 weeks = 643.5 vs Scratch = 90.5 (6.3× worse). Root cause: all 5 sources are Eagle/Education — source homogeneity prevents distributional diversity. PRIME does outperform Scratch at 32+ weeks (+21.8% at 64 weeks).
8. **Adapter strategy untrained**: Adapter CSVs contain NaN — adapter training was not completed. Results are Scratch/Transfer/Frozen only for all experiments.

## 🔄 Next Steps / Open Questions

- Incorporate cross-site sources into PRIME to overcome Eagle source homogeneity
- Score sources by transfer utility (e.g., MMD, cosine distance) rather than in-domain MAE
- Statistical significance testing across experiments (t-test on MAE improvements)
- Complete Adapter strategy training to enable 4-way strategy comparison
- True temporal validation (chronological split, future forecasting scenarios)

---

**Status**: ✅ All 12 experiments + PRIME experiment complete. See `notebooks/comprehensive_analysis.ipynb` for full analysis.
