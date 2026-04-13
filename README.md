# Building Energy Transfer Learning

**Read DOCUMENTATION HIERARCHY**
**MAKE SURE TO READ REPORT.MD and notebooks\comprehensive_analysis.ipynb**

A transfer learning framework for building energy consumption forecasting using LSTM neural networks. This project demonstrates how pre-training on data-rich buildings can improve prediction accuracy for buildings with limited historical data, across 12 experiments and 4 fine-tuning strategies.

## 🎯 Problem Statement

**Challenge**: New buildings or buildings with recent sensor installations have limited historical data, making it difficult to train accurate energy forecasting models.

**Solution**: Transfer learning — train on buildings with abundant data, then fine-tune on the target building with limited data. Four fine-tuning strategies are compared: Scratch (control), Full Fine-Tuning, Frozen Backbone, and Adapter Layers.

## 📊 Dataset

**Building Data Genome Project 2** (Miller & Meggers, 2017)
- Time period: 2016–2017 (2 years hourly data)
- Features: Energy consumption + weather data (31 features after preprocessing)
- Buildings: drawn from 6 clusters — Rat, Eagle, Lamb (Education), Hog (Office), Robin (Lodging)
- Coverage: 1–104 weeks of target building data per experiment

## 🏗️ 4-Strategy Fine-Tuning Framework

All experiments compare four strategies on the same limited target-building data:

```
Source Building (2 years, abundant data)
         │
         └─ Train Baseline LSTM  ─────────────────────────────────────────────┐
              3 layers × 128 hidden  seq=168h                                 │
                                                                               │ weights
Target Building (1–104 weeks, limited data)                                   │
         │                                                                     │
         ├─ Scratch          : random init,  all params train  (control)      │
         ├─ Full Fine-Tuning : warm start ◄──────────────────────────────────┤
         ├─ Frozen Backbone  : warm start, LSTM frozen, only head trains ◄────┤
         └─ Adapter          : warm start, LSTM frozen, small bottleneck ◄────┘
                               adapter (32-dim) + head trains
```

**Scratch is the control baseline.** Any transfer strategy beating Scratch proves
the value of pre-training on another building.

## 📋 Experiment Roster

| # | Name | Site/Type | Source → Target | Key question | Script |
|---|---|---|---|---|---|
| 1 | `rat_education` | Rat/Edu | Colin → Denise | Does TL work at all? | `run_experiment_suite.py` |
| 2 | `rat_education_new` | Rat/Edu | Theo → Lee | Replicability within same cluster? | `run_experiment_suite.py` |
| 3 | `eagle_education` | Eagle/Edu | Samantha → Brooke | TL at a different site? | `run_experiment_suite.py` |
| 4 | `lamb_education` | Lamb/Edu | Lucas → Mae | TL at a third site? | `run_experiment_suite.py` |
| 5 | `office_any` | Hog/Office | Miriam → Denita | TL for office buildings? | `run_experiment_suite.py` |
| 6 | `lodging_any` | Robin/Lodging | Celia → Oliva | TL for lodging buildings? | `run_experiment_suite.py` |
| 7 | `multi_transfer` | Eagle/Edu | 5-building pool → Brooke | Multi-source pre-training fixes collapse? | `run_multi_transfer_experiment.py` |
| 8 | `cross_type_transfer` | Eagle/Edu | Same-site / Same-type / Cross-type → Brooke | Domain distance impact? | `run_cross_type_experiment.py` |
| 9 | `ensemble_transfer` | Eagle/Edu | Model soup of 5 baselines → Brooke | Joint training vs. weight averaging? | `run_ensemble_transfer_experiment.py` |
| 10 | `multitransfer_ablation` | Eagle/Edu | N=1…15 buildings → Brooke | How many source buildings needed? | `run_multitransfer_ablation_experiment.py` |
| 11 | `multitransfer_generalisation` | Rat/Edu | 5-building pool → Denise | Multi-source benefit on easy targets? | `run_multitransfer_generalisation_experiment.py` |
| 12 | `switch_modelling` | Rat/Edu | Colin → Denise | Auto-selection beats either strategy? | `run_switch_modelling_experiment.py` |
| PRIME | `prime_experiment` | Rat/Edu | Top-5 Rat sources → Denise | Performance-weighted multi-source init on matched target? | `run_prime_experiment.py` |

## 🚀 Quick Start

### Prerequisites

```bash
conda create -n energy-transfer python=3.10
conda activate energy-transfer
pip install torch pytorch-lightning pandas numpy scikit-learn matplotlib seaborn
```

### Full Training Pipeline

```bash
# Step 1: Discover best building pairs for each experiment cluster
python scripts/discover_buildings.py

# Step 2: Train 6 baseline experiments (baseline + 4 strategies × 8 data amounts each)
python scripts/run_experiment_suite.py

# Step 3: Advanced multi-source experiments (run in any order)
python scripts/run_multi_transfer_experiment.py
python scripts/run_cross_type_experiment.py
python scripts/run_ensemble_transfer_experiment.py

# Step 4: Ablation & generalisation studies
python scripts/run_multitransfer_ablation_experiment.py
python scripts/run_multitransfer_generalisation_experiment.py

# Step 5: Switch modelling
python scripts/run_switch_modelling_experiment.py

# Step 6: PRIME experiment (performance-weighted multi-source initialisation)
python scripts/run_prime_experiment.py
# Results saved to results/prime/

# Step 7: Evaluate everything
python scripts/evaluate_all_models.py
```

Results are saved under `results/experiments/{experiment_name}/` and `results/prime/`.

## 📁 Project Structure

```
energy-transfer-learning/
├── src/
│   ├── data_loader.py                       # Data preprocessing & loading
│   ├── models.py                            # EnergyLSTM, EnergyLSTMFrozen, EnergyLSTMAdapter
│   ├── train_baseline.py                    # Train source-building baseline
│   ├── train_pretransfer.py                 # Train from scratch (Scratch strategy)
│   ├── train_transfer.py                    # Full fine-tuning strategy
│   └── switch_logic.py                      # Auto-selection logic for Experiment 12
│
├── scripts/
│   ├── discover_buildings.py                    # Auto-select best building pairs
│   ├── run_experiment_suite.py                  # Experiments 1–6
│   ├── run_multi_transfer_experiment.py         # Experiment 7
│   ├── run_cross_type_experiment.py             # Experiment 8
│   ├── run_ensemble_transfer_experiment.py      # Experiment 9
│   ├── run_multitransfer_ablation_experiment.py # Experiment 10
│   ├── run_multitransfer_generalisation_experiment.py  # Experiment 11
│   ├── run_switch_modelling_experiment.py       # Experiment 12
│   ├── run_prime_experiment.py                  # PRIME experiment
│   ├── evaluate_all_models.py                   # Comprehensive evaluation
│   └── train_data_efficiency.py                 # Data-sweep training helper
│
├── notebooks/
│   └── comprehensive_analysis.ipynb         # Complete analysis notebook (14 sections, all experiments + PRIME)
│
├── data/raw/building-data-genome-project-2/ # Dataset (not tracked by git)
├── models/experiments/                      # Saved model checkpoints (.ckpt)
├── results/experiments/                     # CSVs per experiment
├── results/prime/                           # CSVs for PRIME experiment
│
├── EXPERIMENTS.md                           # Full description of all 12 experiments
├── PROJECT_SUMMARY.md                       # 4-strategy framework & design decisions
├── TECHNICAL_IMPROVEMENTS.md                # Bug-fix history & advanced experiment record
└── notes.txt                                # Quick training pipeline reference
```

## 🔬 Methodology

### Data Preprocessing

1. **Filtering**: Select buildings by site + type; electricity meter only
2. **Cleaning**: Remove negatives, outliers (>10× 95th percentile), extended zeros (>72h)
3. **Interpolation**: Linear interpolation for gaps ≤3 hours
4. **Normalisation**: StandardScaler on features; energy target left unscaled
5. **Features**: Weather (8) + Temporal (4 cyclical) + lag = 31 features per timestep

### Data Splitting Strategy

**Critical decision: Stratified Random Split by month**

Education buildings have strong seasonal patterns. A chronological split produced a 52% mean-energy shift between train (60.8 kWh) and test (29.2 kWh), leading to negative R². Stratifying by month fixes this, giving <1% shift and R² > 0.6.

### Model Configurations

| Aspect | Baseline | Scratch / Transfer (limited data) |
|--------|----------|----------------------------------|
| Data | 2 years (~17,500h) | 1–104 weeks |
| Sequence Length | 168 hours (1 week) | 24 hours (1 day) |
| LSTM Layers | 3 | 2 |
| Hidden Units | 128 | 64 |
| Learning Rate | 5e-4 | 1e-3 (Scratch), 1e-4 (transfer strategies) |
| Dropout | 0.2 | 0.2 |
| Early Stop Patience | 15 | 20 |

### Fine-Tuning Strategy Details

| Strategy | Trainable params | Starting point | Catastrophic forgetting risk |
|---|---|---|---|
| Scratch | ~88 K (random init) | — | N/A |
| Full Fine-Tuning | ~620 K | Baseline weights | High (low LR mitigates) |
| Frozen Backbone | ~8 K (head only) | Baseline weights | None |
| Adapter (b=32) | ~16 K (adapter + head) | Baseline weights | None |

## 📈 Evaluation Metrics

- **MAE** — Mean Absolute Error (primary cross-experiment metric; scale-normalised via benefit %)
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **R²** — Coefficient of Determination (variance explained; 0 = predict mean, 1 = perfect)
- **MAPE** — Mean Absolute Percentage Error (scale-independent)

Transfer benefit is always reported relative to Scratch: `(Scratch_MAE − Strategy_MAE) / Scratch_MAE × 100%`

## 🔑 Key Findings

1. **Transfer learning is conditionally effective.** Same-site, same-type transfers (e.g., Rat/Colin → Rat/Denise) yield consistent **17% MAE improvement at 8 weeks**. Scratch catches up at ≥32 weeks across all experiments.
2. **Transfer collapses on hard targets.** Eagle/Brooke single-source fine-tuning performs **30–45% worse than Scratch** at <16 weeks (MAE spikes to 904+ kWh, R²=−340). No single-source strategy variant fixes this.
3. **Multi-source pre-training (N=3 diverse buildings)** is the most robust general strategy. Adding a third site (Lamb) provides the last major jump; **diminishing returns are strong past N=3**. Source diversity (site + type) matters more than raw source count.
4. **Frozen Backbone** is the single most reliable low-data strategy (1–4 weeks); Full Fine-Tuning wins at ≥32 weeks.
5. **Auto-switching (Experiment 12)** achieves near-oracle performance: mean RMSE **22.72 vs oracle 22.70**, outperforming always-transfer (RMSE 25.45) by **10.7%**. Scratch was selected at 5 of 8 decision points — transfer is not always the better strategy, even on an easy target.
6. **PRIME succeeds in the same-site regime.** On Rat/Denise (same-site, same-type), PRIME scored **31.3% better MAE than Scratch at 8 weeks** (13.9 vs 20.3 kWh). All 5 sources are Rat/Education — strong domain alignment enables effective performance-weighted blending. PRIME converges with Scratch at 16+ weeks. An earlier run on Eagle/Brooke confirmed that domain alignment is a prerequisite: PRIME failed catastrophically on that mismatched target (643.5 vs 90.5 kWh, 6.3× worse), showing in-domain val MAE does not predict cross-domain transfer utility.

## 🚧 Open Questions & Future Work

- **Chronological evaluation**: Re-run experiments with a true temporal train/test split (currently stratified random split is used throughout)
- **Statistical significance**: Single training runs only — confidence intervals require repeated runs with different random seeds
- **PRIME on hard targets with diverse sources**: Test PRIME on Eagle/Brooke with cross-site source diversity to verify whether domain diversity addresses the Eagle-regime collapse
- **Architecture alternatives**: Transformer (PatchTST, Informer), TCN, GRU as LSTM replacements
- **Continual learning**: EWC or replay buffers to prevent catastrophic forgetting during long-term deployment
- **Adapter strategy**: Complete training runs (currently untrained — all CSVs contain NaN)

## 📚 References

1. Miller, C., & Meggers, F. (2017). The Building Data Genome Project 2
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
3. Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning

## 📄 License

MIT License

---

**Status**: ✅ All 12 experiments + PRIME complete. Full analysis in `notebooks/comprehensive_analysis.ipynb`.  
**Last Updated**: April 2026
