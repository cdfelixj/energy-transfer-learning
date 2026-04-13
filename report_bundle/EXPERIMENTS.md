# Experiments Guide

**Few-Shot Transfer Learning for Building Energy Forecasting**

This document explains every experiment in the project — what it tests, why it exists, and what conclusions can be drawn from it. Experiments are ordered from foundational to advanced.

---

## Background

All experiments share the same structure:

1. **Train a baseline** on a *source* building with full historical data (2 years, `seq_length=168`)
2. **Sweep data amounts** — train models with 1, 2, 4, 8, 16, 32, 64, and 104 weeks of *target* building data
3. **Compare strategies** — evaluate each model type across the sweep and save CSVs to `results/experiments/{name}/`

The core question across all experiments is: **does transfer learning help, and under what conditions?**

### Model Strategies

| Strategy | Abbreviation | Description |
|---|---|---|
| Pre-Transfer | `pretransfer` | Trained from scratch on limited target data — the *control* |
| Full Fine-Tuning | `transfer` | Baseline weights warm-started then all parameters updated |
| Frozen Backbone | `frozen` | Baseline LSTM locked; only the MLP head updated |
| Adapter Layers | `adapter` | Baseline LSTM locked; lightweight bottleneck adapter (32-dim) + head updated |
| Multi-Transfer | `multitransfer` | Fine-tune a baseline pre-trained across 5 diverse buildings |
| Ensemble Transfer | `ensembletransfer` | Fine-tune a *weight-averaged* (model soup) ensemble of 5 individually trained models |

Pre-Transfer is the benchmark for everything else. Any transfer strategy that beats Pre-Transfer has demonstrated a genuine benefit from pre-training.

---

## Experiment 1 — `rat_education`

**Script:** `run_experiment_suite.py`
**Runner file:** part of the 6-experiment suite

### Intent

The foundational experiment. Establishes that transfer learning works at all for building energy forecasting, using the cleanest available data pair.

### Setup

| Role | Building | Site | Type | Data quality |
|---|---|---|---|---|
| Source (baseline) | `Rat_education_Colin` | Rat | Education | 99.57% complete |
| Target | `Rat_education_Denise` | Rat | Education | ~95% complete |

Both buildings are same-site and same-type, making this the *easiest possible* transfer scenario. Colin has exceptional data quality and was chosen as the source precisely to ensure a strong baseline model.

### What is compared

Pre-Transfer vs Full Fine-Tuning (Transfer) across 1–104 weeks of Denise's data.

### What we can learn

- Whether transfer learning outperforms training from scratch when source and target are from the same site and building type
- The *magnitude* of improvement, especially at very low data amounts (1–4 weeks)
- At what point Pre-Transfer "catches up" to Transfer (i.e., how many weeks of data eliminates the transfer advantage)
- The data efficiency curve shape — does Transfer converge faster?

### Expected result

Transfer should comfortably outperform Pre-Transfer at low data amounts. Being same-site, the two buildings share weather patterns and operational rhythms, so the baseline model's learned representations should transfer smoothly. At high data amounts (64–104 weeks), the gap should narrow.

---

## Experiment 2 — `rat_education_new`

**Script:** `run_experiment_suite.py`

### Intent

A replication of `rat_education` using a *different, automatically selected* Rat/Education building pair. Verifies the first experiment was not a fluke due to the specific choice of Colin and Denise, and provides a second data point for the same-site/same-type regime.

### Setup

Building selection is performed by `discover_buildings.py` which ranks all Rat/Education buildings by data completeness and selects the best source/target pair that excludes Colin and Denise (to avoid overlap with Experiment 1).

### What we can learn

- Reproducibility: does transfer learning work for a different pair within the same site and type?
- Sensitivity to building choice within the same category
- Whether Experiment 1's results generalise to other Rat/Education building combinations

---

## Experiment 3 — `eagle_education`

**Script:** `run_experiment_suite.py`

### Intent

Tests whether transfer learning works on a *different site* — Eagle rather than Rat. This is the first test of cross-site generalisation within the same building type.

### Setup

| Role | Building | Site | Type |
|---|---|---|---|
| Source (baseline) | Auto-selected Eagle/Education building | Eagle | Education |
| Target | Auto-selected Eagle/Education building | Eagle | Education |

Building selection is automated via `discover_buildings.py`. The primary Eagle source used in later experiments is `Eagle_education_Samantha`.

### What we can learn

- Does the transfer learning conclusion from Experiment 1 hold at a different campus?
- Eagle is a different physical environment from Rat — does this affect the baseline model's quality or the transfer benefit?
- **Key finding expected here:** the single-source Transfer model may *collapse* at fewer than 16 weeks on certain Eagle/Education targets (e.g. Brooke). This failure case motivates Experiments 7–10.

### Significance

Eagle/Education emerges as a *harder* transfer target than Rat/Education. Understanding why — and fixing it — is a major thread of the later experiments.

---

## Experiment 4 — `lamb_education`

**Script:** `run_experiment_suite.py`

### Intent

Tests transfer learning on a third site (Lamb), again within the Education building type. Each new site tests robustness of the transfer learning approach to different building populations and climate conditions.

### Setup

Automated building selection within Lamb/Education buildings.

### What we can learn

- Does transfer learning generalise across three different sites (Rat, Eagle, Lamb)?
- Lamb buildings constrain the *feature intersection* — Lamb site data has 29 features vs 31 for Rat/Eagle. This means multi-source baselines trained with Lamb in the pool will have `input_size=29`, affecting later ablation experiments.
- Establishes whether the benefit is consistent or site-dependent.

---

## Experiment 5 — `office_any`

**Script:** `run_experiment_suite.py`

### Intent

Tests transfer learning across a *different building type*: Office buildings, drawn from any site. This is the first test of whether the framework generalises beyond Education buildings.

### Setup

Buildings are selected from any site but restricted to the Office `primaryspaceusage` category. The source and target are both Office buildings but may come from entirely different sites.

### What we can learn

- Does the knowledge learned from one Office building transfer to another Office building across different sites?
- How does transfer performance compare to the Education experiments? Office buildings have different occupancy profiles (weekday business hours) vs education buildings (term-time patterns).
- Whether the approach is building-type agnostic or specialised to Education.

---

## Experiment 6 — `lodging_any`

**Script:** `run_experiment_suite.py`

### Intent

Tests transfer learning on Lodging/residential buildings — the most different occupancy profile from Education. Lodging buildings have near-constant 24/7 occupancy rather than scheduled use, which represents a genuinely different energy consumption pattern.

### Setup

Buildings selected from any site within the Lodging/residential `primaryspaceusage` category.

### What we can learn

- Whether the transfer learning approach is robust to very different operational patterns
- Whether the LSTM, trained primarily on Education buildings in earlier experiments, can learn transferable representations for Lodging buildings
- Provides the most challenging same-type transfer test

---

## Experiment 7 — Multi-Transfer (`multi_transfer`)

**Script:** `run_multi_transfer_experiment.py`

### Intent

Addresses the **collapse failure** observed in `eagle_education`: when training with fewer than 16 weeks of target data, the single-source Transfer model degrades severely on Eagle/Brooke (MAE spikes). The hypothesis is that a baseline pre-trained on *diverse* buildings provides a more robust initialisation.

### Setup

| Role | Building | Details |
|---|---|---|
| Target | `Eagle_education_Brooke` | Eagle / Education; 99.78% complete |
| Single-source baseline | `Eagle_education_Samantha` | Same site — fairest single-source comparison |
| Multi-source pool (5 buildings) | `Rat_education_Colin` + `Eagle_education_Samantha` + `Lamb_education_Lucas` + `Hog_office_Miriam` + `Robin_lodging_Celia` | 3 sites, 3 building types |

### Strategies compared

1. **Pre-Transfer** — scratch on N weeks of Brooke (control)
2. **Transfer** — fine-tune Eagle/Samantha baseline
3. **Multi-Transfer** — fine-tune 5-building diverse baseline

### What we can learn

- Does a diverse multi-source baseline eliminate the collapse that affects single-source Transfer at low data amounts?
- Is there a trade-off — does the multi-source model perform *worse* than single-source Transfer at moderate/high data amounts (because it's been trained on potentially irrelevant buildings)?
- How many weeks of data does Multi-Transfer need before Pre-Transfer catches up?
- The core hypothesis: **broader pre-training provides more robust initialisation**, especially for targets that are hard to transfer to from a single source.

### Significance

This is the central advanced experiment of the project. If Multi-Transfer fixes the collapse while maintaining competitive performance at higher data amounts, it demonstrates that source diversity is a key factor in transfer learning robustness.

---

## Experiment 8 — Cross-Type Transfer (`cross_type_transfer`)

**Script:** `run_cross_type_experiment.py`

### Intent

Isolates the effect of **source-target compatibility** on transfer performance. Specifically: does matching site and building type matter, or can a completely unrelated source still provide useful initialisations?

### Setup

All variants transfer to the same target: `Eagle_education_Brooke`.

| Strategy | Source building | Match type |
|---|---|---|
| `transfer_samesite` | `Eagle_education_Samantha` | Same site + same type (tightest match) |
| `transfer_sametype` | `Rat_education_Colin` | Different site, same building type |
| `transfer_crosstype` | `Hog_office_Miriam` | Different site + different building type |

All three source buildings happen to share `input_size=31`, so no feature truncation is needed — weights transfer directly without any alignment overhead.

### What we can learn

- Is same-site always better than different-site when the building type is the same? (`samesite` vs `sametype`)
- Is same-type necessary for transfer to work, or does a completely different building type (Office) still produce useful representations? (`sametype` vs `crosstype`)
- At what data amounts do the differences between variants converge?
- Whether the transfer learning benefit comes from *type-specific patterns* or from more general time-series representations (seasonality, weather response, etc.)

### Significance

This experiment provides strong evidence about what makes a good source domain. If `transfer_crosstype` performs similarly to `transfer_sametype`, it suggests the model learns general temporal patterns that transfer regardless of building function.

---

## Experiment 9 — Ensemble Transfer (`ensemble_transfer`)

**Script:** `run_ensemble_transfer_experiment.py`

### Intent

Tests an alternative multi-source strategy: instead of training a single baseline on combined data from all sources, train each source building *individually* and then **weight-average their parameters** (a "model soup") before fine-tuning on the target.

This addresses whether the way multi-source knowledge is aggregated matters — joint training (Experiment 7) vs. averaging independently trained models.

### Setup

| Role | Details |
|---|---|
| Target | `Eagle_education_Brooke` |
| Single-source baseline | `Eagle_education_Samantha` |
| Ensemble pool | Same 5 buildings as Experiment 7 |
| Feature alignment | All 5 individual models trained with `input_size=29` (intersection of all sites) |

### How the ensemble is built

1. Each of the 5 source buildings is trained individually to convergence with features truncated to the common feature count
2. The 5 state dicts are uniformly averaged: `θ_soup = (1/5) Σ θᵢ`
3. The averaged initialisation is fine-tuned on the target building

### Strategies compared

1. **Pre-Transfer** — scratch (control)
2. **Transfer** — fine-tune Eagle/Samantha baseline
3. **Ensemble Transfer** — fine-tune the model-soup initialisation

### What we can learn

- Is joint training (Multi-Transfer) or weight averaging (Ensemble Transfer) a better strategy for combining multi-source knowledge?
- Does the model-soup approach produce a more "central" initialisation that generalises better?
- At low data amounts, which multi-source approach is more stable (less prone to collapse)?

### Significance

Model soup / weight averaging has shown strong results in NLP fine-tuning literature. This experiment tests whether that finding transfers to time-series building energy forecasting.

---

## Experiment 10 — Multi-Transfer N-Source Ablation (`multitransfer_ablation`)

**Script:** `run_multitransfer_ablation_experiment.py`

### Intent

Studies how multi-transfer performance **scales with the number of source buildings** in the pre-training pool. Is there diminishing returns? Does diversity matter more than quantity? Does adding more buildings always help?

### Setup

Target: `Eagle_education_Brooke`. The pool grows progressively:

| N | Buildings in pool | Diversity added |
|---|---|---|
| 1 | Eagle/Samantha | Same site/type baseline |
| 2 | + Rat/Colin | Different site, same type |
| 3 | + Lamb/Lucas | Third site, same type |
| 4 | + Hog/Miriam | First cross-type (Office) |
| 5 | + Robin/Celia | Second cross-type (Lodging) |
| 10 | + 5 more Eagle/Rat buildings | More quantity, same diversity |
| 15 | + 5 more Eagle buildings | Further quantity scaling |

Note: at N≥3, Lamb constrains the feature intersection to 29 features for all larger pools.

### What we can learn

- Is the benefit from Multi-Transfer due to source *diversity* or simply *quantity* of pre-training data?
- Where does diminishing returns set in — is N=3 already sufficient, or does adding N=5, 10, 15 continue to help?
- Does adding buildings of the *same* site/type as N increases (N=10, 15 add more Eagle buildings) help as much as adding new sites/types did at N=2–5?
- The optimal pool size for practical deployment (balancing training cost vs. benefit)

### Significance

This is the most computationally expensive experiment (trains N=1,2,3,4,5,10,15 separate baselines). The results directly answer: *how diverse and how large does the source pool need to be?*

---

## Experiment 11 — Multi-Transfer Generalisation (`multitransfer_generalisation`)

**Script:** `run_multitransfer_generalisation_experiment.py`

### Intent

Tests whether the multi-source approach is *generally* better, or only fixes a specific failure case. Experiment 7 showed Multi-Transfer helps on Eagle/Brooke (a hard target). Does it also help — or even hurt — on an easy target where single-source Transfer already works well?

### Setup

| Role | Building | Details |
|---|---|---|
| Target | `Rat_education_Denise` | Rat / Education — the "easy" target from Experiment 1 |
| Single-source | `Rat_education_Colin` | Same site, same type |
| Multi-source | Same 5-building pool as Experiment 7 | |

Note: Denise has `input_size=31`; the 5-building baseline has `input_size=29`, so two features are truncated when fine-tuning.

### Strategies compared

1. **Pre-Transfer** — scratch on N weeks of Denise
2. **Transfer** — fine-tune Rat/Colin baseline
3. **Multi-Transfer** — fine-tune 5-building diverse baseline

### What we can learn

- Does Multi-Transfer hurt on easy targets? (Possibly, because the multi-source baseline is less specialised to Rat/Education than the Colin-only baseline)
- Does Multi-Transfer provide consistent improvements, or is it only a rescue strategy for collapse cases?
- Is the feature truncation at the 29-feature boundary a meaningful penalty?
- The practical question: should a real deployment always use multi-source pre-training, or only when single-source is known to be problematic?

### Significance

Without this experiment, one could not distinguish between: (a) Multi-Transfer is universally better, or (b) Multi-Transfer only helps in specific failure cases. The answer shapes the deployment recommendation.

---

## Experiment 12 — Switch Modelling (`switch_modelling`)

**Script:** `run_switch_modelling_experiment.py`

### Intent

Instead of committing to a single strategy, this experiment explores **automatic model selection**: at each data amount, choose whichever of Pre-Transfer or Transfer performs better, subject to a minimum improvement threshold. The idea is that in practice an operator might train both and pick the better one.

### Setup

| Role | Building |
|---|---|
| Target | `Rat_education_Denise` |
| Source | `Rat_education_Colin` |

The switch logic (in `src/switch_logic.py`) compares the test RMSE of Pre-Transfer and Transfer at each week count. A switch away from the Transfer default is only triggered if one model is more than `threshold`% better (default: 2.0%). The full decision hierarchy is:

1. If one model has NaN RMSE → select the other automatically
2. If the margin exceeds `threshold` → select the clearly better model
3. If models are within `threshold` → prefer Transfer (warm-start bias)

### What we can learn

- Does automatic switching produce consistently better performance than either strategy alone?
- How often does Pre-Transfer actually beat Transfer (i.e., when does training from scratch win)?
- How sensitive is the switched outcome to the threshold value?
- What is the "oracle" performance (always picking the best available model) vs. the switched performance?
- Practical insights: at which week counts is Transfer reliably dominant, and where is Pre-Transfer competitive?

### Outputs

Uniquely, this experiment produces a `data_efficiency_switched.csv` containing the *selected* model's metrics at each week count, plus a `switch_summary.csv` with aggregate statistics (switching rate, average improvement from switching, etc.).

### Significance

This experiment represents the most practical deployment scenario: rather than a fixed transfer strategy, use both and let performance decide. If the switched strategy reliably outperforms either individual strategy, it justifies the added training cost.

---

## PRIME Experiment — `prime_experiment`

**Script:** `run_prime_experiment.py`

### Intent

PRIME (**P**erformance-weighted **R**obust **I**nitialisation for **M**ulti-source **E**nergy forecasting) is the project's proposed novel contribution. It extends multi-source transfer learning by ranking and weighting source buildings by their predictive quality before combining their model weights. The hypothesis is that selectively emphasising high-quality sources produces a better initialisation than uniform averaging (as in Experiments 9 and 10).

The target is **Rat/Denise** — the same easy same-site target from Experiment 1, where standard single-source transfer already works. PRIME is evaluated here to quantify whether performance-weighted source blending provides additional benefit over standard transfer on a well-matched same-site target.

### Setup

| Role | Building | Site | Type | Val MAE | PRIME Weight |
|---|---|---|---|---|---|
| Target | `Rat_education_Denise` | Rat | Education | — | — |
| Source 1 | `Rat_education_Earnest` | Rat | Education | 11.81 | 0.3305 (highest) |
| Source 2 | `Rat_education_Meghan` | Rat | Education | 11.85 | 0.3293 |
| Source 3 | `Rat_education_Irma` | Rat | Education | 14.19 | 0.2751 |
| Source 4 | `Rat_education_Nellie` | Rat | Education | 97.07 | 0.0402 |
| Source 5 | `Rat_education_Beverly` | Rat | Education | 156.97 | 0.0249 (lowest) |

Sources are selected by a composite quality score (data completeness + validation MAE on in-domain data). The top-5 Rat/Education buildings are used. Source weights are dominated by Earnest and Meghan (val MAE ≈ 11.8 kWh each); Beverly receives near-zero weight due to high validation error (156.97 kWh).

### Mechanism

1. **Rank** candidate source buildings by composite score (completeness × inverse val-MAE).
2. **Compute weights** using inverse-MAE normalisation: `weight_i = (1/val_mae_i) / Σ(1/val_mae_j)`. Buildings with lower MAE receive proportionally greater weight.
3. **Blend parameters**: `θ_PRIME = Σ weight_i × θ_source_i` — a weighted average of 5 individually trained source model state dicts.
4. **Fine-tune**: Use the blended initialisation as the starting point for standard fine-tuning on Rat/Denise at each data level (1–104 weeks).

### Strategies compared

| Strategy | Description |
|---|---|
| `pretransfer` | Train from scratch on N weeks of Rat/Denise data (control) |
| `prime_transfer` | Fine-tune the PRIME-blended initialisation on N weeks of Rat/Denise data |

### What we can learn

- Does performance-weighted source selection produce a better initialisation than uniform averaging (Experiment 9) or joint multi-source training (Experiment 7)?
- Does performance-weighted weighting within a same-site pool provide a better head-start than standard single-source transfer?
- Is per-source validation MAE a useful proxy for weighting source contributions?
- At what data amount does Scratch catch up to PRIME?

### Key results

| Weeks | PRIME MAE | Scratch MAE | Winner |
|---|---|---|---|
| 1 | 11.91 | 14.09 | **PRIME (+15.5%)** |
| 2 | 15.39 | 21.82 | **PRIME (+29.5%)** |
| 4 | 15.73 | 20.48 | **PRIME (+23.2%)** |
| 8 | 13.92 | 20.27 | **PRIME (+31.3%)** |
| 16 | 14.46 | 13.66 | Scratch (+5.9%) |
| 32 | 20.84 | 20.86 | Essentially tied |
| 64 | 19.62 | 19.59 | Essentially tied |
| 104 | 20.43 | 19.83 | Scratch marginal |

At 8 weeks, PRIME MAE (13.92) is **31.3% better** than Scratch (20.27). PRIME remains ahead through 8 weeks; Scratch becomes competitive at 16+ weeks. Evaluation snapshot at 8 weeks: **PRIME RMSE = 18.70 vs Scratch RMSE = 22.19** (15.7% RMSE improvement).

### Streaming results

The experiment also evaluates a streaming (online) variant where the model accumulates target data incrementally:

| Weeks | Streaming MAE |
|---|---|
| 1 | 32.32 |
| 2 | 29.75 |
| 4 | 19.14 |
| 8 | 17.78 |
| 16 | 19.66 |
| 32 | 10.33 |

### Output files (in `results/prime/Rat_education_Denise_sweep/`)

| File | Contents |
|---|---|
| `data_efficiency_prime.csv` | PRIME vs Scratch MAE/RMSE at each data level (includes streaming columns) |
| `evaluation_comparison.csv` | Snapshot at 8 weeks: PRIME_Transfer vs PreTransfer |
| `source_rankings.csv` | Full ranked list of Rat/Education candidate sources with composite scores |
| `source_weights.csv` | Final inverse-MAE weights for the 5 selected sources |

### Significance

**PRIME succeeds in the same-site regime.** For Rat/Denise — a same-site, same-type easy target — performance-weighted source blending consistently outperforms Scratch in the 1–8 week data-scarce regime (up to 31.3% MAE improvement at 8 weeks). The performance-weighted blending mechanism correctly downweights poor sources: Beverly (val MAE = 157 kWh) receives weight 0.025 while Earnest (val MAE = 11.8 kWh) receives weight 0.33, providing a meaningfully different initialisation than uniform averaging.

**PRIME converges with Scratch at 16+ weeks.** Once sufficient target data is available, Scratch training becomes equivalently effective and the initialisation advantage diminishes. This is consistent with the pattern observed for all transfer strategies across easy targets.

**Key lesson: domain alignment is required for PRIME to succeed.** An earlier experimental run on Eagle/Brooke (where source and target have stronger distribution mismatch) confirmed that PRIME's performance-weighted blending cannot overcome a fundamental domain gap at low data: PRIME MAE reached 643.5 kWh at 8 weeks (vs Scratch 90.5 kWh, 6.3× worse). The same homogeneous source pool that succeeds for Rat/Denise fails catastrophically on Eagle/Brooke, demonstrating that source-target domain alignment is the critical prerequisite for effective PRIME application.

---

## Experiment Dependency Map

The experiments build on each other. Some models and checkpoints are reused across experiments to avoid redundant training:

```
rat_education (Exp 1)
  └─ baseline_Colin → reused by:
       ├─ switch_modelling (Exp 12)
       ├─ cross_type_transfer as transfer_sametype (Exp 8)
       └─ multitransfer_generalisation single-source (Exp 11)

eagle_education (Exp 3)
  └─ baseline_Samantha → reused by:
       ├─ multi_transfer single-source (Exp 7)
       ├─ cross_type_transfer samesite (Exp 8)
       ├─ ensemble_transfer single-source (Exp 9)
       └─ multitransfer_ablation N=1 (Exp 10)

multi_transfer (Exp 7)
  ├─ baseline_single_eagle_samantha → reused by:
  │    ├─ cross_type_transfer samesite (Exp 8)
  │    └─ ensemble_transfer single-source (Exp 9)
  ├─ baseline_multi_5buildings → reused by:
  │    ├─ ensemble_transfer feature_reference (Exp 9)
  │    ├─ multitransfer_ablation N=5 (Exp 10)
  │    └─ multitransfer_generalisation multi-source (Exp 11)
  └─ pretransfer_Brooke_Nweek_*.ckpt → reused by:
       ├─ cross_type_transfer (Exp 8)
       └─ multitransfer_ablation (Exp 10)

office_any (Exp 5)
  └─ baseline_Hog_Miriam → reused by:
       └─ cross_type_transfer crosstype (Exp 8)
```

---

## Quick Reference

| # | Name | Target | Key question | Script |
|---|---|---|---|---|
| 1 | `rat_education` | Rat/Denise | Does TL work at all? | `run_experiment_suite.py` |
| 2 | `rat_education_new` | Auto Rat/Edu | Replicability? | `run_experiment_suite.py` |
| 3 | `eagle_education` | Auto Eagle/Edu | Does TL hold at another site? | `run_experiment_suite.py` |
| 4 | `lamb_education` | Auto Lamb/Edu | Does TL hold at a third site? | `run_experiment_suite.py` |
| 5 | `office_any` | Auto Office | Does TL work for Office type? | `run_experiment_suite.py` |
| 6 | `lodging_any` | Auto Lodging | Does TL work for Lodging type? | `run_experiment_suite.py` |
| 7 | `multi_transfer` | Eagle/Brooke | Does multi-source fix collapse? | `run_multi_transfer_experiment.py` |
| 8 | `cross_type_transfer` | Eagle/Brooke | Does source type/site match matter? | `run_cross_type_experiment.py` |
| 9 | `ensemble_transfer` | Eagle/Brooke | Joint training vs. model soup? | `run_ensemble_transfer_experiment.py` |
| 10 | `multitransfer_ablation` | Eagle/Brooke | How does N sources scale? | `run_multitransfer_ablation_experiment.py` |
| 11 | `multitransfer_generalisation` | Rat/Denise | Does multi-source help easy targets? | `run_multitransfer_generalisation_experiment.py` |
| 12 | `switch_modelling` | Rat/Denise | Can automatic model selection beat either strategy? | `run_switch_modelling_experiment.py` |
