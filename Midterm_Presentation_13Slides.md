# MIDTERM PRESENTATION: 13-Slide Structure
## Few-Shot Transfer Learning for Building Energy Forecasting

---

## SLIDE 1: Title Slide
**[VISUAL: Clean title slide with university logo]**

### Main Title (Large, Bold)
Few-Shot Transfer Learning for Building Energy Forecasting

### Subtitle
Rapid Model Adaptation with Minimal Historical Data

### Author & Details (Bottom)
Felix Lau Pangestu | Student ID: 23222727
Supervisor: Dr. YU, Wilson Shih Bun
Observer: Professor Samson Tai Kin Hon
[University Name & Date]

---

## SLIDE 2: Background Part 1 - Energy Crisis & Motivation
**[VISUAL: World map showing energy consumption by region, or building silhouettes with energy icons]**

### Title
Why Building Energy Forecasting Matters

### Content (Bullet Points)
• Buildings consume 30% of global energy and account for 27% of CO₂ emissions
• Energy forecasting enables three critical outcomes:
  - Demand response programs: Reduce peak consumption and stabilize grids
  - Efficiency optimization: Identify waste and reduce operational costs
  - Grid planning: Enable renewable energy integration and storage deployment
• Traditional barrier: Deploying models to new buildings requires 6–12 months of data collection
  - Cost: High data collection and infrastructure expenses
  - Time: Months before actionable insights
  - Scalability: Impractical for rapid deployment across hundreds of buildings

### Bottom Key Message
**The Problem:** Energy prediction models cannot deploy quickly; data collection is the bottleneck.

---

## SLIDE 3: Background Part 2 - Transfer Learning Solution
**[VISUAL: Diagram showing source building → knowledge transfer → target building]**

### Title
How Transfer Learning Solves the Deployment Problem

### Content (Bullet Points)
• Transfer learning: Leveraging knowledge from existing buildings (source) to rapidly adapt models to new buildings (target)

### Three Key Advantages (Numbered List)
1. **Data Efficiency:** Achieve accurate predictions with weeks instead of months of target building data
2. **Cost Reduction:** Reduce deployment timeline from 6-12 months to 2-4 weeks; lower data collection costs
3. **Generalization:** Source building knowledge (daily/weekly patterns, weather relationships) applies across diverse building types

### Why It Works
• Core mechanism: Early LSTM layers learn generic energy patterns (applicable everywhere); later layers adapt to building-specific details
• Physics-based intuition: Thermodynamic relationships and occupancy-energy coupling are invariant across buildings

### Bottom Key Message
**The Solution:** Transfer learning unlocks rapid, cost-effective deployment by reusing knowledge from existing buildings.

---

## SLIDE 4: Methodology Part 1 - What is Transfer Learning?
**[VISUAL: Two parallel paths diagram - "Train from Scratch" vs "Transfer Learning"]**

### Title
Transfer Learning vs. Training from Scratch

### Left Column: Training from Scratch
```
New Building Data
        ↓
    [Random Weights]
        ↓
   Train LSTM (50 epochs)
        ↓
   Result: Weak Model
   (R² ≈ -0.00, MAE ≈ 17 kWh)
```

### Right Column: Transfer Learning (Fine-Tuning)
```
Existing Building (2 years)
        ↓
   Train LSTM (50 epochs)
        ↓
[Pre-Trained Baseline Model]
        ↓
New Building Data + Fine-tune (100 epochs)
        ↓
Result: Strong Model
(R² ≈ 0.25, MAE ≈ 12.4 kWh)
```

### Key Difference
**Fine-tuning starts with learned weights (not random); therefore requires less data to converge to good solutions.**

---

## SLIDE 5: Methodology Part 2 - Transfer Learning Techniques
**[VISUAL: Three horizontal boxes showing different approaches]**

### Title
Three Transfer Learning Approaches Compared

### Box 1: Fine-Tuning (Status: Implemented ✓)
• Unfreeze all LSTM layers
• Train all parameters on target data
• Lower learning rate (1×10⁻⁴ vs 1×10⁻³)
• Pros: Best accuracy (26.8% improvement)
• Cons: Requires more data; longer training time
• Use Case: Standard deployment (2-4 weeks)

### Box 2: Adapter Layers (Status: Planned)
• Freeze baseline LSTM weights
• Add small trainable adapters (64→16→64)
• Only adapters + final layer trainable
• Pros: 50-90% fewer parameters; faster training
• Cons: Slightly lower accuracy (expected 20% improvement)
• Use Case: Resource-constrained environments

### Box 3: Frozen Backbone (Status: Planned)
• Freeze entire LSTM
• Train only final output layer
• Fastest training, minimal parameters
• Pros: Extreme speed; minimal memory
• Cons: Lower accuracy (expected 10-15% improvement)
• Use Case: Ultra-fast deployment (<1 minute)

### Bottom Key Message
**Current focus: Fine-tuning (best accuracy). Future: Compare trade-offs across all three.**

---

## SLIDE 6: Objectives & Research Questions
**[VISUAL: Four colored boxes, one per research question]**

### Title
Project Objectives & Research Questions

### Primary Objective
Develop and evaluate transfer learning methods enabling rapid adaptation of building energy models to new buildings with minimal historical data.

### Four Core Research Questions

**RQ 1.1: Data Efficiency Curve**
What is the minimum data needed? At what point does transfer learning achieve 90% of full-data performance?
📊 *Status: ANSWERED* - Optimal at 8 weeks; 25.3% improvement over baseline

**RQ 2.1: Building Type Variation**
How does transfer success vary across different building types (offices, schools, hospitals, retail, labs)?
📊 *Status: In Progress* - Testing on 10+ buildings

**RQ 3.1: Method Comparison**
Which transfer approach (fine-tuning vs. adapters vs. frozen) maximizes accuracy-efficiency trade-offs?
📊 *Status: Planned* - Implementation underway

**RQ 5.4: Robustness Under Failures**
How do transferred models degrade under real-world failures (20% missing data, equipment changes, occupancy shocks)?
📊 *Status: Planned* - Testing in next phase

---

## SLIDE 7: Data & Preprocessing
**[VISUAL: Building Data Genome 2 logo/diagram; world map showing 19 sites]**

### Title
Data Source: Building Data Genome 2 (BDG2)

### Dataset Overview
• **1,636 non-residential buildings** across North America and Europe
• **2 years of hourly data** (January 2016 – December 2017)
• **3,053 total energy meters** (electricity, chilled water, steam, gas)
• **19 geographic sites** providing climate diversity
• **Standardized metadata:** Building type, size, location, LEED certification
• **Weather data:** Hourly temperature, humidity, cloud cover

### Current Focus: Two Education Buildings
• **Source Building (Colin):** 2 years complete data, used for baseline pre-training
• **Target Building (Denise):** Testing transfer learning with 1-104 weeks of data

### Data Cleaning & Preprocessing
1. **Missing Value Handling:** Linear interpolation for gaps ≤3 hours; drop remaining NaN; exclude buildings with excessive missing data
2. **Outlier Detection:** Remove values >10× 95th percentile; remove extended zero periods (>72 consecutive hours); remove negative values
3. **Normalization:** StandardScaler per building on features only (target 'energy' remains unnormalized for interpretability)
4. **Feature Engineering:**
   - Time features: Cyclical encoding (sin/cos) for hour, day-of-week, month, day-of-year
   - Lag features: Energy at t-1, t-2, t-24h, t-48h, t-72h, t-168h, t-336h, t-720h (up to 30 days)
   - Rolling statistics: 24h and 168h mean/std/min/max
   - Binary flags: is_weekend, is_business_hours
5. **Temporal Split Strategy:** Stratified random split by month (60% train, 20% val, 20% test) to ensure balanced seasonal representation

### Key Discovery (Critical Fix)
❌ **Problem:** Chronological splitting created 52% distribution mismatch (training on winter, testing on summer break)
✓ **Solution:** Stratified random split reduced shift to <1%; enabled valid model training

---

## SLIDE 8: Progress - Experimental Design
**[VISUAL: Four-way comparison table with color coding]**

### Title
Experimental Framework: Four-Way Comparison

### Table Layout
| Model | Training Building | Data Amount | Transfer? | Purpose |
|-------|---|---|---|---|
| **Baseline-Source** | Colin (source) | 2 years | No | Best-case reference |
| **Baseline-Target** | Denise (target) | 2 years | No | Quantify domain shift |
| **Pre-Transfer** | Denise (target) | 2 months | No | Control (from scratch) |
| **Transfer** | Denise (target) | 2 months | Yes | Experimental (fine-tuned) |

### Experimental Controls
✓ Pre-Transfer and Transfer use identical data, building, test set, and architecture
✓ Only difference: initialization (random vs. pre-trained)
✓ This isolates the pure transfer learning effect

### Key Finding from Design
Domain shift is severe: Baseline-Source achieves MAE=14.48 kWh on Colin, but MAE=36.22 kWh on Denise (150% penalty).
→ This proves zero-shot transfer impossible; fine-tuning essential.

---

## SLIDE 9: Progress - Core Results (2 Months Data)
**[VISUAL: Four-metric bar chart comparison (MAE, RMSE, R², MAPE)]**

### Title
Transfer Learning Effectiveness: 2 Months of Target Data

### Results Table
| Metric | Pre-Transfer | Transfer | Improvement |
|--------|---|---|---|
| **MAE (kWh)** | 17.32 | 12.45 | **-4.87 (28.1%)** ✓ |
| **RMSE (kWh)** | 20.51 | 17.76 | **-2.75 (13.4%)** ✓ |
| **R² Score** | -0.0024 | 0.2488 | **+0.2512** ✓ |
| **MAPE (%)** | 23.66 | 15.96 | **-7.70 (32.6%)** ✓ |
| **Median AE (kWh)** | 16.90 | 8.08 | **-8.82 (52.2%)** ✓ |

### Interpretation
✓ All metrics show consistent improvement in same direction
✓ **28.1% MAE reduction** = 4.87 kWh less error per prediction (practical significance)
✓ **R² improvement** from nearly zero (useless) to 0.249 (explains 24.9% of variance; acceptable for building energy)
✓ **Median AE 52.2% reduction** = Transfer eliminates largest errors, not just averages

### What This Means
With only 2 months of target data, transfer learning produces a working model. Training from scratch produces a non-functional model (R²≈0). **This proves transfer learning's core value.**

---

## SLIDE 10: Progress - Data Efficiency Analysis (RQ 1.1 - Main Finding)
**[VISUAL: Multi-panel line chart showing MAE, RMSE, R², MAPE curves for 1-104 weeks]**

### Title
Data Efficiency: Transfer Learning Across Different Data Amounts

### Key Findings (Bullet Points)

**Three Critical Patterns Discovered:**

1. **Minimum Viable Data: 2 Weeks**
   - At 1 week: Transfer shows -0.7% (no benefit)
   - At 2 weeks: Transfer shows +16.9% improvement (inflection point)
   - Interpretation: Need at least 2 weeks before pre-trained weights help

2. **Optimal Sweet Spot: 8 Weeks** ⭐ **MOST IMPORTANT FINDING**
   - Transfer MAE reaches minimum: 12.91 kWh
   - Transfer improvement peaks: 25.3% over pre-transfer
   - R² = 0.230 (highest variance explained)
   - Interpretation: 8 weeks = best balance between "enough target data" and "retains source knowledge"

3. **Surprising Collapse: 16+ Weeks** ⚠️ **CRITICAL ANOMALY**
   - Week 16: MAE increases to 16.31 kWh (+26% worse than week 8)
   - Week 32: MAE = 23.89 kWh (+85% worse; below pre-transfer!)
   - Week 104: MAE = 22.11 kWh (still degraded)
   - Interpretation: More data paradoxically hurts transfer learning
   - Hypotheses: (a) Catastrophic forgetting, (b) Architecture saturation, (c) Learning rate mismatch
   - **Action:** Will retrain with higher learning rates and larger architecture

### Data Efficiency Multiplier
At 8 weeks: Transfer learning achieves with 8 weeks what from-scratch needs 26 weeks.
→ **3.2× data efficiency gain**

---

## SLIDE 11: Progress - Technical Challenges Overcome
**[VISUAL: Three-column layout showing Problem → Root Cause → Solution for each issue]**

### Title
Technical Improvements: From Broken Models to Working Framework

### Issue 1: Data Distribution Mismatch
| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Train mean: 60.85 kWh | Chronological split | Stratified random split |
| Test mean: 29.20 kWh | separated seasons | by month (random_state=42) |
| 52% shift ❌ | (winter vs. summer break) | <1% shift ✓ |
| R² = negative | Model couldn't learn | R² = 0.36+ |

### Issue 2: Model Collapse (Pre-Transfer)
| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Constant predictions | 353K params on | Simplified: 128/3 |
| Std = 0.00 kWh ❌ | 720 samples (490:1 ratio) | → 64/2 layers |
| R² = -0.08 | Architecture too large | Increased data: |
| | for data amount | 1mo → 2mo (1,440 samples) |
| | | Now Std = 17.45 ✓ |

### Issue 3: Premature Training Convergence
| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Training stopped | Early stopping | Baseline: patience |
| at epoch 7 | patience too low | 10 → 15 epochs |
| R² = -0.09 ❌ | Too aggressive | Pre-Transfer: patience 10 |
| | High LR caused | Transfer: patience 5 |
| | oscillations | LR: 5e-4 (baseline) ✓ |
| | | Result: R² = 0.36 ✓ |

### Impact
These three fixes transformed non-functional models → valid experimental framework.

---

## SLIDE 12: Progress - Architecture & Model Specifications
**[VISUAL: Two side-by-side LSTM architecture diagrams with layer counts and parameters]**

### Title
Model Architectures: Baseline vs. Limited-Data Variants

### Architecture Comparison Table
| Component | Baseline-Source | Pre-Transfer | Transfer |
|-----------|---|---|---|
| **Input Features** | 31 (energy + lags + weather + time) | 31 | 31 |
| **Hidden Size (Layer 1)** | 128 units | 64 units | 64 units |
| **Hidden Size (Layer 2)** | 128 units | 64 units | 64 units |
| **Hidden Size (Layer 3)** | 128 units | — | — |
| **Total Parameters** | ~353K | ~88K | ~88K |
| **Sequence Length** | 168 hours (1 week) | 24 hours (1 day) | 24 hours (1 day) |
| **Dropout** | 0.2 | 0.2 | 0.2 |
| **Learning Rate** | 5×10⁻⁴ | 1×10⁻³ | 1×10⁻⁴ |
| **Training Data** | 2 years (17,520h) | 2 months (1,440h) | 2 months (1,440h) |
| **Epochs** | 50 | 100 | 50 |
| **Early Stop Patience** | 15 | 10 | 5 |

### Design Rationale
✓ **Baseline:** Larger architecture (128/3) leverages abundant data (2 years); 168h sequences capture weekly patterns
✓ **Pre-Transfer/Transfer:** Simplified (64/2) prevents overfitting on limited data; 24h sequences provide more training examples
✓ **Learning rates:** Transfer uses 10× lower rate (1e-4) to preserve pre-trained knowledge with small updates

---

## SLIDE 13: Future Plans & Next Steps
**[VISUAL: Timeline/roadmap showing remaining phases]**

### Title
Future Phases: Completing the Research

### Phase III (Next - 2 Weeks): Answer RQ 2.1 - Building Type Variation
**Goal:** Test transfer learning on diverse buildings to understand which types benefit most

**Actions:**
• Select 10-15 Education buildings from Rat site (current filtered dataset)
• Replicate 8-week transfer experiment on each building
• Analyze: Does 8-week peak hold across buildings? Which building characteristics predict transfer success?
• Compare building metadata: size, LEED status, occupancy patterns

**Expected Outcome:** Performance distribution across Education buildings; identify characteristics that predict transfer learning success

---

### Phase IV (Following - 3 Weeks): Resolve the 16+ Week Collapse & Answer RQ 3.1

**Part A: Debug the Anomaly**
• Retrain Transfer models at 16, 32, 64, 104 weeks with:
  - Higher learning rates (5e-4, 1e-3)
  - Larger architecture (128/3 layers)
  - More epochs (100 instead of 50)
  - Learning rate scheduling
• Hypothesis testing: Which factor fixes degradation?

**Part B: Method Comparison**
• Implement Adapter Layers approach (freezing baseline, adding trainable adapters)
• Implement Frozen Backbone approach (only output layer trainable)
• Compare all three on accuracy, training time, memory usage
• Trade-off analysis: accuracy vs. efficiency

**Expected Outcome:** Method selection guidelines; best practice recommendations per deployment scenario

---

### Phase V (Final - 3 Weeks): Robustness Testing & Paper Writing (RQ 5.4)

**Robustness Testing:**
• Introduce realistic failures to trained models:
  - 10%, 20% missing data (dropped readings)
  - ±5% sensor noise (measurement error)
  - ±15% equipment degradation (HVAC efficiency loss)
  - 50% occupancy shock (pandemic scenario)
• Measure: How does each failure type degrade accuracy? Rank by impact.

**Paper Writing:**
• Compile results into research paper
• Key contributions: Data efficiency curve, multi-building validation, robustness analysis
• Submit to building/energy AI journals

**Expected Outcome:** Complete research answering all four RQs; publication-ready manuscript

---

### Key Milestones Remaining
✓ Completed: Baseline development, Pre-Transfer implementation, Transfer learning evaluation (8 weeks)
📍 In Progress: Data efficiency analysis (1-104 weeks)
⏳ Next: Multi-building evaluation (RQ 2.1)
⏳ Following: Debug collapse, method comparison (RQ 3.1)
⏳ Final: Robustness testing (RQ 5.4), paper writing

### Timeline
**Mid-Year Point:** Today (50% complete)
**Final Submission:** 6 weeks remaining
**Anticipated completion:** End of February 2026

---

### Conclusion Slide (Optional 14th Slide)

### Title
Progress Summary & Key Takeaway

### Summary
✓ **Completed:** Established valid experimental framework; demonstrated transfer learning's core effectiveness (26.8% MAE improvement with 2 months data)
✓ **Discovered:** Optimal deployment window is 8 weeks; 3.2× data efficiency gain
⚠️ **Identified:** Unexpected collapse at 16+ weeks (investigating)
📍 **Next:** Scale to multiple buildings; complete method comparison; test robustness

### Key Takeaway
Transfer learning is a practical, impactful solution for rapid building energy model deployment. With proper implementation, models achieve acceptable accuracy (R²≈0.25) in 8 weeks—reducing traditional 6-12 month timelines to 2-month deployment windows.

### Questions?

---

# PRESENTATION NOTES FOR YOU

## Visual Assets to Use (from your files)
- Slide 9: Use file:28 (4-metric bar chart comparison)
- Slide 10: Use file:41 (multi-panel line chart with 6 subplots)
- Slide 10 (bottom): Use file:44 (data efficiency bar chart showing weeks comparison)
- Slide 11: file:43 (heatmaps) for showing performance variation
- Slide 4-5: Create simple diagrams (or describe verbally if slides are text-only)

## Delivery Tips
1. **Slide 1:** Show title, make eye contact, introduce yourself and supervisors
2. **Slides 2-3 (Background):** Start slow; build emotional case (climate change, energy crisis)
3. **Slides 4-7 (Methodology):** Technical but accessible; explain LSTM simply: "A type of neural network good at learning from time series"
4. **Slides 8-12 (Progress):** YOUR STRENGTH; dive into numbers, show graphs, tell the story of discovery
5. **Slide 13 (Future):** Be confident; show you have clear next steps
6. **Questions:** Prepare for: "Why does the model collapse at 16 weeks?" (Your honest answer: "Investigating now; likely overfitting or learning rate issue")

## Time Allocation (13 slides, assume 20-30 min presentation)
- Slide 1: 1 min (title)
- Slides 2-3: 4 min (background)
- Slides 4-7: 5 min (methodology)
- Slides 8-12: 15 min (progress - your main content)
- Slide 13: 3 min (future plans)
- Questions: 5-10 min

Good luck! 🎓
