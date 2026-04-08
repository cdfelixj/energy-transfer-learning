# Documentation Navigation

## Start Here → README.md
Quick start guide, 4-strategy framework overview, 12-experiment roster, full training pipeline, key findings, project structure map.

## Experimental Design → PROJECT_SUMMARY.md
4 fine-tuning strategies (Scratch, Full Fine-Tuning, Frozen Backbone, Adapter), model configurations, research questions, expected results, all 12 experiments with source → target pairs.

## Experiment Reference → EXPERIMENTS.md
Detailed setup for each of the 12 experiments: building selections, hyperparameters, what each experiment answers, output files, dependency map.

## Technical History → TECHNICAL_IMPROVEMENTS.md
Bug-fix chronicle (distribution mismatch, model collapse, early stopping, sequence length), architecture decisions and rationale, Sections 1–9 core fixes, Section 10 — advanced experiment implementations (Frozen Backbone, Adapter, Multi-Transfer, Ensemble Transfer, N-Source Ablation, Cross-Type, Switch Modelling).

## Quick Reference → notes.txt
Training pipeline commands in order, known issues and their fixes.

## Analysis Notebook → notebooks/comprehensive_analysis.ipynb
The complete analysis — 14 sections covering all 12 experiments + PRIME with charts and interpreted findings:
- Section 0: Setup, imports, project overview
- Sections 1–2: Dataset overview, architecture & strategies
- Sections 3–8: Core experiments 1–11 (including collapse analysis, multi-source, ablation)
- Section 9: Experiment 12 — auto-switch modelling (RMSE 22.72 vs oracle 22.70)
- Section 10: PRIME experiment (negative result, source homogeneity analysis)
- Section 11: Cross-experiment summary (master heatmap, transfer benefit heatmap)
- Sections 12–14: Discussion, Limitations, Conclusion
