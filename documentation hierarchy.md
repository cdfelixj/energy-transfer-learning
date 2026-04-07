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

## Analysis Notebook → notebooks/model_evaluation_analysis.ipynb
Interactive analysis — 16 sections covering all 12 experiments and all 4 strategies:
- Sections 1–6: Core experiment data efficiency curves
- Sections 7–9: Ensemble, Cross-Type, N-Source Ablation
- Sections 10–12: Architecture details, parameter counts
- Sections 13–15: Multi-Transfer, Generalisation, Cross-Type deep dive
- Section 16: Switch Modelling (auto-selection results)
