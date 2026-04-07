"""
Building Auto-Discovery Script

For each experiment category, identifies the best source and target buildings
by measuring electricity data completeness (% non-null hourly readings).

Experiment categories:
  1. rat_education     - Rat site, Education  (existing: Colin→Denise, excluded from auto-select)
  2. rat_education_new - Rat site, Education  (exclude Colin & Denise)
  3. eagle_education   - Eagle site, Education
  4. lamb_education    - Lamb site, Education
  5. office_any        - Any site, Office
  6. lodging_any       - Any site, Lodging/residential

Outputs:
  results/experiments/building_selections.csv
"""

import os
import sys
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import pandas as pd
import numpy as np
from data_loader import load_electricity_data


def compute_completeness(electricity_df, min_weeks=16):
    """Return sorted Series of completeness % for each building, filtered to >= min_weeks of data."""
    total = len(electricity_df)
    completeness = electricity_df.notna().sum() / total * 100

    # Also check absolute amount: need at least min_weeks * 7 * 24 non-null readings
    min_readings = min_weeks * 7 * 24
    sufficient = electricity_df.notna().sum() >= min_readings
    completeness = completeness[sufficient]

    return completeness.sort_values(ascending=False)


def select_pair(completeness, exclude=None, n=2):
    """Pick top-n buildings from completeness, optionally excluding certain IDs."""
    exclude = set(exclude or [])
    candidates = [b for b in completeness.index if b not in exclude]
    if len(candidates) < n:
        raise ValueError(
            f"Not enough candidates after exclusion. Found {len(candidates)}, need {n}.\n"
            f"Candidates: {candidates}"
        )
    return candidates[:n]  # [source, target]


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(project_root, 'results', 'experiments')
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Define experiment categories
    # ------------------------------------------------------------------ #
    categories = [
        {
            'name': 'rat_education',
            'site_id': 'Rat',
            'building_type': 'Education',
            'fixed_source': 'Rat_education_Colin',
            'fixed_target': 'Rat_education_Denise',
            'exclude': [],
        },
        {
            'name': 'rat_education_new',
            'site_id': 'Rat',
            'building_type': 'Education',
            'fixed_source': None,
            'fixed_target': None,
            'exclude': ['Rat_education_Colin', 'Rat_education_Denise'],
        },
        {
            'name': 'eagle_education',
            'site_id': 'Eagle',
            'building_type': 'Education',
            'fixed_source': None,
            'fixed_target': None,
            'exclude': [],
        },
        {
            'name': 'lamb_education',
            'site_id': 'Lamb',
            'building_type': 'Education',
            'fixed_source': None,
            'fixed_target': None,
            'exclude': [],
        },
        {
            'name': 'office_any',
            'site_id': None,          # any site
            'building_type': 'Office',
            'fixed_source': None,
            'fixed_target': None,
            'exclude': [],
        },
        {
            'name': 'lodging_any',
            'site_id': None,          # any site
            'building_type': 'Lodging/residential',
            'fixed_source': None,
            'fixed_target': None,
            'exclude': [],
        },
    ]

    selections = []

    for cat in categories:
        print(f"\n{'='*70}")
        print(f"  Category: {cat['name']}")
        print(f"  Site: {cat['site_id'] or 'Any'}  |  Type: {cat['building_type']}")
        print(f"{'='*70}")

        if cat['fixed_source'] and cat['fixed_target']:
            # Existing experiment — just validate and record
            source = cat['fixed_source']
            target = cat['fixed_target']
            try:
                electricity, _, valid = load_electricity_data(
                    site_id=cat['site_id'], building_type=cat['building_type']
                )
                comp = compute_completeness(electricity, min_weeks=16)
                src_comp = comp.get(source, np.nan)
                tgt_comp = comp.get(target, np.nan)
            except Exception as e:
                print(f"  Warning: Could not compute completeness: {e}")
                src_comp = tgt_comp = np.nan

            print(f"  Fixed pair: {source} ({src_comp:.1f}%) → {target} ({tgt_comp:.1f}%)")
        else:
            electricity, _, valid = load_electricity_data(
                site_id=cat['site_id'], building_type=cat['building_type']
            )
            comp = compute_completeness(electricity, min_weeks=16)

            if len(comp) == 0:
                print(f"  ERROR: No buildings with >=16 weeks of data found!")
                continue

            print(f"  Available buildings (sorted by completeness):")
            for b, c in comp.head(10).items():
                print(f"    {b}: {c:.1f}%")

            try:
                pair = select_pair(comp, exclude=cat['exclude'], n=2)
            except ValueError as e:
                print(f"  ERROR: {e}")
                continue

            source, target = pair
            src_comp = comp[source]
            tgt_comp = comp[target]
            print(f"\n  Selected: {source} ({src_comp:.1f}%) → {target} ({tgt_comp:.1f}%)")

        selections.append({
            'experiment_name': cat['name'],
            'site_id': cat['site_id'] or 'Any',
            'building_type': cat['building_type'],
            'source_building': source,
            'target_building': target,
            'source_completeness_pct': round(src_comp, 2) if not np.isnan(src_comp) else np.nan,
            'target_completeness_pct': round(tgt_comp, 2) if not np.isnan(tgt_comp) else np.nan,
        })

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(selections)
    out_path = os.path.join(out_dir, 'building_selections.csv')
    df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print("  BUILDING SELECTION SUMMARY")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\n✓ Saved to: {out_path}")


if __name__ == '__main__':
    main()
