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
from sklearn.metrics.pairwise import cosine_similarity
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


def compute_building_profile(electricity_df: pd.DataFrame, building_id: str) -> np.ndarray:
    """Build a 43-dimensional normalised energy profile vector for a building.

    Concatenates three sub-profiles, each L2-normalised:
        - 24-dim: mean energy by hour-of-day  (daily pattern)
        -  7-dim: mean energy by day-of-week   (weekly pattern)
        - 12-dim: mean energy by month         (seasonal pattern)

    Returns a (43,) float32 array, or zeros if data is insufficient.
    """
    if building_id not in electricity_df.columns:
        return np.zeros(43, dtype=np.float32)

    series = electricity_df[building_id].dropna()
    if len(series) < 168:  # need at least one week of data
        return np.zeros(43, dtype=np.float32)

    idx = series.index
    hourly  = series.groupby(idx.hour).mean().reindex(range(24), fill_value=0.0).values
    daily   = series.groupby(idx.dayofweek).mean().reindex(range(7), fill_value=0.0).values
    monthly = series.groupby(idx.month).mean().reindex(range(1, 13), fill_value=0.0).values

    def _l2(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    return np.concatenate([_l2(hourly), _l2(daily), _l2(monthly)]).astype(np.float32)


def score_source_candidates(
    electricity_df: pd.DataFrame,
    target_id: str,
    candidates: list[str],
    metadata: pd.DataFrame,
    completeness: pd.Series,
) -> pd.Series:
    """Rank candidate source buildings using a composite multi-factor score.

    Score = 0.30 × completeness_pct
          + 0.20 × type_match  (100 if same building type, else 0)
          + 0.20 × site_match  (100 if same site_id, else 0)
          + 0.30 × profile_similarity (cosine similarity × 100)

    Args:
        electricity_df: Full electricity DataFrame with building columns.
        target_id:      Building ID of the target building.
        candidates:     List of candidate source building IDs to rank.
        metadata:       Building metadata DataFrame with 'building_id',
                        'building_type', and 'site_id' columns.
        completeness:   Series of completeness % indexed by building ID.

    Returns:
        pd.Series of composite scores, sorted descending, indexed by building ID.
    """
    def _meta(building_id: str, column: str, default=''):
        row = metadata[metadata['building_id'] == building_id]
        return row[column].values[0] if len(row) > 0 else default

    target_type = _meta(target_id, 'primaryspaceusage')
    target_site = _meta(target_id, 'site_id')
    target_profile = compute_building_profile(electricity_df, target_id).reshape(1, -1)

    scores = {}
    for bid in candidates:
        completeness_score = float(completeness.get(bid, 0.0))

        type_match = 100.0 if _meta(bid, 'primaryspaceusage') == target_type else 0.0
        site_match = 100.0 if _meta(bid, 'site_id') == target_site else 0.0

        src_profile = compute_building_profile(electricity_df, bid).reshape(1, -1)
        if np.any(target_profile != 0) and np.any(src_profile != 0):
            sim = float(cosine_similarity(target_profile, src_profile)[0, 0])
        else:
            sim = 0.0
        profile_score = max(sim, 0.0) * 100.0  # clamp to [0, 100]

        scores[bid] = (
            0.30 * completeness_score
            + 0.20 * type_match
            + 0.20 * site_match
            + 0.30 * profile_score
        )

    return pd.Series(scores).sort_values(ascending=False)


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


def select_sources(
    electricity_df: pd.DataFrame,
    target_id: str,
    metadata: pd.DataFrame,
    completeness: pd.Series,
    exclude: list[str] | None = None,
    n_sources: int = 5,
) -> list[str]:
    """Select the top-N source buildings for PRIME using multi-factor ranking.

    Args:
        electricity_df: Full electricity DataFrame.
        target_id:      Target building ID (always excluded from sources).
        metadata:       Building metadata DataFrame.
        completeness:   Series of completeness % indexed by building ID.
        exclude:        Additional building IDs to exclude.
        n_sources:      Number of source buildings to return.

    Returns:
        List of up to n_sources building IDs, ranked by composite score.
    """
    exclude_set = set(exclude or []) | {target_id}
    candidates = [b for b in completeness.index if b not in exclude_set]

    if not candidates:
        return []

    scores = score_source_candidates(
        electricity_df, target_id, candidates, metadata, completeness
    )
    return scores.index.tolist()[:n_sources]


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

        electricity, metadata, valid = load_electricity_data(
            site_id=cat['site_id'], building_type=cat['building_type']
        )
        comp = compute_completeness(electricity, min_weeks=16)

        if cat['fixed_source'] and cat['fixed_target']:
            # Existing experiment — validate, record, and compute ranked sources for PRIME
            source = cat['fixed_source']
            target = cat['fixed_target']
            src_comp = comp.get(source, np.nan)
            tgt_comp = comp.get(target, np.nan)
            print(f"  Fixed pair: {source} ({src_comp:.1f}%) → {target} ({tgt_comp:.1f}%)")

            # Rank sources for PRIME (excluding the fixed target)
            prime_sources = select_sources(
                electricity, target, metadata, comp,
                exclude=cat['exclude'], n_sources=5,
            )
            prime_source_scores = score_source_candidates(
                electricity, target,
                [b for b in comp.index if b != target and b not in set(cat['exclude'])],
                metadata, comp,
            )
        else:
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

            source, target = pair[0], pair[1]
            src_comp = comp[source]
            tgt_comp = comp[target]
            print(f"\n  Selected: {source} ({src_comp:.1f}%) → {target} ({tgt_comp:.1f}%)")

            # Rank top-5 sources for PRIME (excluding target + any other excludes)
            prime_sources = select_sources(
                electricity, target, metadata, comp,
                exclude=cat['exclude'], n_sources=5,
            )
            prime_source_scores = score_source_candidates(
                electricity, target,
                [b for b in comp.index if b != target and b not in set(cat['exclude'])],
                metadata, comp,
            )

        print(f"  PRIME sources (ranked):")
        for rank, bid in enumerate(prime_sources, 1):
            score = prime_source_scores.get(bid, np.nan)
            print(f"    {rank}. {bid}  (score={score:.1f})")

        selections.append({
            'experiment_name': cat['name'],
            'site_id': cat['site_id'] or 'Any',
            'building_type': cat['building_type'],
            'source_building': source,
            'target_building': target,
            'source_completeness_pct': round(src_comp, 2) if not np.isnan(src_comp) else np.nan,
            'target_completeness_pct': round(tgt_comp, 2) if not np.isnan(tgt_comp) else np.nan,
            'prime_sources': ','.join(prime_sources),
            'prime_source_scores': ','.join(
                f"{prime_source_scores.get(b, 0):.1f}" for b in prime_sources
            ),
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
