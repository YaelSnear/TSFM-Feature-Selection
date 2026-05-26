"""Compute pairwise inter-layer CKA for all cached Chronos-2 embeddings.

Usage:
    python scripts/diagnose_layer_similarity.py \
        --cache_dir outputs/EXP_metr_la_lag_2h_20260525_131114/cache

Output:
    Per-series CKA table:  sha1  CKA(L6,L8)  CKA(L6,L10)  CKA(L8,L10)
    Summary row:  mean ± std across all series
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scoring.tsfm_scorers import _cka_core


def load_mean_pooled(path: Path) -> np.ndarray:
    """Load [N, P, D] from .npz and mean-pool patches → [N, D]."""
    arr = np.load(path)["emb"]          # [N, P, D]
    return arr.mean(axis=1)             # [N, D]


def main(cache_dir: str) -> None:
    cache = Path(cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache}")

    # Group files by SHA1 prefix → {sha1: {layer: Path}}
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for f in sorted(cache.glob("*.npz")):
        sha, layer_str = f.stem.rsplit("_layer", 1)
        groups[sha][int(layer_str)] = f

    required = {6, 8, 10}
    complete = {sha: lmap for sha, lmap in groups.items() if required.issubset(lmap)}
    print(f"Found {len(complete)} complete triplets (L6+L8+L10) out of {len(groups)} series.\n")
    print(f"{'SHA1':10s}  {'CKA(L6,L8)':>12s}  {'CKA(L6,L10)':>13s}  {'CKA(L8,L10)':>13s}")
    print("-" * 56)

    rows = []
    for sha in sorted(complete):
        lmap = complete[sha]
        H6  = load_mean_pooled(lmap[6])
        H8  = load_mean_pooled(lmap[8])
        H10 = load_mean_pooled(lmap[10])
        c68  = _cka_core(H6, H8)
        c610 = _cka_core(H6, H10)
        c810 = _cka_core(H8, H10)
        rows.append((c68, c610, c810))
        print(f"{sha[:10]:10s}  {c68:12.4f}  {c610:13.4f}  {c810:13.4f}")

    if not rows:
        print("No complete triplets found.")
        return

    arr = np.array(rows)
    print("\n" + "=" * 56)
    print("SUMMARY (mean ± std across all series):")
    labels = ["CKA(L6,L8)", "CKA(L6,L10)", "CKA(L8,L10)"]
    for i, lbl in enumerate(labels):
        print(f"  {lbl:14s}: {arr[:, i].mean():.4f} ± {arr[:, i].std():.4f}"
              f"  [min={arr[:, i].min():.4f}, max={arr[:, i].max():.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inter-layer CKA diagnostic for Chronos-2 cache")
    ap.add_argument("--cache_dir", required=True, help="Path to experiment cache directory")
    main(ap.parse_args().cache_dir)
