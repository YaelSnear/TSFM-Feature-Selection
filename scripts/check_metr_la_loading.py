"""METR-LA loading sanity check.

Usage (from project root):
    python scripts/check_metr_la_loading.py --config configs/metr_la_loading_check.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.data.real_traffic import load_metr_la


def _freq_label(index) -> str:
    try:
        freq = index.inferred_freq
        return freq if freq else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(cfg.data.zip_path)
    n_rows   = cfg.data.n_rows
    print(f"Loading {zip_path}  (n_rows={n_rows}) …")

    data = load_metr_la(zip_path, n_rows=n_rows)
    df   = data.df

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print("=== zip contents ===")
    for name in data.metadata["zip_contents"]:
        print(f"  {name}")

    print()
    print("=== detected files ===")
    print(f"  data file : {data.metadata['data_file']}")
    print(f"  adj  file : {data.metadata['adj_file']}")

    print()
    print("=== dataframe ===")
    print(f"  shape          : {df.shape}  (timestamps × sensors)")
    print(f"  first timestamp: {df.index[0]}")
    print(f"  last timestamp : {df.index[-1]}")
    print(f"  inferred freq  : {_freq_label(df.index)}")
    print(f"  n_sensors      : {data.n_sensors}")
    print(f"  dtype          : {df.dtypes.unique()[0]}")

    values    = df.values
    nan_count = int(np.isnan(values).sum())
    nan_frac  = nan_count / values.size
    print()
    print("=== data quality ===")
    print(f"  NaN count    : {nan_count}  ({nan_frac*100:.2f}%)")
    finite    = values[np.isfinite(values)]
    print(f"  min          : {finite.min():.4f}")
    print(f"  max          : {finite.max():.4f}")
    print(f"  mean         : {finite.mean():.4f}")
    print(f"  std          : {finite.std():.4f}")
    zero_frac = float((finite == 0).sum() / finite.size)
    print(f"  zero fraction: {zero_frac*100:.2f}%")

    if data.adj_matrix.size > 0:
        print()
        print("=== adjacency matrix ===")
        print(f"  shape        : {data.adj_matrix.shape}")
        print(f"  min/max      : {data.adj_matrix.min():.4f} / {data.adj_matrix.max():.4f}")
        nonzero = int((data.adj_matrix > 0).sum())
        print(f"  non-zero entries: {nonzero}")
        print(f"  sensor_id_to_idx size: {len(data.sensor_id_to_idx)}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    summary = {
        "zip_path":         str(zip_path),
        "zip_contents":     data.metadata["zip_contents"],
        "detected_data_file": data.metadata["data_file"],
        "detected_adj_file":  data.metadata["adj_file"],
        "shape":            list(df.shape),
        "n_timestamps":     data.n_timestamps,
        "n_sensors":        data.n_sensors,
        "first_timestamp":  str(df.index[0]),
        "last_timestamp":   str(df.index[-1]),
        "inferred_freq":    _freq_label(df.index),
        "nan_count":        nan_count,
        "nan_fraction":     round(nan_frac, 6),
        "data_min":         round(float(finite.min()), 4),
        "data_max":         round(float(finite.max()), 4),
        "data_mean":        round(float(finite.mean()), 4),
        "data_std":         round(float(finite.std()), 4),
        "zero_fraction":    round(zero_frac, 6),
        "adj_matrix_shape": list(data.adj_matrix.shape) if data.adj_matrix.size > 0 else None,
        "adj_nonzero_entries": int((data.adj_matrix > 0).sum()) if data.adj_matrix.size > 0 else None,
    }

    out_path = out_dir / "data_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
