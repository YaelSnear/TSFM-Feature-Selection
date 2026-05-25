"""METR-LA full-load and windowing sanity check.

Usage (from project root):
    python scripts/check_metr_la_windowing.py --config configs/metr_la_windowing_check.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.real_traffic import load_metr_la, make_forecast_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Full load ────────────────────────────────────────────────────────────
    zip_path = Path(cfg.data.zip_path)
    n_rows   = cfg.data.n_rows   # None = all
    print(f"Loading {zip_path}  (n_rows={n_rows}) …")
    data = load_metr_la(zip_path, n_rows=n_rows)
    print(f"  loaded shape : {data.df.shape}  (expected (34272, 207) for full load)")
    assert data.df.shape == (34272, 207), (
        f"Unexpected full-load shape: {data.df.shape}"
    )
    print("  shape assertion OK")

    # ── Windowing ────────────────────────────────────────────────────────────
    wcfg          = cfg.windowing
    target_sensor = getattr(wcfg, "target_sensor", None)   # None → first sensor

    print()
    print("Building forecast windows …")
    windows = make_forecast_windows(
        df=data.df,
        target_sensor=target_sensor,
        context_length=wcfg.context_length,
        horizon=wcfg.horizon,
        stride=wcfg.stride,
        max_windows=wcfg.max_windows,
    )

    # ── Print shapes ─────────────────────────────────────────────────────────
    print()
    print("=== windowing results ===")
    print(f"  target sensor  : {windows.target_sensor}")
    print(f"  X_context shape: {windows.X_context.shape}"
          f"  (n_windows, context_length, n_sensors)")
    print(f"  y_target  shape: {windows.y_target.shape}"
          f"  (n_windows, horizon)")
    print(f"  timestamps     : {windows.timestamps[0]}  →  {windows.timestamps[-1]}")
    print(f"  n_valid_windows: {windows.metadata['n_valid_windows']}")
    print(f"  n_windows_ret  : {windows.metadata['n_windows_returned']}")

    # ── Save shapes ──────────────────────────────────────────────────────────
    summary = {
        "full_load_shape":    list(data.df.shape),
        "target_sensor":      windows.target_sensor,
        "context_length":     wcfg.context_length,
        "horizon":            wcfg.horizon,
        "stride":             wcfg.stride,
        "max_windows":        wcfg.max_windows,
        "X_context_shape":    list(windows.X_context.shape),
        "y_target_shape":     list(windows.y_target.shape),
        "n_valid_windows":    windows.metadata["n_valid_windows"],
        "n_windows_returned": windows.metadata["n_windows_returned"],
        "first_forecast_ts":  str(windows.timestamps[0]),
        "last_forecast_ts":   str(windows.timestamps[-1]),
    }

    out_path = out_dir / "windowing_shapes.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
