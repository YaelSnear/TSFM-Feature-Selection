"""Save experiment outputs: metrics.csv, feature_scores.csv, used_config.yaml."""

from __future__ import annotations
import csv
import shutil
from pathlib import Path
from types import SimpleNamespace


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metrics(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    path = out_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {path}")


def save_feature_scores(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    path = out_dir / "feature_scores.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {path}")


def save_config(config_path: str | Path, out_dir: Path) -> None:
    dest = out_dir / "used_config.yaml"
    shutil.copy(config_path, dest)
    print(f"  saved {dest}")


def save_all(
    metrics_rows: list[dict],
    score_rows: list[dict],
    config_path: str | Path,
    cfg: SimpleNamespace,
) -> None:
    out_dir = Path(cfg.output.dir)
    _ensure_dir(out_dir)
    save_metrics(metrics_rows, out_dir)
    save_feature_scores(score_rows, out_dir)
    save_config(config_path, out_dir)
