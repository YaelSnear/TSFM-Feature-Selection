"""Standalone Chronos extraction smoke test.

Usage (from project root):
    python scripts/test_chronos_extraction.py --config configs/chronos_extraction_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.extraction.chronos_hooks import ChronosEmbeddingExtractor


def make_test_series(T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    return np.sin(t) + 0.2 * rng.standard_normal(T)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = make_test_series(T=cfg.series.T, seed=cfg.series.seed)
    print(f"Series: T={len(series)}, mean={series.mean():.4f}, std={series.std():.4f}")

    wl = cfg.windows.window_length
    stride = cfg.windows.stride

    extractor = ChronosEmbeddingExtractor.from_config(cfg.extraction)

    try:
        extractor.load()

        embeddings = extractor.extract(series, window_length=wl, stride=stride)

        print(f"\nExtracted embeddings:")
        shapes = {}
        for layer_idx, emb in sorted(embeddings.items()):
            print(f"  layer {layer_idx}: {emb.shape}  (n_windows={emb.shape[0]}, embed_dim={emb.shape[1]})")
            shapes[str(layer_idx)] = list(emb.shape)

    finally:
        extractor.remove_hooks()
        print("\nHooks removed.")

    out_path = out_dir / "embedding_shapes.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_id": cfg.extraction.model_id,
                "device": extractor.device,
                "n_layers": extractor.n_layers,
                "window_length": wl,
                "stride": stride,
                "pooling": cfg.extraction.pooling,
                "series_length": len(series),
                "layers": shapes,
            },
            f,
            indent=2,
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
