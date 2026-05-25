"""Chronos layer-wise embedding extractor.

Pipeline:
    series → make_windows() → pipe.embed() → forward hooks on transformer layers
          → pool over token dimension → {layer_idx: np.ndarray (n_windows, embed_dim)}

Not connected to ExperimentRunner yet. Use standalone or via from_config().

Expected config section (under a key you choose, e.g. cfg.chronos):
    model_id : amazon/chronos-2
    device   : cuda          # optional — auto-detected if omitted
    layers   : [2, 7, 9]     # optional — defaults to all layers
    pooling  : mean          # mean | last | max
"""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from src.extraction.windows import make_windows


class ChronosEmbeddingExtractor:
    """Load Chronos, register hooks on transformer layers, extract per-layer embeddings."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        layers: list[int] | None = None,
        pooling: str = "mean",
    ) -> None:
        self.model_id = model_id
        self.pooling = pooling
        self._requested_layers = layers   # None = all layers after load
        self._device: str | None = device

        self._pipe = None
        self._n_layers: int | None = None
        self._layer_states: dict[int, object] = {}
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: SimpleNamespace) -> ChronosEmbeddingExtractor:
        """Instantiate from a SimpleNamespace config section."""
        return cls(
            model_id=cfg.model_id,
            device=getattr(cfg, "device", None),
            layers=getattr(cfg, "layers", None),
            pooling=getattr(cfg, "pooling", "mean"),
        )

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load Chronos from HuggingFace and register forward hooks."""
        import torch
        from chronos import BaseChronosPipeline

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self._pipe = BaseChronosPipeline.from_pretrained(
            self.model_id, device_map=device
        )
        self._pipe.model.eval()

        blocks = self._find_blocks()
        self._n_layers = len(blocks)

        target = (
            self._requested_layers
            if self._requested_layers is not None
            else list(range(self._n_layers))
        )
        for idx in target:
            if idx >= self._n_layers:
                raise ValueError(
                    f"Requested layer {idx} but model only has {self._n_layers} layers."
                )

        self._register_hooks(blocks, target)
        print(
            f"[ChronosEmbeddingExtractor] loaded {self.model_id} on {device}, "
            f"{self._n_layers} layers, hooks on {target}"
        )

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        series: np.ndarray,
        window_length: int,
        stride: int = 1,
        layers: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Embed windowed series and return pooled hidden states per layer.

        Args:
            series        : 1-D array of length T
            window_length : number of time steps per window
            stride        : stride between consecutive windows
            layers        : layer indices to return; defaults to self._requested_layers
                            (or all if that is also None)

        Returns:
            {layer_idx: np.ndarray of shape (n_windows, embed_dim)}
            When pooling="none": shape is (n_windows, n_patches, embed_dim).
        """
        if self._pipe is None:
            raise RuntimeError("Call .load() before .extract().")

        import torch

        windows = make_windows(series, window_length, stride)   # (n_windows, WL)
        self._layer_states.clear()

        tensors = [torch.tensor(w, dtype=torch.float32) for w in windows]
        with torch.no_grad():
            self._pipe.embed(tensors)

        target = layers if layers is not None else self._requested_layers
        if target is None:
            target = list(range(self._n_layers))

        result: dict[int, np.ndarray] = {}
        for idx in target:
            if idx not in self._layer_states:
                continue
            h = self._layer_states[idx]   # tensor (n_windows, seq_len, embed_dim)
            result[idx] = self._pool(h)

        return result

    def extract_windows(
        self,
        windows: np.ndarray,
        layers: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Embed pre-computed windows and return hidden states per layer.

        Args:
            windows : np.ndarray of shape (n_windows, window_length) — already sliced
            layers  : layer indices to return; defaults to self._requested_layers

        Returns:
            {layer_idx: np.ndarray}
            When pooling="none": shape (n_windows, n_patches, embed_dim).
            Otherwise: shape (n_windows, embed_dim).
        """
        if self._pipe is None:
            raise RuntimeError("Call .load() before .extract_windows().")

        import torch

        self._layer_states.clear()
        tensors = [torch.tensor(w, dtype=torch.float32) for w in windows]
        with torch.no_grad():
            self._pipe.embed(tensors)

        target = layers if layers is not None else self._requested_layers
        if target is None:
            target = list(range(self._n_layers))

        result: dict[int, np.ndarray] = {}
        for idx in target:
            if idx not in self._layer_states:
                continue
            h = self._layer_states[idx]   # tensor (n_windows, seq_len, embed_dim)
            result[idx] = self._pool(h)

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int | None:
        return self._n_layers

    @property
    def device(self) -> str | None:
        return self._device

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ChronosEmbeddingExtractor:
        return self

    def __exit__(self, *_) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_blocks(self):
        """Return the nn.ModuleList of transformer blocks.

        Three-tier search (mirrors notebook cell-2):
          1. model.layers directly on the HF model wrapper
          2. model.model.layers one level deeper
          3. First nn.ModuleList found via named_modules()
        """
        import torch.nn as nn

        model = self._pipe.model
        hf = model.model if hasattr(model, "model") else model

        if hasattr(hf, "layers"):
            return hf.layers
        if hasattr(hf, "model") and hasattr(hf.model, "layers"):
            return hf.model.layers

        lists = [m for _, m in hf.named_modules() if isinstance(m, nn.ModuleList)]
        if not lists:
            raise RuntimeError(
                "Cannot find transformer layer blocks in model. "
                "Inspect the model architecture and extend _find_blocks()."
            )
        return lists[0]

    def _register_hooks(self, blocks, target_layers: list[int]) -> None:
        self.remove_hooks()

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                if hasattr(output, "hidden_states"):
                    self._layer_states[layer_idx] = output.hidden_states.detach()
                elif isinstance(output, tuple):
                    self._layer_states[layer_idx] = output[0].detach()
            return hook

        for idx in target_layers:
            handle = blocks[idx].register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    def _pool(self, h) -> np.ndarray:
        """Pool hidden states (n_windows, seq_len, embed_dim) → output array.

        pooling="none" returns (n_windows, seq_len, embed_dim) — full patch tensor.
        All other strategies collapse seq_len → (n_windows, embed_dim).
        """
        if self.pooling == "none":
            return h.cpu().numpy()
        if self.pooling == "mean":
            return h.mean(dim=1).cpu().numpy()
        if self.pooling == "last":
            return h[:, -1, :].cpu().numpy()
        if self.pooling == "max":
            return h.max(dim=1).values.cpu().numpy()
        raise ValueError(f"Unknown pooling strategy: {self.pooling!r}")
