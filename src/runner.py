"""ExperimentRunner: orchestrates dataset → scoring → evaluation → reporting."""

from __future__ import annotations
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from src.data import InstanceData
from src.scoring.classical import REGISTRY as SCORER_REGISTRY, lagged_pearson_scorer
from src.evaluation.labeled_ranking import evaluate
from src.reporting.saver import save_all


def _build_scorer(name: str, scoring_cfg: SimpleNamespace, rng: np.random.Generator):
    if name == "random":
        def fn(inst: InstanceData):
            return SCORER_REGISTRY["random"](inst, rng)
        return fn
    if name == "pearson":
        return SCORER_REGISTRY["pearson"]
    if name == "lagged_pearson":
        max_lag = getattr(getattr(scoring_cfg, "lagged_pearson", None), "max_lag", 15)
        def fn(inst: InstanceData):
            return lagged_pearson_scorer(inst, max_lag=max_lag)
        return fn
    raise ValueError(f"Unknown scorer: {name}")


def _load_dataset(cfg: SimpleNamespace) -> list[InstanceData]:
    dtype = cfg.dataset.type
    if dtype == "synthetic_variable_lag":
        from src.data.synthetic_variable_lag import generate_dataset
        return generate_dataset(cfg.dataset)
    raise ValueError(f"Unknown dataset type: {dtype}")


class ExperimentRunner:
    def __init__(self, config: SimpleNamespace, config_path: str | Path):
        self.cfg = config
        self.config_path = Path(config_path)

    def run(self) -> None:
        print(f"[{self.cfg.experiment.name}] loading dataset …")
        instances = _load_dataset(self.cfg)
        print(f"  {len(instances)} instances, {instances[0].n_features} features, "
              f"T={instances[0].series_length}")

        scorer_names: list[str] = self.cfg.scoring.methods
        rng = np.random.default_rng(0)
        scorers = {name: _build_scorer(name, self.cfg.scoring, rng) for name in scorer_names}

        k_values: list[int] = self.cfg.evaluation.k_values
        metrics_rows: list[dict] = []
        score_rows: list[dict] = []

        for inst in instances:
            iid = inst.metadata.get("instance_id", "?")
            for sname, scorer_fn in scorers.items():
                scores = scorer_fn(inst)

                # ranking by index (1 = best)
                order = np.argsort(-scores)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)

                for j, fname in enumerate(inst.feature_names):
                    score_rows.append({
                        "instance_id": iid,
                        "scorer": sname,
                        "feature": fname,
                        "score": round(float(scores[j]), 6),
                        "rank": int(ranks[j]),
                        "is_relevant": int(j in (inst.relevant_feature_indices or [])),
                    })

                if inst.relevant_feature_indices is not None:
                    row = evaluate(scores, inst.relevant_feature_indices, k_values)
                    metrics_rows.append({"instance_id": iid, "scorer": sname, **row})

        print(f"  scoring done — {len(metrics_rows)} metric rows")
        save_all(metrics_rows, score_rows, self.config_path, self.cfg)
        print("done.")
