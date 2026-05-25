# tsfm_feature_selection_framework

Minimal, modular experiment framework for TSFM feature-selection and target-relevance experiments.

## Run the smoke test

```bash
cd tsfm_feature_selection_framework
python run_experiment.py --config configs/synthetic_smoke.yaml
```

Outputs land in `outputs/synthetic_smoke/`:
- `metrics.csv` — per-instance, per-scorer ranking metrics
- `feature_scores.csv` — per-instance, per-scorer, per-feature scores and ranks
- `used_config.yaml` — copy of the config that produced the run

## Project layout

```
├── run_experiment.py          entry point
├── configs/                   one YAML per experiment
├── src/
│   ├── config.py              YAML → SimpleNamespace loader
│   ├── runner.py              ExperimentRunner
│   ├── data/                  dataset generators + InstanceData dataclass
│   ├── scoring/               feature scorers (random, pearson, lagged_pearson)
│   ├── evaluation/            ranking metrics (P@K, R@K, NDCG, AP, MRR)
│   ├── reporting/             CSV / YAML savers
│   └── extraction/            (stub) Chronos embedding extractor
└── outputs/                   results written here (gitignored except .gitkeep)
```

## Experiment types

| Type | Labels | Metrics |
|---|---|---|
| `synthetic_labeled` | ground-truth relevant features | P@K, R@K, NDCG@K, AP, MRR |
| `proxy_labeled` | adjacency/correlation proxy | Proxy Recovery@K, MRR |
| `forecasting` | none | MASE, sMAPE, RMSE (future) |

## Adding a new scorer

1. Implement `my_scorer(instance: InstanceData) -> np.ndarray` in `src/scoring/`.
2. Register it in `REGISTRY` or add a branch in `runner._build_scorer`.
3. Add its name to `scoring.methods` in your config.

## Adding a new dataset

1. Implement `generate_dataset(cfg) -> list[InstanceData]` in `src/data/`.
2. Add a branch in `runner._load_dataset`.
3. Set `dataset.type` in your config.
