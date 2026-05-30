"""Role-aware target sensor selection for METR-LA.

Selects 10 target sensors (2 per role) from the 207-sensor graph using graph-theoretic
metrics derived from the adjacency matrix and temporal variance from the train split only.

Roles:
    A — central      : highest weighted degree
    B — peripheral   : lowest weighted degree
    C — bridge       : highest betweenness centrality
    D — dense        : highest clustering coefficient
    E — high_variance: highest variance on train split

The adjacency matrix encodes road-network connectivity/proximity, NOT true Euclidean
distance. Role labels reflect connectivity structure only.

No graph sparsification or thresholds are applied. If any are needed, the user must
be consulted first.

Outputs:
    outputs/selected_target_sensors_by_role.json
    outputs/target_role_rankings_full.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.real_traffic import MetrLaData


# ---------------------------------------------------------------------------
# Adjacency graph properties report
# ---------------------------------------------------------------------------

def _report_adj_properties(adj_matrix: np.ndarray) -> None:
    N = adj_matrix.shape[0]
    is_symmetric = bool(np.allclose(adj_matrix, adj_matrix.T))
    nonzero_mask = adj_matrix != 0
    n_nonzero = int(nonzero_mask.sum())
    density = n_nonzero / (N * (N - 1)) if N > 1 else 0.0
    nonzero_weights = adj_matrix[nonzero_mask]
    print("[target_selection] Adjacency graph properties:", flush=True)
    print(f"  symmetric        : {is_symmetric}", flush=True)
    print(f"  nonzero edges    : {n_nonzero}", flush=True)
    print(f"  graph density    : {density:.6f}", flush=True)
    if len(nonzero_weights) > 0:
        print(
            f"  nonzero weights  : min={nonzero_weights.min():.6f}  "
            f"max={nonzero_weights.max():.6f}  mean={nonzero_weights.mean():.6f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def select_target_sensors(
    data: MetrLaData,
    train_df: pd.DataFrame,
    n_per_role: int = 2,
    seed: int = 42,
) -> tuple[dict[str, list[dict]], pd.DataFrame, dict[str, str]]:
    """Select n_per_role target sensors for each of 5 structural/temporal roles.

    Args:
        data        : full MetrLaData (adj_matrix, sensor_ids, sensor_id_to_idx, df)
        train_df    : train-split DataFrame (no test leakage)
        n_per_role  : sensors selected per role (default 2, total 10)
        seed        : unused for roles A-E (all deterministic); kept for API consistency

    Returns:
        (role_dict, rankings_df, sensor_role_map)
        role_dict       : {role_name: [list of sensor dicts]}
        rankings_df     : full 207-row rankings DataFrame
        sensor_role_map : flat {sensor_id: role_name}
    """
    adj = data.adj_matrix
    N   = adj.shape[0]

    _report_adj_properties(adj)

    # ------------------------------------------------------------------
    # Sanity-check index mapping: adj_idx ≠ df_col in general
    # ------------------------------------------------------------------
    assert all(sid in data.sensor_id_to_idx for sid in data.sensor_ids), \
        "Some sensor_ids are missing from sensor_id_to_idx"
    assert all(sid in data.df.columns for sid in data.sensor_ids), \
        "Some sensor_ids are missing from df.columns"

    # adj_idx → df_col (same pattern as select_sensors() in real_traffic.py)
    adj_to_df_col: dict[int, int] = {
        adj_i: data.df.columns.get_loc(sid)
        for sid, adj_i in data.sensor_id_to_idx.items()
        if sid in data.df.columns
    }
    df_col_to_adj: dict[int, int] = {v: k for k, v in adj_to_df_col.items()}

    valid_adj_indices = sorted(adj_to_df_col.keys())

    # ------------------------------------------------------------------
    # Graph metrics (networkx)
    # ------------------------------------------------------------------
    try:
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        rows, cols = np.nonzero(adj)
        for r, c in zip(rows, cols):
            G.add_edge(int(r), int(c), weight=float(adj[r, c]))

        betweenness_raw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        clustering_raw  = nx.clustering(G, weight="weight")
        betweenness: dict[int, float] = betweenness_raw
        clustering: dict[int, float]  = clustering_raw
        print("[target_selection] Graph metrics computed via networkx.", flush=True)

    except ImportError:
        print("[target_selection] networkx not available; using adjacency-only fallback.", flush=True)
        # Betweenness approximation: normalised degree centrality on inverse-weight graph
        inv_deg = np.zeros(N)
        for i in range(N):
            row = adj[i].copy()
            row[row == 0] = np.inf
            inv_deg[i] = (1.0 / row[row != np.inf]).sum() if np.any(row != np.inf) else 0.0
        inv_deg_norm = inv_deg / (inv_deg.max() + 1e-12)
        betweenness = {i: float(inv_deg_norm[i]) for i in range(N)}
        clustering  = {i: 0.0 for i in range(N)}

    # Weighted degree (row sum)
    weighted_degree: dict[int, float] = {i: float(adj[i].sum()) for i in range(N)}

    # ------------------------------------------------------------------
    # Temporal metrics (train split only)
    # ------------------------------------------------------------------
    variance_train: dict[int, float]   = {}
    missingness: dict[int, float]      = {}
    lag1_autocorr: dict[int, float]    = {}
    mean_abs_train_corr: dict[int, float] = {}

    for sid in data.sensor_ids:
        if sid not in train_df.columns:
            continue
        adj_i  = data.sensor_id_to_idx[sid]
        series = train_df[sid].values.astype(np.float64)

        variance_train[adj_i] = float(np.nanvar(series))
        missingness[adj_i]    = float((np.isnan(series) | (series == 0)).mean())

        s = pd.Series(series)
        lag1_autocorr[adj_i] = float(s.autocorr(lag=1)) if len(s) > 1 else 0.0

    # mean_abs_train_corr: sample up to 50 other sensors for speed
    all_sid_list = [s for s in data.sensor_ids if s in train_df.columns]
    sample_size  = min(50, len(all_sid_list) - 1)
    rng_sample   = np.random.default_rng(seed)
    for sid in all_sid_list:
        adj_i  = data.sensor_id_to_idx[sid]
        series = train_df[sid].values.astype(np.float64)
        others = [s for s in all_sid_list if s != sid]
        sample = rng_sample.choice(others, size=sample_size, replace=False).tolist()
        corrs  = [
            abs(np.corrcoef(series, train_df[o].values.astype(np.float64))[0, 1])
            for o in sample
            if not np.any(np.isnan(train_df[o].values))
        ]
        mean_abs_train_corr[adj_i] = float(np.mean(corrs)) if corrs else 0.0

    # ------------------------------------------------------------------
    # Build full rankings DataFrame
    # ------------------------------------------------------------------
    records = []
    for adj_i in valid_adj_indices:
        sid    = [s for s, a in data.sensor_id_to_idx.items() if a == adj_i][0]
        df_col = adj_to_df_col[adj_i]
        records.append({
            "sensor_id":           sid,
            "adj_idx":             adj_i,
            "df_col":              df_col,
            "weighted_degree":     weighted_degree.get(adj_i, 0.0),
            "betweenness":         betweenness.get(adj_i, 0.0),
            "clustering":          clustering.get(adj_i, 0.0),
            "variance_train":      variance_train.get(adj_i, 0.0),
            "missingness":         missingness.get(adj_i, 0.0),
            "lag1_autocorr":       lag1_autocorr.get(adj_i, 0.0),
            "mean_abs_train_corr": mean_abs_train_corr.get(adj_i, 0.0),
        })
    rankings_df = pd.DataFrame(records).sort_values("adj_idx").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Role assignment (deterministic, with deduplication)
    # ------------------------------------------------------------------
    roles_spec = [
        ("role_A_central",       "weighted_degree",  True),   # True = descending
        ("role_B_peripheral",    "weighted_degree",  False),  # False = ascending
        ("role_C_bridge",        "betweenness",      True),
        ("role_D_dense",         "clustering",       True),
        ("role_E_high_variance", "variance_train",   True),
    ]

    selected_so_far: set[int] = set()
    role_dict: dict[str, list[dict]] = {}

    for role_name, metric_col, descending in roles_spec:
        sorted_df = rankings_df.sort_values(metric_col, ascending=not descending)
        chosen: list[dict] = []

        for _, row in sorted_df.iterrows():
            if row["adj_idx"] in selected_so_far:
                continue
            if len(chosen) >= n_per_role:
                break
            metric_val = float(row[metric_col])
            chosen.append({
                "sensor_id":    str(row["sensor_id"]),
                "df_col":       int(row["df_col"]),
                "adj_idx":      int(row["adj_idx"]),
                "role":         role_name,
                "metric_key":   metric_col,
                "metric_value": metric_val,
                "explanation":  (
                    f"Selected for {role_name}: "
                    f"{metric_col}={metric_val:.6f} "
                    f"({'highest' if descending else 'lowest'} among unselected sensors)"
                ),
            })
            selected_so_far.add(int(row["adj_idx"]))

        role_dict[role_name] = chosen
        print(
            f"[target_selection] {role_name}: {[c['sensor_id'] for c in chosen]}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Flat sensor_role_map
    # ------------------------------------------------------------------
    sensor_role_map: dict[str, str] = {
        s["sensor_id"]: role_name
        for role_name, sensors in role_dict.items()
        for s in sensors
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "selected_target_sensors_by_role.json"
    with open(json_path, "w") as f:
        json.dump(role_dict, f, indent=2)
    print(f"[target_selection] Saved {json_path}", flush=True)

    csv_path = out_dir / "target_role_rankings_full.csv"
    if sample_size < len(all_sid_list) - 1:
        rankings_df.attrs["mean_abs_train_corr_note"] = (
            f"mean_abs_train_corr computed from {sample_size} randomly sampled peers (approximate)"
        )
    rankings_df.to_csv(csv_path, index=False)
    print(f"[target_selection] Saved {csv_path}", flush=True)

    total_selected = sum(len(v) for v in role_dict.values())
    print(f"[target_selection] Total target sensors selected: {total_selected}", flush=True)

    return role_dict, rankings_df, sensor_role_map
