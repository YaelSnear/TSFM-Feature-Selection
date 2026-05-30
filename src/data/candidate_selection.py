"""Candidate covariate pool construction for METR-LA.

For each target sensor, builds a pool of candidate sensors to use as potential
covariates in the downstream Chronos-2 evaluation.

Pool modes:
    all_206       : all 206 non-target sensors
    structured_50 : deterministic pool of 50 sensors:
                    20 nearest (highest adj weight)
                    10 far (lowest nonzero or zero adj)
                    10 high-corr (train split Pearson)
                    10 random distractors (seed=42)

all_candidates scope:
    When pool_mode=structured_50, "all_candidates" means all 50 structured candidates.
    When pool_mode=all_206, "all_candidates" means all 206 sensors.

Candidate selection uses TRAIN windows only — no test leakage.

Output:
    outputs/candidate_sensors_by_target.json  (one entry per target, appended)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.real_traffic import MetrLaData


def build_candidate_pool(
    data: MetrLaData,
    target_sensor_id: str,
    train_context_windows: dict[int, np.ndarray],
    target_train_series: np.ndarray,
    max_candidates: int | None = None,
    seed: int = 42,
) -> dict:
    """Build the candidate covariate pool for a given target sensor.

    Args:
        data                   : full MetrLaData (adj_matrix, sensor_id_to_idx, df)
        target_sensor_id       : sensor ID string of the target
        train_context_windows  : {df_col: [N_train, context_length]} — TRAIN split only
        target_train_series    : [N_train, context_length] — target's train context windows
        max_candidates         : None = all_206; 50 = structured_50
        seed                   : RNG seed for random distractors

    Returns:
        dict with candidate_df_cols, pool_mode, candidate_pool_size, details, etc.
    """
    target_adj_idx = data.sensor_id_to_idx[target_sensor_id]
    target_df_col  = data.df.columns.get_loc(target_sensor_id)

    # All non-target df_cols present in train_context_windows
    all_other_cols: list[int] = sorted(
        col for col in train_context_windows.keys() if col != target_df_col
    )

    # Adjacency weight from target to each candidate (row = target adj_idx)
    def _adj_weight(df_col: int) -> float:
        sid = data.df.columns[df_col]
        adj_i = data.sensor_id_to_idx.get(sid)
        if adj_i is None:
            return 0.0
        return float(data.adj_matrix[target_adj_idx, adj_i])

    # ------------------------------------------------------------------
    # Pool mode: all_206
    # ------------------------------------------------------------------
    if max_candidates is None:
        pool_mode = "all_206"
        candidate_df_cols = all_other_cols
        details: dict[str, dict] = {}
        for col in candidate_df_cols:
            sid = data.df.columns[col]
            details[str(col)] = {
                "sensor_id": sid,
                "adj_weight": _adj_weight(col),
                "train_corr": float("nan"),
                "pool_reason": "all_206",
            }
        _print_pool_summary(pool_mode, len(candidate_df_cols), {})

    # ------------------------------------------------------------------
    # Pool mode: structured_N for N <= 20 (debug / small-N only)
    # Selects the top max_candidates sensors by adjacency weight.
    # No category split (near/far/high_corr/random) — all labelled "near".
    # NOT intended for scientific experiments; use structured_50 or all_206 for those.
    # ------------------------------------------------------------------
    elif max_candidates <= 20:
        pool_mode = f"structured_{max_candidates}"
        adj_weights = {col: _adj_weight(col) for col in all_other_cols}
        candidate_df_cols = sorted(
            all_other_cols, key=lambda c: adj_weights[c], reverse=True
        )[:max_candidates]
        details = {
            str(col): {
                "sensor_id":   data.df.columns[col],
                "adj_weight":  float(adj_weights[col]),
                "train_corr":  float("nan"),
                "pool_reason": "near",
            }
            for col in candidate_df_cols
        }
        _print_pool_summary(pool_mode, len(candidate_df_cols), {"near": len(candidate_df_cols)})

    # ------------------------------------------------------------------
    # Pool mode: structured_50
    # ------------------------------------------------------------------
    else:
        pool_mode = "structured_50"
        target_quota = max_candidates  # 50

        adj_weights = {col: _adj_weight(col) for col in all_other_cols}

        chosen: list[int]             = []
        pool_reason: dict[int, str]   = {}
        n_counts: dict[str, int]      = {"near": 0, "far": 0, "high_corr": 0, "random": 0}

        # ---- 20 nearest (highest adj weight) ----
        quota_near = 20
        sorted_by_adj_desc = sorted(all_other_cols, key=lambda c: adj_weights[c], reverse=True)
        for col in sorted_by_adj_desc:
            if len([c for c in chosen if pool_reason.get(c) == "near"]) >= quota_near:
                break
            if col not in chosen:
                chosen.append(col)
                pool_reason[col] = "near"
                n_counts["near"] += 1

        # ---- 10 far (lowest adj weight, including zeros) ----
        quota_far = 10
        sorted_by_adj_asc = sorted(all_other_cols, key=lambda c: adj_weights[c])
        for col in sorted_by_adj_asc:
            if n_counts["far"] >= quota_far:
                break
            if col not in chosen:
                chosen.append(col)
                pool_reason[col] = "far"
                n_counts["far"] += 1

        # ---- 10 high-correlation (train split Pearson) ----
        quota_corr = 10
        remaining_for_corr = [col for col in all_other_cols if col not in chosen]
        train_corr_scores: dict[int, float] = _compute_train_corr(
            remaining_for_corr, target_train_series, train_context_windows
        )
        sorted_by_corr = sorted(remaining_for_corr,
                                key=lambda c: train_corr_scores.get(c, 0.0), reverse=True)
        for col in sorted_by_corr:
            if n_counts["high_corr"] >= quota_corr:
                break
            if col not in chosen:
                chosen.append(col)
                pool_reason[col] = "high_corr"
                n_counts["high_corr"] += 1

        # ---- 10 random distractors ----
        quota_rand = 10
        remaining_for_rand = [col for col in all_other_cols if col not in chosen]
        if len(remaining_for_rand) < quota_rand:
            # Fill from any remaining
            rand_chosen = remaining_for_rand
        else:
            rng = np.random.default_rng(seed)
            rand_chosen = rng.choice(remaining_for_rand, size=quota_rand, replace=False).tolist()
        for col in rand_chosen:
            if n_counts["random"] >= quota_rand:
                break
            if col not in chosen:
                chosen.append(col)
                pool_reason[col] = "random"
                n_counts["random"] += 1

        # ---- Fill quota if any category fell short ----
        total_chosen = len(chosen)
        if total_chosen < target_quota:
            still_remaining = [col for col in all_other_cols if col not in chosen]
            rng2 = np.random.default_rng(seed + 1)
            fill_size = min(target_quota - total_chosen, len(still_remaining))
            if fill_size > 0:
                fill_cols = rng2.choice(still_remaining, size=fill_size, replace=False).tolist()
                for col in fill_cols:
                    chosen.append(col)
                    pool_reason[col] = "fill"
                print(f"[candidate_pool] Quota shortfall — filled {fill_size} extra from remaining pool", flush=True)

        candidate_df_cols = chosen
        _print_pool_summary(pool_mode, len(candidate_df_cols), n_counts)

        # Build details
        all_train_corr = _compute_train_corr(
            candidate_df_cols, target_train_series, train_context_windows
        )
        details = {}
        for col in candidate_df_cols:
            sid = data.df.columns[col]
            details[str(col)] = {
                "sensor_id":   sid,
                "adj_weight":  adj_weights.get(col, 0.0),
                "train_corr":  float(all_train_corr.get(col, float("nan"))),
                "pool_reason": pool_reason.get(col, "unknown"),
            }

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    candidate_sensor_ids = [data.df.columns[col] for col in candidate_df_cols]
    result = {
        "target_sensor_id":    target_sensor_id,
        "candidate_df_cols":   candidate_df_cols,
        "candidate_sensor_ids": candidate_sensor_ids,
        "n_candidates":        len(candidate_df_cols),
        "pool_mode":           pool_mode,
        "candidate_pool_size": len(candidate_df_cols),
        "all_candidates_scope": pool_mode,
        "details":             details,
    }

    # ------------------------------------------------------------------
    # Save (append/update per target)
    # ------------------------------------------------------------------
    out_path = Path("outputs") / "candidate_sensors_by_target.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing[target_sensor_id] = result
    # Convert ndarray ints to plain ints for JSON serialisation
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=_json_default)
    print(f"[candidate_pool] Saved entry for {target_sensor_id} → {out_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_train_corr(
    cols: list[int],
    target_train: np.ndarray,
    train_wins: dict[int, np.ndarray],
) -> dict[int, float]:
    """Mean absolute Pearson correlation with target on train context windows."""
    # target_train: [N, ctx] — average each window to a scalar for correlation
    y = target_train.mean(axis=1)   # [N]
    scores: dict[int, float] = {}
    for col in cols:
        if col not in train_wins:
            scores[col] = 0.0
            continue
        x = train_wins[col].mean(axis=1)   # [N]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            scores[col] = 0.0
        else:
            scores[col] = float(abs(np.corrcoef(x, y)[0, 1]))
    return scores


def _print_pool_summary(pool_mode: str, n_total: int, counts: dict[str, int]) -> None:
    if pool_mode == "all_206":
        print(f"[candidate_pool] {pool_mode}: {n_total} sensors (all non-target)", flush=True)
    else:
        parts = "  ".join(f"{k}={v}" for k, v in counts.items())
        print(f"[candidate_pool] {pool_mode}: {n_total} sensors  ({parts})", flush=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
