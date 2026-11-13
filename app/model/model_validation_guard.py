import numpy as np
import pandas as pd


def safe_numeric_array(values):
    """Convert arbitrary inputs to a 1D finite numeric ndarray or return None."""
    if values is None:
        return None

    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=np.float32, copy=False)
    elif isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype=np.float32, copy=False)
    elif isinstance(values, (list, tuple, set)):
        arr = np.asarray(list(values), dtype=np.float32)
    else:
        try:
            arr = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            return None

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def score_simulation_reasonableness(simulation, historical_close):
    """仅用于过滤明显异常的模拟路径，不能代替真实验证得分。"""
    sim = safe_numeric_array(simulation)
    hist = safe_numeric_array(historical_close)
    if sim is None or hist is None or sim.size < 2 or hist.size < 2:
        return 0.0

    if sim.size > hist.size:
        sim = sim[-hist.size:]

    if sim.size < 2 or sim[0] <= 0 or sim[-1] <= 0:
        return 0.0

    sim_returns = np.diff(sim) / np.maximum(np.abs(sim[:-1]), 1e-8)
    base_returns = np.diff(hist) / np.maximum(np.abs(hist[:-1]), 1e-8)
    if base_returns.size == 0 or sim_returns.size == 0:
        return 0.0

    sim_scale = float(np.nanmean(np.abs(sim_returns))) if sim_returns.size else 0.0
    base_scale = float(np.nanmean(np.abs(base_returns))) if base_returns.size else 0.0
    if not np.isfinite(sim_scale) or not np.isfinite(base_scale):
        return 0.0

    if np.any(np.abs(sim_returns) > 0.5):
        return 0.0

    scale_gap = abs(sim_scale - base_scale) / max(base_scale, 1e-6)
    score = 1.0 - min(1.0, scale_gap)
    return float(np.clip(score, 0.0, 1.0))
