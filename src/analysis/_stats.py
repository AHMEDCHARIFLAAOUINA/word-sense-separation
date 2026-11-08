"""Partial correlation with depth control and the model-level tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def residualise(values, depth, degree: int = 1):
    """Residuals of `values` after a polynomial fit of `degree` on `depth`."""
    v = np.asarray(values, dtype=float)
    X = np.vander(np.asarray(depth, dtype=float), degree + 1)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def partial_r(x, y, depth, degree: int = 1) -> float:
    """Correlation of x and y after removing a polynomial depth trend from both."""
    return pearson(residualise(x, depth, degree), residualise(y, depth, degree))


def lag1(v) -> float:
    v = np.asarray(v, dtype=float)
    return pearson(v[:-1], v[1:])


def t_p(r: float, df: float) -> float:
    """Two-sided p for a Pearson r with `df` degrees of freedom."""
    df = max(float(df), 1.0)
    t = r * np.sqrt(df / max(1.0 - r * r, 1e-12))
    return float(2 * stats.t.sf(abs(t), df))


def naive_p(r: float, n: int, degree: int) -> float:
    """The original per-cell p: n layers treated as independent."""
    return t_p(r, n - 2 - degree)


def effective_n(rx, ry, degree: int = 1) -> float:
    n = len(rx)
    k = lag1(rx) * lag1(ry)
    n_eff = n * (1 - k) / (1 + k)
    return float(min(max(n_eff, degree + 3), n))


def effective_n_p(rx, ry, degree: int = 1) -> tuple[float, float, float]:
    """(r, n_eff, p) for residual series rx, ry."""
    r = pearson(rx, ry)
    n_eff = effective_n(rx, ry, degree)
    return r, n_eff, t_p(r, n_eff - 2 - degree)


def circular_shift_p(rx, rys) -> tuple[float, float, int]:
    """Permutation test of the word-averaged r for one model."""
    rx = np.asarray(rx, float)
    rys = [np.asarray(y, float) for y in rys]
    n = len(rx)
    obs = float(np.mean([pearson(rx, y) for y in rys]))
    null = [np.mean([pearson(rx, np.roll(y, k)) for y in rys]) for k in range(1, n)]
    p = (1 + sum(abs(v) >= abs(obs) for v in null)) / n
    return obs, float(p), n


def model_level_summary(values) -> dict:
    """Exploratory model-level inference on one value per model."""
    if isinstance(values, dict):
        names, v = list(values.keys()), np.asarray(list(values.values()), float)
    else:
        v = np.asarray(values, float)
        names = [str(i) for i in range(len(v))]
    n = len(v)
    out = {"n_models": n, "mean": float(v.mean()), "median": float(np.median(v)),
           "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum()),
           "min": float(v.min()), "max": float(v.max())}
    if n >= 3:
        loo = np.array([np.delete(v, i).mean() for i in range(n)])
        out["loo_min"], out["loo_max"] = float(loo.min()), float(loo.max())
        out["loo_min_dropped"] = names[int(loo.argmin())]
        out["loo_max_dropped"] = names[int(loo.argmax())]
        out["t_p"] = float(stats.ttest_1samp(v, 0.0).pvalue)
        out["wilcoxon_p"] = float(stats.wilcoxon(v).pvalue) if np.any(v != 0) else float("nan")
    else:
        out.update(loo_min=np.nan, loo_max=np.nan, loo_min_dropped="", loo_max_dropped="",
                   t_p=np.nan, wilcoxon_p=np.nan)
    return out
