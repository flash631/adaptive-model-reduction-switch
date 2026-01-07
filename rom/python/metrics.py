from __future__ import annotations

import numpy as np


def l2_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    num = float(np.linalg.norm(a - b))
    den = max(float(np.linalg.norm(b)), float(eps))
    return float(num / den)


def l2_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    return float(np.linalg.norm(a - b))


def l2_rel_volweighted(a: np.ndarray, b: np.ndarray, volumes: np.ndarray, eps: float = 1e-8) -> float:
    """
    Relative L2 error with cell-volume weighting:
        err = sqrt(sum(V * ||a-b||^2)) / sqrt(sum(V * ||b||^2)).

    Supports scalar fields (nCells,) and vector fields (nCells, nComp).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.asarray(volumes, dtype=float).reshape(-1)
    if v.ndim != 1:
        raise ValueError("volumes must be 1D")
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    if a.ndim == 1:
        if a.shape[0] != v.shape[0]:
            raise ValueError("volumes length mismatch")
        diff2 = (a - b) ** 2
        b2 = b**2
    elif a.ndim == 2:
        if a.shape[0] != v.shape[0]:
            raise ValueError("volumes length mismatch")
        diff2 = np.sum((a - b) ** 2, axis=1)
        b2 = np.sum(b**2, axis=1)
    else:
        raise ValueError("Expected 1D or 2D fields")

    num = float(np.sqrt(np.sum(v * diff2)))
    den = max(float(np.sqrt(np.sum(v * b2))), float(eps))
    return float(num / den)


def l2_rel_time_series(X: np.ndarray, Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("Shape mismatch")
    if X.ndim < 2:
        raise ValueError("Expected at least 2D arrays with time axis first")
    n = X.shape[0]
    out = np.empty((n,), dtype=float)
    for i in range(n):
        out[i] = l2_rel(X[i], Y[i], eps=eps)
    return out


def l2_abs_time_series(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("Shape mismatch")
    if X.ndim < 2:
        raise ValueError("Expected at least 2D arrays with time axis first")
    n = X.shape[0]
    out = np.empty((n,), dtype=float)
    for i in range(n):
        out[i] = l2_abs(X[i], Y[i])
    return out


def l2_rel_time_series_volweighted(X: np.ndarray, Y: np.ndarray, volumes: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    v = np.asarray(volumes, dtype=float).reshape(-1)
    if X.shape != Y.shape:
        raise ValueError("Shape mismatch")
    if X.ndim < 2:
        raise ValueError("Expected at least 2D arrays with time axis first")
    n = X.shape[0]
    out = np.empty((n,), dtype=float)
    for i in range(n):
        out[i] = l2_rel_volweighted(X[i], Y[i], volumes=v, eps=eps)
    return out


def l2_abs_volweighted(a: np.ndarray, b: np.ndarray, volumes: np.ndarray) -> float:
    """
    Absolute L2 error with cell-volume weighting:
        abs = sqrt(sum(V * ||a-b||^2)).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.asarray(volumes, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    if a.ndim == 1:
        if a.shape[0] != v.shape[0]:
            raise ValueError("volumes length mismatch")
        diff2 = (a - b) ** 2
    elif a.ndim == 2:
        if a.shape[0] != v.shape[0]:
            raise ValueError("volumes length mismatch")
        diff2 = np.sum((a - b) ** 2, axis=1)
    else:
        raise ValueError("Expected 1D or 2D fields")
    return float(np.sqrt(np.sum(v * diff2)))


def l2_abs_time_series_volweighted(X: np.ndarray, Y: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    v = np.asarray(volumes, dtype=float).reshape(-1)
    if X.shape != Y.shape:
        raise ValueError("Shape mismatch")
    if X.ndim < 2:
        raise ValueError("Expected at least 2D arrays with time axis first")
    n = X.shape[0]
    out = np.empty((n,), dtype=float)
    for i in range(n):
        out[i] = l2_abs_volweighted(X[i], Y[i], volumes=v)
    return out


def mean_after_discard(x: np.ndarray, discard_frac: float = 0.2) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return float("nan")
    frac = float(discard_frac)
    if not (0.0 <= frac < 1.0):
        raise ValueError("discard_frac must be in [0,1)")
    k0 = int(max(0, round(frac * x.size)))
    return float(np.mean(x[k0:]))


def rms_fluct_after_discard(x: np.ndarray, discard_frac: float = 0.2) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return float("nan")
    frac = float(discard_frac)
    if not (0.0 <= frac < 1.0):
        raise ValueError("discard_frac must be in [0,1)")
    k0 = int(max(0, round(frac * x.size)))
    xs = x[k0:]
    mu = float(np.mean(xs))
    return float(np.sqrt(np.mean((xs - mu) ** 2)))
