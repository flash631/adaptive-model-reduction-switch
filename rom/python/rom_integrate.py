from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class IntegrateResult:
    t: np.ndarray  # (n_time,)
    a: np.ndarray  # (n_time, r)
    success: bool
    message: str


def integrate_ivp(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    a0: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    method: str = "RK45",
    max_step: float | None = None,
) -> IntegrateResult:
    try:
        from scipy.integrate import solve_ivp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for integrate_ivp") from exc

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 2:
        raise ValueError("t_eval must have at least 2 points")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be strictly increasing")

    a0 = np.asarray(a0, dtype=float).reshape(-1)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=a0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=method,
        max_step=max_step,
    )

    # SciPy returns sol.y as an ndarray, but treat it defensively as it is stored in an OdeResult
    # (a dict-like container) and some environments may present it as a list.
    y = np.asarray(sol.y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    a = y.T
    if not np.all(np.isfinite(a)):
        return IntegrateResult(t=t_eval, a=a, success=False, message="NaN/Inf encountered in ROM integration")

    return IntegrateResult(t=np.asarray(sol.t, dtype=float), a=a, success=bool(sol.success), message=str(sol.message))


def integrate_fixed_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    *,
    clip: tuple[np.ndarray, np.ndarray] | None = None,
) -> IntegrateResult:
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 2:
        raise ValueError("t_eval must have at least 2 points")
    dt_all = np.diff(t_eval)
    if np.any(dt_all <= 0):
        raise ValueError("t_eval must be strictly increasing")
    if not np.allclose(dt_all, dt_all[0], rtol=1e-6, atol=1e-12):
        raise ValueError("integrate_fixed_step requires uniform time steps")
    dt = float(dt_all[0])

    y0 = np.asarray(y0, dtype=float).reshape(-1)
    r = y0.shape[0]
    y = np.empty((t_eval.size, r), dtype=float)
    y[0] = y0

    lo = hi = None
    if clip is not None:
        lo, hi = (np.asarray(clip[0], dtype=float).reshape(-1), np.asarray(clip[1], dtype=float).reshape(-1))
        if lo.shape != (r,) or hi.shape != (r,):
            raise ValueError("clip bounds shape mismatch")

    def _clip(v: np.ndarray) -> np.ndarray:
        if lo is None:
            return v
        return np.minimum(np.maximum(v, lo), hi)

    for k in range(t_eval.size - 1):
        t = float(t_eval[k])
        yk = y[k]
        k1 = rhs(t, yk)
        k2 = rhs(t + 0.5 * dt, yk + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, yk + 0.5 * dt * k2)
        k4 = rhs(t + dt, yk + dt * k3)
        yn = yk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        yn = _clip(yn)
        y[k + 1] = yn
        if not np.all(np.isfinite(yn)):
            return IntegrateResult(t=t_eval[: k + 2], a=y[: k + 2], success=False, message="NaN/Inf in fixed-step integration")

    return IntegrateResult(t=t_eval, a=y, success=True, message="fixed-step RK4")
