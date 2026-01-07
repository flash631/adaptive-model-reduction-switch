from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OpInfModel:
    # Learned dynamics in *normalized* coordinates z = (a - mu) / sigma
    c: np.ndarray  # (r,)
    A: np.ndarray  # (r, r)
    H: np.ndarray  # (r, r*r)
    mu: np.ndarray  # (r,)
    sigma: np.ndarray  # (r,)

    def rhs(self, _t: float, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.mu.shape[0]:
            raise ValueError("State size mismatch")
        z = (a - self.mu) / self.sigma
        zz = np.kron(z, z)
        dz = self.c + self.A @ z + self.H @ zz
        da = dz * self.sigma  # since a = mu + sigma * z
        return da


def _finite_difference(times: np.ndarray, a: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    if a.ndim != 2:
        raise ValueError("a must be (n_time, r)")
    if times.shape[0] != a.shape[0]:
        raise ValueError("times length mismatch")
    if times.shape[0] < 3:
        raise ValueError("Need >= 3 time points for finite-difference derivative")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(times)):
        raise ValueError("NaN/Inf in inputs")

    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing")
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12):
        # Use gradient for nonuniform.
        return np.gradient(a, times, axis=0)
    h = float(dt[0])

    da = np.empty_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2.0 * h)
    da[0] = (a[1] - a[0]) / h
    da[-1] = (a[-1] - a[-2]) / h
    return da


def _build_library(z: np.ndarray) -> np.ndarray:
    # Theta = [1, z, kron(z,z)]
    if z.ndim != 2:
        raise ValueError("z must be (n, r)")
    n, r = z.shape
    ones = np.ones((n, 1), dtype=float)
    quad = np.einsum("bi,bj->bij", z, z).reshape(n, r * r)
    return np.hstack([ones, z, quad])


def fit_quadratic(
    times: np.ndarray,
    a: np.ndarray,
    *,
    ridge: float = 1e-6,
    normalize: bool = True,
) -> OpInfModel:
    times = np.asarray(times, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError("a must be (n_time, r)")
    if times.shape[0] != a.shape[0]:
        raise ValueError("times length mismatch vs a")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(times)):
        raise ValueError("NaN/Inf in a/times")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    da = _finite_difference(times, a)

    if normalize:
        mu = a.mean(axis=0)
        sigma = a.std(axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
    else:
        mu = np.zeros((a.shape[1],), dtype=float)
        sigma = np.ones((a.shape[1],), dtype=float)

    z = (a - mu[None, :]) / sigma[None, :]
    dz = da / sigma[None, :]

    Theta = _build_library(z)  # (n, 1+r+r^2)
    if not np.all(np.isfinite(Theta)):
        raise ValueError("NaN/Inf in library matrix")

    # Ridge regression: min ||Theta W - dz||_F^2 + ridge||W||_F^2
    # Solve normal equations: (Theta^T Theta + ridge I) W = Theta^T dz
    G = Theta.T @ Theta
    rhs = Theta.T @ dz
    if ridge > 0:
        G = G + ridge * np.eye(G.shape[0])
    W = np.linalg.solve(G, rhs)  # (1+r+r^2, r)

    c = W[0, :]
    A = W[1 : 1 + a.shape[1], :].T
    H = W[1 + a.shape[1] :, :].T

    if c.shape != (a.shape[1],):
        raise ValueError("Unexpected c shape")
    if A.shape != (a.shape[1], a.shape[1]):
        raise ValueError("Unexpected A shape")
    if H.shape != (a.shape[1], a.shape[1] * a.shape[1]):
        raise ValueError("Unexpected H shape")

    return OpInfModel(c=c.astype(float), A=A.astype(float), H=H.astype(float), mu=mu.astype(float), sigma=sigma.astype(float))

