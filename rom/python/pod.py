from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PODModel:
    mean: np.ndarray  # (n_dof,)
    basis: np.ndarray  # (n_dof, r)
    svals: np.ndarray  # (min(n_snap, n_dof),)
    energy: np.ndarray  # cumulative energy fraction

    def project(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_snap, n_dof), got {X.shape}")
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError("X n_dof mismatch vs mean")
        Xc = X - self.mean[None, :]
        return Xc @ self.basis

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        if a.ndim != 2:
            raise ValueError(f"a must be 2D (n_time, r), got {a.shape}")
        if a.shape[1] != self.basis.shape[1]:
            raise ValueError("a r mismatch vs basis")
        return self.mean[None, :] + a @ self.basis.T


def fit_pod(
    X: np.ndarray,
    *,
    r: int | None = None,
    energy_threshold: float = 0.999,
    center: bool = True,
) -> PODModel:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_snap, n_dof), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN/Inf")
    n_snap, n_dof = X.shape
    if n_snap < 2:
        raise ValueError("Need at least 2 snapshots for POD")

    mean = X.mean(axis=0) if center else np.zeros((n_dof,), dtype=float)
    Xc = X - mean[None, :]

    # Economy SVD on snapshot matrix.
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    if s.size == 0:
        raise ValueError("SVD produced no singular values")

    s2 = s**2
    total = float(np.sum(s2))
    if total <= 0.0:
        energy = np.zeros_like(s, dtype=float)
    else:
        energy = np.cumsum(s2) / total
        # Guard against tiny roundoff.
        energy = np.clip(energy, 0.0, 1.0)

    if r is None:
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError("energy_threshold must be in (0, 1]")
        r_sel = int(np.searchsorted(energy, energy_threshold) + 1)
    else:
        r_sel = int(r)
        if r_sel < 1:
            raise ValueError("r must be >= 1")
    r_sel = min(r_sel, Vt.shape[0])

    basis = Vt[:r_sel, :].T  # (n_dof, r)

    if basis.shape != (n_dof, r_sel):
        raise ValueError("Unexpected basis shape")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis contains NaN/Inf")

    return PODModel(mean=mean.astype(float), basis=basis.astype(float), svals=s.astype(float), energy=energy.astype(float))
