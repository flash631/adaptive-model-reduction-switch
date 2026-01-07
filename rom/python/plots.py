from __future__ import annotations

from pathlib import Path

import numpy as np


def save_singular_values_plot(svals: np.ndarray, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    svals = np.asarray(svals, dtype=float).reshape(-1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 3.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(np.arange(1, svals.size + 1), svals, "-o", ms=3, lw=1)
    ax.set_xlabel("mode")
    ax.set_ylabel("singular value")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_energy_plot(cum_energy: np.ndarray, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    e = np.asarray(cum_energy, dtype=float).reshape(-1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 3.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, e.size + 1), e, "-o", ms=3, lw=1)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("modes retained")
    ax.set_ylabel("cumulative energy")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_error_plot(times: np.ndarray, err: np.ndarray, out: Path, title: str, ylabel: str = "L2 relative error") -> None:
    import matplotlib.pyplot as plt  # type: ignore

    t = np.asarray(times, dtype=float).reshape(-1)
    e = np.asarray(err, dtype=float).reshape(-1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 3.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    m = np.isfinite(t) & np.isfinite(e)
    if np.count_nonzero(m) >= 2:
        ax.semilogy(t[m], np.maximum(e[m], 1e-16), "-", lw=1.5)
    elif np.count_nonzero(m) == 1:
        ax.semilogy(t[m], np.maximum(e[m], 1e-16), "o", ms=3)
        ax.text(0.5, 0.5, "Only 1 finite point", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No finite validation points", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_error_plot_two(
    times: np.ndarray,
    err_u: np.ndarray,
    err_p: np.ndarray,
    out: Path,
    title: str,
    ylabel: str = "L2 relative error",
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    t = np.asarray(times, dtype=float).reshape(-1)
    eu = np.asarray(err_u, dtype=float).reshape(-1)
    ep = np.asarray(err_p, dtype=float).reshape(-1)
    if t.size != eu.size or t.size != ep.size:
        raise ValueError("times/error length mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 3.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    mu = np.isfinite(t) & np.isfinite(eu)
    mp = np.isfinite(t) & np.isfinite(ep)
    if np.count_nonzero(mu):
        ax.semilogy(t[mu], np.maximum(eu[mu], 1e-16), "-o", lw=1.2, ms=3, mew=0, label="U")
    if np.count_nonzero(mp):
        ax.semilogy(t[mp], np.maximum(ep[mp], 1e-16), "-o", lw=1.2, ms=3, mew=0, label="p")
    if not (np.count_nonzero(mu) or np.count_nonzero(mp)):
        ax.text(0.5, 0.5, "No finite validation points", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_error_plot_two_rel_abs(
    times: np.ndarray,
    rel_u: np.ndarray,
    rel_p: np.ndarray,
    abs_u: np.ndarray,
    abs_p: np.ndarray,
    out: Path,
    title: str,
    *,
    rel_ylabel: str = "L2 relative error",
    abs_ylabel: str = "L2 absolute error",
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    t = np.asarray(times, dtype=float).reshape(-1)
    rel_u = np.asarray(rel_u, dtype=float).reshape(-1)
    rel_p = np.asarray(rel_p, dtype=float).reshape(-1)
    abs_u = np.asarray(abs_u, dtype=float).reshape(-1)
    abs_p = np.asarray(abs_p, dtype=float).reshape(-1)
    if not (t.size == rel_u.size == rel_p.size == abs_u.size == abs_p.size):
        raise ValueError("times/error length mismatch")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.2, 5.0), dpi=150)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    mu = np.isfinite(t) & np.isfinite(rel_u)
    mp = np.isfinite(t) & np.isfinite(rel_p)
    if np.count_nonzero(mu):
        ax1.semilogy(t[mu], np.maximum(rel_u[mu], 1e-16), "-o", lw=1.2, ms=3, mew=0, label="U")
    if np.count_nonzero(mp):
        ax1.semilogy(t[mp], np.maximum(rel_p[mp], 1e-16), "-o", lw=1.2, ms=3, mew=0, label="p")
    if not (np.count_nonzero(mu) or np.count_nonzero(mp)):
        ax1.text(0.5, 0.5, "No finite validation points", transform=ax1.transAxes, ha="center", va="center", fontsize=9)
    ax1.set_ylabel(rel_ylabel)
    ax1.set_title(title)
    ax1.grid(True, which="both", ls=":", lw=0.5)
    ax1.legend(loc="best", frameon=False)

    mu = np.isfinite(t) & np.isfinite(abs_u)
    mp = np.isfinite(t) & np.isfinite(abs_p)
    if np.count_nonzero(mu):
        ax2.semilogy(t[mu], np.maximum(abs_u[mu], 1e-16), "-o", lw=1.2, ms=3, mew=0, label="U")
    if np.count_nonzero(mp):
        ax2.semilogy(t[mp], np.maximum(abs_p[mp], 1e-16), "-o", lw=1.2, ms=3, mew=0, label="p")
    if not (np.count_nonzero(mu) or np.count_nonzero(mp)):
        ax2.text(0.5, 0.5, "No finite validation points", transform=ax2.transAxes, ha="center", va="center", fontsize=9)
    ax2.set_xlabel("time")
    ax2.set_ylabel(abs_ylabel)
    ax2.grid(True, which="both", ls=":", lw=0.5)
    ax2.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_model_timeline_plot(times: np.ndarray, models: list[str], out: Path, title: str = "Model selection timeline") -> None:
    import matplotlib.pyplot as plt  # type: ignore

    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size != len(models):
        raise ValueError("times/models length mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 2.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    uniq = ["FOM", "ROM"]
    colors = {"FOM": "#d95f02", "ROM": "#1b9e77"}

    # Shade segments [t[i-1], t[i]] by the model used to reach t[i].
    for i in range(1, t.size):
        m = models[i].upper()
        if m not in colors:
            continue
        ax.axvspan(t[i - 1], t[i], color=colors[m], alpha=0.15, lw=0)

    # Step plot for model index (visual continuity).
    y = np.array([uniq.index(m) if m in uniq else np.nan for m in models], dtype=float)
    ax.step(t, y, where="post", lw=1.5, color="k")
    ax.plot(t, y, "o", ms=3, color="k")
    ax.set_yticks([0, 1], labels=uniq)
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.grid(True, axis="x", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_speed_bar(wall: dict[str, float], out: Path, title: str = "Wall time breakdown") -> None:
    import matplotlib.pyplot as plt  # type: ignore

    labels = ["FOM", "ROM", "I/O"]
    vals = [float(wall.get("fom_s", 0.0)), float(wall.get("rom_s", 0.0)), float(wall.get("io_s", 0.0))]
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5.5, 2.8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, vals)
    ax.set_ylabel("seconds")
    ax.set_title(title)
    ax.grid(True, axis="y", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_forces_plot(times: np.ndarray, Fx: np.ndarray, Fy: np.ndarray, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    t = np.asarray(times, dtype=float).reshape(-1)
    Fx = np.asarray(Fx, dtype=float).reshape(-1)
    Fy = np.asarray(Fy, dtype=float).reshape(-1)
    if t.size != Fx.size or t.size != Fy.size:
        raise ValueError("Forces series length mismatch")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 3.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, Fx, "-", lw=1.2, label="Fx (drag)")
    ax.plot(t, Fy, "-", lw=1.2, label="Fy (lift)")
    ax.set_xlabel("time")
    ax.set_ylabel("Force")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_force_coeffs_plot(
    t: np.ndarray,
    cd: np.ndarray,
    cl: np.ndarray,
    out: Path,
    title: str,
    *,
    t2: np.ndarray | None = None,
    cd2: np.ndarray | None = None,
    cl2: np.ndarray | None = None,
    t3: np.ndarray | None = None,
    cd3: np.ndarray | None = None,
    cl3: np.ndarray | None = None,
    label1: str = "FOM-only",
    label2: str = "adaptive (FOM segments)",
    label3: str = "ROM fields (postProcess)",
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    # Use distinct colors for Cd vs Cl (even for the same "source") so the legend remains readable.
    # Styles (solid/dashed/markers) still distinguish sources within each variable.
    cd_color = "#1f77b4"  # blue
    cl_color = "#d62728"  # red

    def robust_ylim(*series: np.ndarray | None) -> tuple[float, float] | None:
        ys = []
        for s in series:
            if s is None:
                continue
            a = np.asarray(s, dtype=float).reshape(-1)
            a = a[np.isfinite(a)]
            if a.size:
                ys.append(a)
        if not ys:
            return None
        y = np.concatenate(ys)
        if y.size < 2:
            m = float(np.median(y))
            return (m - 1.0, m + 1.0)
        lo, hi = np.percentile(y, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if float(hi) <= float(lo):
            m = float(np.median(y))
            return (m - 1.0, m + 1.0)
        pad = 0.08 * float(hi - lo)
        return (float(lo - pad), float(hi + pad))

    t = np.asarray(t, dtype=float).reshape(-1)
    cd = np.asarray(cd, dtype=float).reshape(-1)
    cl = np.asarray(cl, dtype=float).reshape(-1)
    if t.size != cd.size or t.size != cl.size:
        raise ValueError("Coefficient series length mismatch")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.6, 4.8), dpi=150)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(t, cd, "-", lw=1.2, color=cd_color, label=f"$C_d$ ({label1})")
    ax2.plot(t, cl, "-", lw=1.2, color=cl_color, label=f"$C_l$ ({label1})")
    if cd2 is not None and cl2 is not None:
        cd2 = np.asarray(cd2, dtype=float).reshape(-1)
        cl2 = np.asarray(cl2, dtype=float).reshape(-1)
        if cd2.size > 1 and cl2.size > 1:
            if t2 is None:
                # Use same time axis for the overlay only if lengths match; otherwise skip overlay.
                if cd2.size == t.size:
                    ax1.plot(t, cd2, "--", lw=1.2, color=cd_color, label=f"$C_d$ ({label2})")
                if cl2.size == t.size:
                    ax2.plot(t, cl2, "--", lw=1.2, color=cl_color, label=f"$C_l$ ({label2})")
            else:
                t2 = np.asarray(t2, dtype=float).reshape(-1)
                if t2.size == cd2.size:
                    ax1.plot(t2, cd2, "--", lw=1.2, color=cd_color, label=f"$C_d$ ({label2})")
                if t2.size == cl2.size:
                    ax2.plot(t2, cl2, "--", lw=1.2, color=cl_color, label=f"$C_l$ ({label2})")

    if cd3 is not None and cl3 is not None:
        cd3 = np.asarray(cd3, dtype=float).reshape(-1)
        cl3 = np.asarray(cl3, dtype=float).reshape(-1)
        if cd3.size > 1 and cl3.size > 1:
            if t3 is None:
                if cd3.size == t.size:
                    ax1.plot(t, cd3, "o", ms=2.0, lw=0, label=f"$C_d$ ({label3})")
                if cl3.size == t.size:
                    ax2.plot(t, cl3, "o", ms=2.0, lw=0, label=f"$C_l$ ({label3})")
            else:
                t3 = np.asarray(t3, dtype=float).reshape(-1)
                if t3.size == cd3.size:
                    ax1.plot(
                        t3,
                        cd3,
                        "o",
                        ms=2.0,
                        lw=0,
                        alpha=0.9,
                        color=cd_color,
                        label=f"$C_d$ ({label3})",
                    )
                if t3.size == cl3.size:
                    ax2.plot(
                        t3,
                        cl3,
                        "o",
                        ms=2.0,
                        lw=0,
                        alpha=0.9,
                        color=cl_color,
                        label=f"$C_l$ ({label3})",
                    )

    ax1.set_ylabel("$C_d$")
    ax2.set_ylabel("$C_l$")
    ax2.set_xlabel("time")
    lim1 = robust_ylim(cd, cd2, cd3)
    lim2 = robust_ylim(cl, cl2, cl3)
    if lim1 is not None:
        ax1.set_ylim(*lim1)
    if lim2 is not None:
        ax2.set_ylim(*lim2)
    ax1.grid(True, ls=":", lw=0.5)
    ax2.grid(True, ls=":", lw=0.5)
    # Single combined legend at the top keeps text from overlapping the data.
    handles = []
    labels = []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    # Preserve order but drop duplicates.
    uniq: dict[str, object] = {}
    for h, l in zip(handles, labels, strict=False):
        if l and l not in uniq:
            uniq[l] = h

    fig.suptitle(title, y=0.985)
    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_fft_plot(
    freq: np.ndarray,
    amp: np.ndarray,
    out: Path,
    title: str,
    *,
    xlabel: str = "frequency [Hz]",
    ylabel: str = "amplitude",
    xlim: tuple[float, float] | None = (0.0, 10.0),
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    f = np.asarray(freq, dtype=float).reshape(-1)
    a = np.asarray(amp, dtype=float).reshape(-1)
    if f.size != a.size:
        raise ValueError("freq/amp length mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.8, 3.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f, a, "-", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is None:
        # Auto-crop the x-range to avoid long near-zero tails (common for short signals).
        m = np.isfinite(f) & np.isfinite(a) & (f >= 0)
        fp = f[m]
        ap = a[m]
        if fp.size >= 8 and fp.size == ap.size:
            # Work on positive frequencies only (ignore DC bin at 0 for peak detection).
            pos = fp > 0
            fpos = fp[pos]
            apos = ap[pos]
            if fpos.size:
                amax = float(np.max(apos)) if np.isfinite(np.max(apos)) else 0.0
                if amax > 0:
                    thr = max(amax * 1e-3, 1e-16)
                    keep = apos >= thr
                    if np.any(keep):
                        fmax = float(np.max(fpos[keep])) * 1.2
                        fmax = min(fmax, float(np.max(fpos)))
                        ax.set_xlim(0.0, fmax)
    else:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.grid(True, ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
