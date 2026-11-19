"""Overview figure of one hb-correlation pass (density, species vs time, orientation, C_PT, C_HB, FES)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CHANNELS = ["A L1OH->L1H2O", "B L1H2O->L2OH", "C L1H2O->L1OH", "D L2H2O->L1OH"]  # = hb_correlation_core.CHANNELS


def plot_overview(path: Path, s: dict, raw: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nf = int(s["frames"]); dt = float(s["dt_ps"])
    zd, rho = s["z_density_A"], s["density_g_cm3"]
    spec = np.fromfile(Path(raw) / "species.i16", dtype="<i2").reshape(nf, 4)
    ch = lambda r, k: 5 + r * 5 + k
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
    a = ax[0, 0]; m = (zd > 0.5) & (zd < 12)
    a.plot(zd[m], rho[[ch(r, 2) for r in range(4)]][:, m].sum(0), label="H$_2$O", color="#e67e22")
    a.plot(zd[m], rho[[ch(r, 1) for r in range(4)]][:, m].sum(0), label="OH$^-$ (mobile O)", color="#1f77b4")
    a.plot(zd[m], rho[1, m], label="O$_s$-H (lattice O)", color="#ff6ec7")
    a.set_xlabel("z$_{rel}$ (Å)"); a.set_ylabel("density (g cm$^{-3}$)"); a.legend(fontsize=8, frameon=False); a.set_title("density, O-centred species")
    a = ax[0, 1]; t = np.arange(nf) * dt / 1000; k = max(1, int(round(20 / dt)))
    for j, (lab, c) in enumerate((("O$_s$-H", "#ff6ec7"), ("L$_1$-OH$^-$", "#1f77b4"), ("L$_2$-OH$^-$", "#8e5bd6"), ("bulk OH$^-$", "#2ca02c"))):
        cs = np.cumsum(np.insert(spec[:, j].astype(float), 0, 0)); sm = (cs[k:] - cs[:-k]) / k
        a.plot(t[k // 2: k // 2 + len(sm)][::50], sm[::50], lw=0.8, color=c, label=f"{lab} {spec[:, j].mean():.2f}")
    a.set_xlabel("t (ns)"); a.set_ylabel("per box (20 ps mean)"); a.legend(fontsize=8, frameon=False, ncol=2); a.set_title("species vs time")
    a = ax[0, 2]; zo, ang, ori = s["z_orientation_A"], s["angle_deg"], s["orientation_bonds_per_frame"]
    for cho, lab, c in ((1, "O$_s$-H", "#ff6ec7"), (0, "OH$^-$ mobile", "#1f77b4"), (3, "H$_2$O, z<4 Å", "#e67e22")):
        p = ori[cho][zo < 4.0].sum(0)
        if p.sum() > 0:
            a.plot(ang, p / p.sum() / 0.5, color=c, label=lab)
    a.set_xlabel("angle of O→H to +z (deg)"); a.set_ylabel("P (deg$^{-1}$)"); a.legend(fontsize=8, frameon=False); a.set_title("O-H orientation, z$_{rel}$ < 4 Å")
    lag = s["lag_ps"]; C, Cb, den = s["C"], s["C_blocks"], s["seed_denominators"]
    col = ["#1f77b4", "#8e5bd6", "#2ca02c", "#d62728"]
    for o, (a, nm) in enumerate(zip(ax[1, :2], ("C$_{PT}$", "C$_{HB}$"))):
        for c4 in range(4):
            a.semilogx(lag[1:], C[1, c4, o, 1:], color=col[c4], label=f"{CHANNELS[c4]} (n={int(den[1, c4])})")
            if Cb.shape[0] > 1:
                ok = np.isfinite(Cb[:, 1, c4, o, 1]).sum()
                if ok > 1:
                    sd = np.nanstd(Cb[:, 1, c4, o, 1:], axis=0, ddof=1) / np.sqrt(ok)
                    a.fill_between(lag[1:], C[1, c4, o, 1:] - sd, C[1, c4, o, 1:] + sd, color=col[c4], alpha=0.2, lw=0)
        a.set_xlabel("lag (ps)"); a.set_ylabel(nm); a.set_title(f"{nm}, strict geometry (band: SEM over blocks)"); a.set_ylim(0, 1.02)
    ax[1, 0].legend(fontsize=7, frameon=False)
    a = ax[1, 2]; F = s["fes_meV"][1, 1]
    im = a.pcolormesh(s["delta_A"], s["R_OO_A"], np.where(np.isfinite(F), F, np.nan).T, vmin=0, vmax=250, cmap="viridis", shading="auto")
    a.set_xlim(-1.3, 1.3); a.set_ylim(2.3, 3.2); a.set_xlabel("δ (Å)"); a.set_ylabel("R$_{OO}$ (Å)"); a.set_title("FES L1–L2 pair, strict (meV)"); fig.colorbar(im, ax=a)
    fig.suptitle(f"{title}: {nf} frames = {nf * dt / 1000:.2f} ns (waterint hb-correlation)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
