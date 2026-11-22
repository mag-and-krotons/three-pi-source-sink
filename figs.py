"""Figures 1-6 of "The Three-pi Singularity Structure as a Closed, Self-Recharging Source and Sink".
Built only from Definitions 1-6 (twisted frame), exact propagation by full diagonalisation. Writes figs/fig1..6.pdf.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from math import pi, sqrt, sin, cos

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL, FULL = 3.4, 7.0
N, R, P = 28, 14, 0
TW = (9, 18, 27)
a = np.ones(N, complex); a[list(TW)] = 1j
H = np.zeros((N, N), complex); Jop = np.zeros((N, N), complex)
for j in range(N):
    k = (j + 1) % N
    H[k, j] += -a[j]; H[j, k] += -np.conj(a[j])
    Jop[k, j] += -1j * a[j] / N; Jop[j, k] += 1j * np.conj(a[j]) / N
E, V = np.linalg.eigh(H); g = V[:, 0]; E0 = E[0]
FS = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
CMAP = plt.get_cmap("viridis")
FCOL = {f: CMAP(0.05 + 0.9 * i / 5) for i, f in enumerate(FS)}
os.makedirs("figs", exist_ok=True)

def beta(f): return 1 - sqrt(1 - f)
def evolve_many(psi, ts):
    c = V.conj().T @ psi
    return (np.exp(-1j * np.outer(ts, E)) * c) @ V.T
def evolve(psi, t): return V @ (np.exp(-1j * E * t) * (V.conj().T @ psi))
def G_RR(ts, Hm=None):
    if Hm is None:
        return np.exp(-1j * np.outer(ts, E)) @ (np.abs(V[R, :]) ** 2)
    e, v = np.linalg.eigh(Hm)
    return np.exp(-1j * np.outer(ts, e)) @ (np.abs(v[R, :]) ** 2)
def ring(Phi):
    Hm = np.zeros((N, N), complex); ph = np.exp(1j * Phi / N)
    for j in range(N):
        k = (j + 1) % N; Hm[k, j] += -ph; Hm[j, k] += -np.conj(ph)
    return Hm

# ---------------- Fig. 1 ----------------
def fig1():
    fig, ax = plt.subplots(figsize=(COL, COL * 0.95))
    ang = lambda j: pi / 2 - 2 * pi * j / N
    xy = np.array([[cos(ang(j)), sin(ang(j))] for j in range(N)])
    for j in range(N):
        k = (j + 1) % N
        tw = j in TW
        ax.plot(*zip(xy[j], xy[k]), color="#d62728" if tw else "0.45", lw=2.6 if tw else 0.8, zorder=1,
                solid_capstyle="round")
        if tw:
            m = (xy[j] + xy[k]) / 2; m = m / np.linalg.norm(m) * 1.13
            ax.text(*m, r"$\times i$", color="#d62728", ha="center", va="center", fontsize=8)
    special = {R: ("#1f77b4", "reader (+)\n$j=14$"), P: ("#ff7f0e", "partner (−)\n$j=0$"),
               7: ("#2ca02c", "observer\n$j=7$"), 21: ("#2ca02c", "observer\n$j=21$")}
    for j in range(N):
        c = special[j][0] if j in special else "white"
        ax.scatter(*xy[j], s=95 if j in special else 38, facecolor=c, edgecolor="k", lw=0.6, zorder=3)
        ax.text(*(xy[j] * 0.86), str(j), ha="center", va="center", fontsize=5.5)
    for j, (c, lab) in special.items():
        p = xy[j] * 1.42
        ax.text(*p, lab, color=c, ha="center", va="center", fontsize=7)
    ax.text(0, 0.12, r"$\Phi=\frac{3\pi}{2}$", ha="center", va="center", fontsize=9)
    ax.text(0, -0.18, r"$J_0=-\frac{2}{N}\sin\frac{\pi}{56}$", ha="center", va="center", fontsize=8)
    arr = FancyArrowPatch((-0.3, 0.42), (0.3, 0.42), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                          mutation_scale=9, lw=0.8, color="k")
    ax.add_patch(arr)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig("figs/fig1.pdf"); plt.close(fig)

# ---------------- Fig. 2 ----------------
def fig2():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 4.4))
    ell = np.arange(N); x = np.linspace(0, N - 1, 400)
    a1.plot(x, -2 * np.cos((2 * x + 1) * pi / 56), color="0.4", lw=0.8, label=r"closed form $-2\cos[(2\ell+1)\pi/56]$")
    a1.plot(ell, E, "o", ms=3.2, mfc="none", mec="#1f77b4", mew=0.8, label="numerical diagonalisation")
    a1.annotate("", xy=(13.5, E[14]), xytext=(13.5, E[13]), arrowprops=dict(arrowstyle="<->", lw=0.7))
    a1.text(14.3, (E[13] + E[14]) / 2, r"$\Delta_{1/2}=4\sin\frac{\pi}{56}=%.4f$" % (4 * sin(pi / 56)), va="center",
            fontsize=7)
    a1.set_xlabel(r"level index $\ell$"); a1.set_ylabel(r"$E_\ell$ (switch units)")
    a1.legend(loc="upper left", frameon=False); a1.text(0.01, 1.02, "(a)", transform=a1.transAxes, weight="bold")
    Ph = np.linspace(-pi, 3 * pi, 1601)
    J0 = []
    for p in Ph:
        Hm = ring(p); e, v = np.linalg.eigh(Hm); gg = v[:, 0]
        Jm = np.zeros((N, N), complex); ph = np.exp(1j * p / N)
        for j in range(N):
            k = (j + 1) % N; Jm[k, j] += -1j * ph / N; Jm[j, k] += 1j * np.conj(ph) / N
        J0.append(np.real(gg.conj() @ Jm @ gg))
    J0 = np.array(J0)
    jump = np.where(np.abs(np.diff(J0)) > 5e-3)[0] + 1
    for seg in np.split(np.arange(len(Ph)), jump):
        a2.plot(Ph[seg] / pi, J0[seg] * 1e3, color="k", lw=0.9)
    a2.axhline(0, color="0.6", lw=0.5)
    a2.axvline(1.5, color="#d62728", ls="--", lw=0.7)
    Jstar = -(2 / N) * sin(pi / 56)
    a2.plot(1.5, Jstar * 1e3, "o", color="#d62728", ms=4)
    a2.text(1.58, -6.2, r"$\Phi=3\pi/2$: $J_0=%.3f\times10^{-3}$" % (Jstar * 1e3), color="#d62728", fontsize=6.5)
    a2.set_xlabel(r"total twist flux $\Phi/\pi$"); a2.set_ylabel(r"ground-state flow $J_0$ ($10^{-3}$)")
    a2.text(0.01, 1.02, "(b)", transform=a2.transAxes, weight="bold")
    fig.tight_layout(h_pad=1.2); fig.savefig("figs/fig2.pdf"); plt.close(fig)

# ---------------- Fig. 3 ----------------
def cycle_series(f, T=70.0, dt=0.05, period=20.0, use=5.0):
    b = beta(f); ts = np.arange(0, T + dt / 2, dt)
    content, flow, low = [], [], []
    starts = [g.copy()]                                    # state at each draw instant, propagated, not assumed
    for c in range(int(T // period) + 1):
        s = starts[-1]; d = np.zeros(N, complex); d[R] = b * s[R]
        back = evolve(s - d, use) + evolve(d, use)         # return along the closure (Definition 6)
        starts.append(evolve(back, period - use))
    for t in ts:
        c = int(t // period); tin = t - c * period
        base = starts[c]
        if tin < use:
            d = np.zeros(N, complex); d[R] = b * base[R]
            band = evolve(base - d, tin)
        else:
            band = evolve(base, tin)
        nb = np.vdot(band, band).real
        content.append(nb); flow.append(np.real(band.conj() @ Jop @ band) / nb)
        low.append(abs(np.vdot(g, band)) ** 2 / nb)
    return ts, np.array(content), np.array(flow), np.array(low)

def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2))
    for f in FS:
        ts, c, fl, lo = cycle_series(f)
        axs[0].plot(ts, c, color=FCOL[f], label=r"$f=%g$" % f)
        axs[1].plot(ts, fl * 1e3, color=FCOL[f])
        axs[2].plot(ts, lo, color=FCOL[f])
    axs[0].set_ylabel(r"band content $\|\psi_{\rm band}\|^2$")
    axs[1].set_ylabel(r"flow $J$ ($10^{-3}$)")
    axs[2].set_ylabel(r"ground-state weight $|\langle g|\hat\psi\rangle|^2$")
    for i, ax in enumerate(axs):
        ax.set_xlabel(r"$t$ (switches)"); ax.text(0.01, 1.02, "(%s)" % "abc"[i], transform=ax.transAxes, weight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.5); fig.savefig("figs/fig3.pdf"); plt.close(fig)

# ---------------- Fig. 4 ----------------
def reader_curve(f, ts):
    b = beta(f); G = G_RR(ts).real
    return 1 - 2 * b * G * np.cos(E0 * ts) + b ** 2 * G ** 2

def fig4():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.3))
    ts = np.linspace(0, 12, 2401)
    for f in FS:
        axs[0].plot(ts, reader_curve(f, ts) / (1 - f / N), color=FCOL[f], label=r"$f=%g$" % f)
    axs[0].axvline(pi / (2 * abs(E0)), color="0.5", ls="--", lw=0.6)
    axs[0].set_xlabel(r"$t$ (switches)"); axs[0].set_ylabel(r"reader share $Ns_R(t)/B_f$")
    axs[0].legend(ncol=2, frameon=False, loc="lower right")
    t2 = np.linspace(0, 30, 3001)
    axs[1].plot(t2, np.abs(G_RR(t2, ring(0.0))), color="0.55", label=r"untwisted ring, $\Phi=0$")
    axs[1].plot(t2, np.abs(G_RR(t2)), color="#d62728", label=r"three-$\pi$ band, $\Phi=3\pi/2$")
    t3 = t2[t2 > 0.3]
    axs[1].plot(t3, (pi * t3) ** -0.5, ":", color="k", lw=0.8, label=r"envelope $(\pi t)^{-1/2}$")
    axs[1].set_ylim(0, 1.02); axs[1].set_xlabel(r"$t$ (switches)"); axs[1].set_ylabel(r"return amplitude $|G_{RR}(t)|$")
    axs[1].legend(frameon=False, loc="upper right")
    fgrid = np.linspace(0.01, 1.0, 100)
    dt = 1e-3; T = np.arange(0, 25 + dt / 2, dt); G = G_RR(T).real; use = T <= 5
    tau1, tau01, bound, mdef = [], [], [], []
    M2 = 2 + E0 ** 2
    for f in fgrid:
        b = beta(f); B = 1 - f / N
        ns = 1 - 2 * b * G * np.cos(E0 * T) + b ** 2 * G ** 2
        tau1.append(T[np.argmax(ns >= B)])
        dev = np.abs(ns / B - 1); bad = np.where(dev > 0.1)[0]
        tau01.append(np.nan if len(bad) == 0 else min(T[bad[-1]], 24.0))
        bound.append(sqrt(f * (1 - 1 / N) / (b * M2)))
        mdef.append(np.mean(dev[use]))
    ax = axs[2]
    ax.plot(fgrid, tau1, color="#1f77b4", label=r"first refill $\tau_1$")
    ax.plot(fgrid, bound, "--", color="#1f77b4", lw=0.8, label="bound (Thm. 3)")
    ax.plot(fgrid, tau01, color="#d62728", label=r"settling $\tau_{0.1}$")
    ax.set_yscale("log"); ax.set_ylim(0.1, 40); ax.set_xlabel(r"draw fraction $f$"); ax.set_ylabel("time (switches)")
    ax2 = ax.twinx(); ax2.plot(fgrid, mdef, color="#2ca02c", label=r"mean deficit, $t<5$")
    ax2.set_ylabel("mean reader deficit", color="#2ca02c"); ax2.tick_params(axis="y", colors="#2ca02c")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=6)
    for i, a_ in enumerate(axs):
        a_.text(0.01, 1.02, "(%s)" % "abc"[i], transform=a_.transAxes, weight="bold")
    fig.tight_layout(w_pad=1.2); fig.savefig("figs/fig4.pdf"); plt.close(fig)

# ---------------- Fig. 5 ----------------
def fig5():
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ts = np.linspace(0, 12, 481)
    d = np.zeros(N, complex); d[R] = g[R]
    st = evolve_many(g - d, ts)
    rel = N * np.abs(st) ** 2 / (1 - 1 / N) - 1
    im = axs[0].pcolormesh(ts, np.arange(N), rel.T, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest",
                           rasterized=True)
    for s in (+1, -1):
        axs[0].plot(ts, R + s * 2 * ts, "k--", lw=0.7)
    for j, c in ((7, "#2ca02c"), (21, "#2ca02c"), (0, "#ff7f0e")):
        axs[0].axhline(j, color=c, ls=":", lw=0.8)
    axs[0].set_ylim(-0.5, N - 0.5); axs[0].set_xlim(0, 12)
    axs[0].set_xlabel(r"$t$ (switches)"); axs[0].set_ylabel(r"site $j$")
    cb = fig.colorbar(im, ax=axs[0], pad=0.02); cb.set_label(r"$Ns_j/B_f-1$  ($f=1$)")
    fgrid = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]); tu = np.linspace(0, 5, 1001)
    marks = {13: ("s", "#9467bd"), 21: ("^", "#2ca02c"), 7: ("v", "#bcbd22"), 0: ("o", "#ff7f0e"), 14: ("x", "k")}
    labels = {13: r"$j=13$", 21: r"$j=21$", 7: r"$j=7$", 0: r"$j=0$ (partner)", 14: r"$j=14$ (reader)"}
    for j, (m, c) in marks.items():
        vals = []
        for f in fgrid:
            d = np.zeros(N, complex); d[R] = beta(f) * g[R]
            vals.append(np.min(N * np.abs(evolve_many(g - d, tu)[:, j]) ** 2))
        axs[1].plot(fgrid, vals, marker=m, color=c, ms=3.5, label=labels[j])
    axs[1].set_xlabel(r"draw fraction $f$"); axs[1].set_ylabel(r"min. share during use, $N\min_t s_j$")
    axs[1].legend(frameon=False, loc="lower left")
    for i, a_ in enumerate(axs):
        a_.text(0.01, 1.02, "(%s)" % "ab"[i], transform=a_.transAxes, weight="bold")
    fig.tight_layout(w_pad=1.0); fig.savefig("figs/fig5.pdf", dpi=300); plt.close(fig)

# ---------------- Fig. 6 ----------------
def fig6():
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    styles = {0.1: "#1f3a93", 0.5: "#16a085", 1.0: "#9acd32"}
    for f, c in styles.items():
        b = beta(f)
        for mode, ls in (("closure", "-"), ("fixed", "--")):
            psi = g.copy(); tt, cc = [], []
            for cyc in range(20):
                t0 = 20.0 * cyc
                amp = b * psi[R]; d = np.zeros(N, complex); d[R] = amp
                band0 = psi - d
                for tin in np.linspace(0, 5, 26)[:-1]:
                    bt = evolve(band0, tin); tt.append(t0 + tin); cc.append(np.vdot(bt, bt).real)
                bt = evolve(band0, 5.0)
                if mode == "closure":
                    psi = bt + evolve(d, 5.0)
                else:
                    pv = np.zeros(N, complex); pv[P] = amp; psi = bt - pv
                for tin in np.linspace(5, 20, 31)[:-1]:
                    tt.append(t0 + tin); cc.append(np.vdot(psi, psi).real)
                psi = evolve(psi, 15.0)
            lab = ("closure return" if mode == "closure" else r"fixed-site $\pi$ reinjection") + r", $f=%g$" % f
            ax.plot(tt, cc, ls=ls, color=c, lw=0.8, label=lab)
    ax.set_xlabel(r"$t$ (switches)"); ax.set_ylabel("band content"); ax.set_ylim(0.2, 1.03)
    ax.legend(frameon=False, ncol=2, fontsize=5.5, loc="lower left")
    fig.savefig("figs/fig6.pdf"); plt.close(fig)

if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        fn(); print(fn.__name__, "done")
