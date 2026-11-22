"""Independent recomputation of every number in
"The Three-pi Singularity Structure as a Closed, Self-Recharging Source and Sink" (A. Singh, 22 Nov 2026).

Built only from Definitions 1-6 of the paper (the twisted frame, Eq. (1)); no gauge reduction is used for the
dynamics. Exact propagation through a full diagonalisation. Writes results/analysis.json.
SMOKE=1 runs a reduced grid.
"""
import json, os, sys
import numpy as np
from scipy.special import jv
from math import factorial, pi, sin, cos, sqrt

SMOKE = os.environ.get("SMOKE") == "1"
N = 28
TW = (9, 18, 27)
R, P = 14, 0
out = {}

# ---------- Definitions 1-2 ----------
a = np.ones(N, complex)
for j in TW:
    a[j] = 1j
H = np.zeros((N, N), complex)
Jop = np.zeros((N, N), complex)
for j in range(N):
    k = (j + 1) % N
    H[k, j] += -a[j]; H[j, k] += -np.conj(a[j])
    Jop[k, j] += -1j * a[j] / N; Jop[j, k] += 1j * np.conj(a[j]) / N
assert np.allclose(H, H.conj().T) and np.allclose(Jop, Jop.conj().T)
E, V = np.linalg.eigh(H)
g = V[:, 0]
E0 = E[0]

def U(t):
    return (V * np.exp(-1j * E * t)) @ V.conj().T

def evolve(psi, t):
    return V @ (np.exp(-1j * E * t) * (V.conj().T @ psi))

# ---------- Theorem 1 ----------
ell = np.arange(N)
closed = np.sort(-2 * np.cos((2 * ell + 1) * pi / 56))
out["thm1"] = {
    "holonomy": complex(np.prod(a)).__repr__(),
    "max_|E_numeric-E_closed|": float(np.max(np.abs(E - closed))),
    "min_level_spacing": float(np.min(np.diff(E))),
    "E0": float(E0), "E0_closed": -2 * cos(pi / 56),
    "uniform_weight_max_dev": float(np.max(np.abs(np.abs(g) ** 2 - 1 / N))),
    "J0": float(np.real(g.conj() @ Jop @ g)), "J0_closed": -(2 / N) * sin(pi / 56),
    "E1-E0": float(E[1] - E[0]), "E1-E0_closed": 4 * sin(pi / 28) * sin(pi / 56),
    "E14-E13": float(E[14] - E[13]), "mid_gap_closed": 4 * sin(pi / 56),
}
# J0 = dE0/dPhi by finite difference on a uniform flux
def E0_of_Phi(Phi):
    Hu = np.zeros((N, N), complex); ph = np.exp(1j * Phi / N)
    for j in range(N):
        k = (j + 1) % N; Hu[k, j] += -ph; Hu[j, k] += -np.conj(ph)
    return np.linalg.eigvalsh(Hu)[0]
h = 1e-6
out["thm1"]["dE0/dPhi_fd"] = float((E0_of_Phi(1.5 * pi + h) - E0_of_Phi(1.5 * pi - h)) / (2 * h))
# Corollary 1 in the uniform frame (Lemma 1)
chi = np.zeros(N); phi = 1.5 * pi / N
for j in range(N - 1):
    chi[j + 1] = chi[j] + np.angle(a[j]) - phi
D = np.diag(np.exp(1j * chi))
gt = D.conj().T @ g
out["cor1"] = {"<P|g~>/<R|g~>": repr(complex(np.round(gt[P] / gt[R], 12))),
               "<P|g>/<R|g> (twisted frame)": repr(complex(np.round(g[P] / g[R], 12)))}

# ---------- Theorem 2 / Table I ----------
fs = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00]
def beta(f):
    return 1 - sqrt(1 - f)

t1 = {}
Ud = U(0.01)
NSTEP = 4000 if SMOKE else 40000          # 400 switches in steps of 0.01 (Sec. VIII)
PERIOD, USE = 2000, 500                   # a draw every 20 switches, a use interval of 5
for f in fs:
    b = beta(f)
    band = g.copy(); d = np.zeros(N, complex); in_use = False; gref = g.copy()
    maxerr = band_min = low_min = None; ledger_content = ledger_energy = 0.0
    band_min, low_min, maxerr = 1.0, 1.0, 0.0
    eb_use = []
    for n in range(NSTEP + 1):
        if n % PERIOD == 0 and n < NSTEP:                 # draw (Definition 5)
            d = np.zeros(N, complex); d[R] = b * band[R]
            band = band - d; in_use = True
            eb_use.append(np.vdot(band, H @ band).real)
        if in_use:
            nb = np.vdot(band, band).real
            band_min = min(band_min, nb)
            low_min = min(low_min, abs(np.vdot(g, band)) ** 2 / nb)
            eb_use.append(np.vdot(band, H @ band).real)
        tot = band + d
        ledger_content = max(ledger_content, abs(np.vdot(tot, tot).real - 1))
        ledger_energy = max(ledger_energy, abs(np.vdot(tot, H @ tot).real - E0))
        if in_use and n % PERIOD == USE:                  # return along the closure (Definition 6)
            band = band + d; d = np.zeros(N, complex); in_use = False
            maxerr = max(maxerr, np.linalg.norm(band - gref))           # against the undisturbed band, same steps
            maxerr_exact = max(locals().get('maxerr_exact', 0.0), np.linalg.norm(band - np.exp(-1j * E0 * n * 0.01) * g))
        if n < NSTEP:
            band = Ud @ band; d = Ud @ d; gref = Ud @ gref
    flow_after = float(np.real(band.conj() @ Jop @ band))
    t1[f] = {"beta": b, "drawn_f/N": f / N, "band_min_sim": band_min, "band_closed": 1 - f / N,
             "lowest_min_sim": low_min, "lowest_closed_Eq13": (1 - b / N) ** 2 / (1 - f / N),
             "energy_in_use_sim(E_band-E0)": float(np.mean(eb_use) - E0), "energy_in_use_closed": 2 * b * abs(E0) / N,
             "band_energy_spread_during_use": float(np.ptp(eb_use)),
             "max_return_error": maxerr, "max_return_error_vs_exact_phase": maxerr_exact, "flow_after": flow_after,
             "lowest_weight_after": float(abs(np.vdot(g, band)) ** 2),
             "content_ledger_resid": ledger_content, "energy_ledger_resid": ledger_energy}
out["table1"] = t1

# Proposition 1: passivity check (random unitaries cannot lower the energy of g)
rng = np.random.default_rng(0)
mins = []
for _ in range(200):
    X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, _r = np.linalg.qr(X)
    v = Q @ g
    mins.append(np.vdot(v, H @ v).real - E0)
out["prop1_min_energy_above_E0_random_unitaries"] = float(min(mins))

# ---------- Lemma 2 ----------
def G_direct(t):
    return np.exp(-1j * np.outer(t, E)) @ (np.abs(V[R, :]) ** 2)

tt = np.linspace(0, 30, 3001)
Gd = G_direct(tt)
Gb = jv(0, 2 * tt) + sum(2 * cos(m * 1.5 * pi) * jv(m * N, 2 * tt) for m in range(1, 6))
G0 = jv(0, 2 * tt) + sum(2 * jv(m * N, 2 * tt) for m in range(1, 6))
out["lemma2"] = {"max_|Im G|": float(np.max(np.abs(Gd.imag))),
                 "max_|G_direct - Bessel|": float(np.max(np.abs(Gd - Gb))),
                 "2*14^56/56!": 2 * 14 ** 56 / factorial(56),
                 "untwisted_echo_max_|G|_t_in_[10,18]": float(np.max(np.abs(G0[(tt >= 10) & (tt <= 18)]))),
                 "three_pi_max_|G|_t_in_[10,18]": float(np.max(np.abs(Gd[(tt >= 10) & (tt <= 18)]))),
                 "envelope_(pi*14)^-1/2": (pi * 14) ** -0.5}

# ---------- Theorem 3 / Table II ----------
M2 = 2 + E0 ** 2
dt = 1e-3
T = np.arange(0, 25 + dt / 2, dt)
GT = G_direct(T).real

def NsR(t_idx, b):
    return 1 - 2 * b * GT[t_idx] * np.cos(E0 * T[t_idx]) + b ** 2 * GT[t_idx] ** 2

idx_all = np.arange(len(T))
use = T <= 5 + 1e-12
UT = None
t2 = {}
for f in fs:
    b = beta(f); B = 1 - f / N
    ns = NsR(idx_all, b)
    # direct check of Eq. (17) against propagation at a few times
    chk = 0.0
    for tq in (0.3, 1.7, 4.2, 11.0):
        d = np.zeros(N, complex); d[R] = b * g[R]
        pb = evolve(g - d, tq)
        chk = max(chk, abs(N * abs(pb[R]) ** 2 - (1 - 2 * b * G_direct(np.array([tq]))[0].real * cos(E0 * tq)
                                                   + b ** 2 * G_direct(np.array([tq]))[0].real ** 2)))
    above = np.where(ns >= B)[0]
    tau1 = float(T[above[0]]) if len(above) else None
    dev = np.abs(ns / B - 1)
    bad = np.where(dev > 0.1)[0]
    tau01 = 0.0 if len(bad) == 0 else (float(T[bad[-1]]) if T[bad[-1]] < 24.999 else ">24")
    c = 0.1 * B - f / N
    tstar = b ** 2 / (pi * (sqrt(1 + c) - 1) ** 2) if c > 0 else None
    shares = {}
    # minimum shares during the 5-switch use interval, exact propagation
    tu = T[use][::10]
    d0 = np.zeros(N, complex); d0[R] = b * g[R]
    ph = np.exp(-1j * np.outer(tu, E)) * (V.conj().T @ (g - d0))
    states = ph @ V.T
    for x in (R, 13, P, 7, 21):
        shares[x] = float(np.min(N * np.abs(states[:, x]) ** 2))
    t2[f] = {"beta": b, "tau1": tau1, "tau1_bound_Eq18": sqrt(f * (1 - 1 / N) / (b * M2)),
             "tau0.1": tau01, "t*0.1_Eq19": tstar, "mean_deficit_0-5": float(np.mean(dev[use])),
             "mean_deficit/beta": float(np.mean(dev[use]) / b),
             "min_share_R": shares[R], "min_share_13": shares[13], "min_share_P": shares[P],
             "min_share_7": shares[7], "min_share_21": shares[21], "Eq17_vs_propagation_maxdiff": chk}
out["table2"] = t2
out["thm3"] = {"M2": M2, "pi/(2|E0|)": pi / (2 * abs(E0))}

# ---------- Theorem 4 ----------
def L(d, t):
    return sum((2 * t) ** n / factorial(n) for n in range(d, d + 80))
out["thm4"] = {"L_14(1.82)": L(14, 1.82), "L_7(0.62)": L(7, 0.62),
               "max_group_velocity": float(np.max(np.abs(2 * np.sin(2 * pi * np.arange(N) / N - 1.5 * pi / N)))),
               "max_|G_PR|_t<=5": float(np.max(np.abs((np.exp(-1j * np.outer(T[use], E)) * V[R, :].conj()) @ V[P, :])))}

# ---------- Proposition 2 / Fig. 6 ----------
grid = np.arange(0, (200 if SMOKE else 20000) + 1e-9, 0.05)
amp = V[P, :] * V[R, :].conj()
mx = 0.0
for s in range(0, len(grid), 20000):
    blk = grid[s:s + 20000]
    mx = max(mx, float(np.max(np.abs(np.exp(-1j * np.outer(blk, E)) @ amp))))
GPR5 = complex(np.exp(-1j * E * 5.0) @ amp)
out["prop2"] = {"max_|G_PR|_grid": mx, "G_PR(5)": repr(complex(np.round(GPR5, 5))),
                "sqrt(2+2ReG_PR(5))": sqrt(2 + 2 * GPR5.real), "error_full_draw": sqrt(2 + 2 * GPR5.real) / sqrt(N)}
fig6 = {}
for f in (0.1, 0.5, 1.0):
    b = beta(f); psi = g.copy(); contents = []
    for c in range(2 if SMOKE else 20):
        amp_d = b * psi[R]
        d = np.zeros(N, complex); d[R] = amp_d
        pb = evolve(psi - d, 5.0)
        Pv = np.zeros(N, complex); Pv[P] = amp_d
        psi = pb - Pv
        contents.append(float(np.vdot(psi, psi).real))
        psi = evolve(psi, 15.0)
    fig6[f] = {"content_after_last_return": contents[-1], "first_three": contents[:3]}
out["fig6_fixed_site"] = fig6

# ---------- Sec. IX ----------
h_planck, e_ch = 6.62607015e-34, 1.602176634e-19
out["secIX_field_mT_r=1um"] = 0.75 * h_planck / e_ch / (pi * 1e-12) * 1e3

os.makedirs("results", exist_ok=True)
with open("results/analysis_smoke.json" if SMOKE else "results/analysis.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
