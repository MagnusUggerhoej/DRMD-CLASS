#!/usr/bin/env python3
"""
Gate 2: Scan over G_eff values and compare Thomas vs Mine interaction effect.
For each G_eff:
  ratio_T = P_T(INT) / P_T(FS)
  ratio_M = P_M(INT) / P_M(FS)
  mismatch = ratio_M / ratio_T - 1
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from classy      import Class as ClassThomas
from classy_NEDE import Class as ClassMine

# ─────────────────────────────────────────────
# Knobs
# ─────────────────────────────────────────────
G_EFF_VALUES = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]   # scan range
M_NCDM_EV    = 0.06
DEG_NCDM     = 1.0
Z_PK         = 0.0
K_LIST       = np.logspace(-4, 0, 200)   # 1/Mpc
RTOL_WARN    = 1e-3   # flag mismatches above this

# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
_unread_re        = re.compile(r"did not read input parameter\(s\):\s*(.*)", re.IGNORECASE)
_hyrec_missing_re = re.compile(r"compiled without including the HyRec code",  re.IGNORECASE)

def parse_unread(exc):
    m = _unread_re.search(str(exc))
    if not m: return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]

def run_with_autofix(ClassObj, params_in, label, max_passes=10):
    params = dict(params_in)
    stripped = []
    for attempt in range(1, max_passes + 1):
        c = ClassObj(); c.set(params)
        try:
            c.compute()
            return c, stripped
        except Exception as e:
            msg = str(e)
            if _hyrec_missing_re.search(msg):
                params["recombination"] = "recfast"; continue
            unread = parse_unread(e)
            if unread:
                removed = [k for k in unread if k in params]
                if not removed: raise RuntimeError(f"{label}: unread {unread} not in dict.")
                for k in removed: params.pop(k)
                stripped.extend(removed); continue
            print(f"=== {label}: FAILED ==="); raise
    raise RuntimeError(f"{label}: exceeded {max_passes} passes.")

# ─────────────────────────────────────────────
# Param builders
# ─────────────────────────────────────────────
def base():
    return {
        "output": "mPk", "z_pk": str(Z_PK), "P_k_max_1/Mpc": 5.0,
        "h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200,
        "A_s": 2.1e-9, "n_s": 0.9649, "tau_reio": 0.0543,
        "T_cmb": 2.7255, "Omega_k": 0.0,
        "lensing": "no", "gauge": "synchronous", "recombination": "recfast",
        "G_eff_ur": 0.0,
    }

def params_fs():
    p = base(); p["N_ur"] = 3.046; return p

def params_thomas_int(g):
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_interacting": 1, "m_ncdm_interacting": M_NCDM_EV,
              "deg_ncdm_interacting": DEG_NCDM, "G_eff_ncdm_interacting": float(g)})
    return p

def params_mine_int(g):
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_standard": 0, "N_ncdm_interacting": 1,
              "m_ncdm": M_NCDM_EV, "G_eff_ncdm_interacting": float(g)})
    return p

# ─────────────────────────────────────────────
# P(k) helper
# ─────────────────────────────────────────────
def get_pk(c):
    try:    return np.array([c.pk_lin(k, Z_PK) for k in K_LIST])
    except: return np.array([c.pk(k,     Z_PK) for k in K_LIST])

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Running free-streaming baseline...")
    cT_fs, _ = run_with_autofix(ClassThomas, params_fs(), "Thomas FS")
    cM_fs, _ = run_with_autofix(ClassMine,   params_fs(), "Mine   FS")
    pkT_fs = get_pk(cT_fs)
    pkM_fs = get_pk(cM_fs)
    h = cT_fs.h()
    k_hMpc = K_LIST / h

    # Storage
    results = []   # list of dicts per G_eff

    print(f"\n{'G_eff':<12s}  {'max|effect_T|':>14s}  {'max|effect_M|':>14s}  {'max|mismatch|':>14s}  {'rms|mismatch|':>14s}")
    print("─" * 72)

    colors = cm.viridis(np.linspace(0.1, 0.9, len(G_EFF_VALUES)))

    for g, col in zip(G_EFF_VALUES, colors):
        cT_int, _ = run_with_autofix(ClassThomas, params_thomas_int(g), f"Thomas g={g:.0e}")
        cM_int, _ = run_with_autofix(ClassMine,   params_mine_int(g),   f"Mine   g={g:.0e}")

        pkT_int = get_pk(cT_int)
        pkM_int = get_pk(cM_int)

        ratio_T  = pkT_int / pkT_fs
        ratio_M  = pkM_int / pkM_fs
        mismatch = ratio_M / ratio_T - 1.0

        flag = "  ← WARN" if np.max(np.abs(mismatch)) > RTOL_WARN else ""
        print(f"  {g:<10.2e}  {np.max(np.abs(ratio_T-1)):>14.3e}  "
              f"{np.max(np.abs(ratio_M-1)):>14.3e}  "
              f"{np.max(np.abs(mismatch)):>14.3e}  "
              f"{np.sqrt(np.mean(mismatch**2)):>14.3e}{flag}")

        results.append({"g": g, "col": col,
                        "ratio_T": ratio_T, "ratio_M": ratio_M,
                        "mismatch": mismatch})

        for c in (cT_int, cM_int):
            try: c.struct_cleanup(); c.empty()
            except: pass

    # ── Plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for r in results:
        lbl = f"G_eff={r['g']:.0e}"
        axes[0].semilogx(k_hMpc, r["ratio_T"] - 1, color=r["col"], label=lbl)
        axes[0].semilogx(k_hMpc, r["ratio_M"] - 1, color=r["col"], ls="--", alpha=0.6)
    axes[0].axhline(0, ls=":", c="k")
    axes[0].set_ylabel("INT/FS - 1  (solid=Thomas, dash=Mine)")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, which="both", ls=":")
    axes[0].set_title(f"Interaction effect on P(k)  (m={M_NCDM_EV} eV)")

    for r in results:
        axes[1].semilogx(k_hMpc, r["mismatch"], color=r["col"], label=f"G_eff={r['g']:.0e}")
    axes[1].axhline(0, ls=":", c="k")
    axes[1].axhline( RTOL_WARN, ls="--", c="r", lw=0.8, label=f"rtol={RTOL_WARN:.0e}")
    axes[1].axhline(-RTOL_WARN, ls="--", c="r", lw=0.8)
    axes[1].set_ylabel(r"Mismatch: $r_M/r_T - 1$")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, which="both", ls=":")

    axes[2].loglog(k_hMpc, pkT_fs * h**3, "k-",  label="FS baseline", lw=1.5)
    for r in results:
        pkT_int = pkT_fs * r["ratio_T"]
        axes[2].loglog(k_hMpc, pkT_int * h**3, color=r["col"],
                       label=f"G_eff={r['g']:.0e}", alpha=0.7)
    axes[2].set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    axes[2].set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
    axes[2].legend(fontsize=7, ncol=2); axes[2].grid(True, which="both", ls=":")

    plt.tight_layout(); plt.show()

    for c in (cT_fs, cM_fs):
        try: c.struct_cleanup(); c.empty()
        except: pass


if __name__ == "__main__":
    main()