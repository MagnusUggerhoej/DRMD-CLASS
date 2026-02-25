#!/usr/bin/env python3
"""
Degeneracy budget diagnostic
=============================
Problem: P(k) agreement collapses when deg > 1.
Hypothesis: one backend misinterprets the degeneracy key,
            leading to inconsistent radiation budget allocation.

This script runs both backends at deg=1.0 and deg=3.046,
then reads the background table at z~0 to check:
  - rho_ur   (massless neutrinos)
  - rho_ncdm (massive/interacting neutrinos, summed over species)
  - rho_g    (photons)
  - rho_crit
  - derived Omega_ur0, Omega_ncdm0

If one backend has leftover rho_ur when N_ur should be 0,
or if rho_ncdm doesn't scale with deg, that identifies the bug.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from classy      import Class as ClassThomas
from classy_NEDE import Class as ClassMine

# ─────────────────────────────────────────────
# Knobs
# ─────────────────────────────────────────────
DEG_VALUES = [1.0, 3.046]
G_EFF_ON   = 1e-1
M_NCDM_EV  = 0.06
K_LIST     = np.logspace(-4, 0, 200)   # for P(k) comparison
Z_PK       = 0.0

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
            print(f"  [{label}] OK" + (f" stripped={stripped}" if stripped else ""))
            return c, params
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
            print(f"  [{label}] FAILED"); raise
    raise RuntimeError(f"{label}: exceeded {max_passes} passes.")

# ─────────────────────────────────────────────
# Param builders
# ─────────────────────────────────────────────
def base():
    return {
        "output":        "mPk",
        "lensing":       "no",
        "gauge":         "synchronous",
        "recombination": "recfast",
        "h":             0.6736,
        "omega_b":       0.02237,
        "omega_cdm":     0.1200,
        "A_s":           2.1e-9,
        "n_s":           0.9649,
        "tau_reio":      0.0543,
        "T_cmb":         2.7255,
        "Omega_k":       0.0,
        "P_k_max_1/Mpc": 10.0,
        "z_pk":          str(Z_PK),
        "G_eff_ur":      0.0,
    }

def params_thomas(deg):
    p = base()
    p["N_ur"] = max(0.0, 3.046 - float(deg))
    p.update({
        "N_ncdm_interacting":   1,
        "m_ncdm_interacting":   M_NCDM_EV,
        "deg_ncdm_interacting": float(deg),
        "G_eff_ncdm_interacting": G_EFF_ON,
    })
    return p

def params_mine(deg):
    p = base()
    p["N_ur"] = max(0.0, 3.046 - float(deg))
    p.update({
        "N_ncdm_standard":      0,
        "N_ncdm_interacting":   1,
        "m_ncdm":               M_NCDM_EV,
        "deg_ncdm_interacting": float(deg),
        "G_eff_ncdm_interacting": G_EFF_ON,
    })
    return p

# ─────────────────────────────────────────────
# Background budget reader
# Robustly finds columns by substring matching
# so naming differences between builds don't matter
# ─────────────────────────────────────────────
def find_col(bg, substring):
    """Return first bg column whose name contains substring (case-insensitive)."""
    for k in bg:
        if substring.lower() in k.lower():
            return k
    return None

def find_all_cols(bg, substring):
    """Return all bg columns whose name contains substring."""
    return [k for k in bg if substring.lower() in k.lower()]

def interp_z0(bg, col):
    """Interpolate background column to z=0."""
    if col not in bg: return np.nan
    z = np.asarray(bg["z"], dtype=float)
    y = np.asarray(bg[col], dtype=float)
    order = np.argsort(z)
    return float(np.interp(0.0, z[order], y[order]))

def read_budget(cosmo, label):
    """
    Extract energy budget at z=0 from background table.
    Sums over all ncdm species columns automatically.
    """
    bg = cosmo.get_background()

    # Find columns by substring
    col_ur   = find_col(bg, "rho_ur")
    col_g    = find_col(bg, "rho_g")
    col_crit = find_col(bg, "rho_crit")
    cols_ncdm = find_all_cols(bg, "rho_ncdm")  # may be multiple species

    rho_ur   = interp_z0(bg, col_ur)   if col_ur   else np.nan
    rho_g    = interp_z0(bg, col_g)    if col_g    else np.nan
    rho_crit = interp_z0(bg, col_crit) if col_crit else np.nan

    # Sum all ncdm species
    rho_ncdm = 0.0
    ncdm_cols_used = []
    for c in cols_ncdm:
        v = interp_z0(bg, c)
        if np.isfinite(v):
            rho_ncdm += v
            ncdm_cols_used.append(c)
    if not ncdm_cols_used:
        rho_ncdm = np.nan

    # Derived fractions
    Omega_ur   = rho_ur   / rho_crit if np.isfinite(rho_ur)   and rho_crit > 0 else np.nan
    Omega_ncdm = rho_ncdm / rho_crit if np.isfinite(rho_ncdm) and rho_crit > 0 else np.nan
    Omega_g    = rho_g    / rho_crit if np.isfinite(rho_g)     and rho_crit > 0 else np.nan

    return {
        "label":       label,
        "rho_ur":      rho_ur,
        "rho_ncdm":    rho_ncdm,
        "rho_g":       rho_g,
        "rho_crit":    rho_crit,
        "Omega_ur":    Omega_ur,
        "Omega_ncdm":  Omega_ncdm,
        "Omega_g":     Omega_g,
        "ncdm_cols":   ncdm_cols_used,
        "col_ur":      col_ur,
    }

def print_budget(b, deg, N_ur_intended):
    print(f"\n  {b['label']}  (deg={deg}, N_ur_intended={N_ur_intended:.3f})")
    print(f"    ncdm columns found: {b['ncdm_cols']}")
    print(f"    ur column found:    {b['col_ur']}")
    print(f"    rho_ur   (z=0) = {b['rho_ur']:.4e}")
    print(f"    rho_ncdm (z=0) = {b['rho_ncdm']:.4e}")
    print(f"    rho_g    (z=0) = {b['rho_g']:.4e}")
    print(f"    rho_crit (z=0) = {b['rho_crit']:.4e}")
    print(f"    Omega_ur        = {b['Omega_ur']:.6f}")
    print(f"    Omega_ncdm      = {b['Omega_ncdm']:.6f}")
    print(f"    Omega_g         = {b['Omega_g']:.6f}")
    # Flag if N_ur~0 but rho_ur is large
    if N_ur_intended < 0.01 and np.isfinite(b['rho_ur']) and b['rho_ur'] > 1e-15:
        print(f"    *** WARNING: N_ur~0 but rho_ur is non-negligible! Budget leak. ***")

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
    print("\n" + "█"*65)
    print("  Degeneracy budget diagnostic")
    print(f"  G_eff={G_EFF_ON:.1e}, m_ncdm={M_NCDM_EV} eV")
    print(f"  Testing deg = {DEG_VALUES}")
    print("█"*65)

    # Storage for P(k) comparison
    pk_store = {}   # (build, deg) -> pk array
    budget_store = {}

    for deg in DEG_VALUES:
        N_ur = max(0.0, 3.046 - deg)
        print(f"\n{'─'*65}")
        print(f"  deg = {deg}   N_ur = {N_ur:.3f}")
        print(f"{'─'*65}")

        pT = params_thomas(deg)
        pM = params_mine(deg)

        cT, pT_final = run_with_autofix(ClassThomas, pT, f"Thomas deg={deg}")
        cM, pM_final = run_with_autofix(ClassMine,   pM, f"Mine   deg={deg}")

        # Print final neutrino-related params for each build
        ncdm_keys = [k for k in pT_final if any(x in k for x in
                     ["ncdm","N_ur","deg","G_eff","m_ncdm","T_ncdm"])]
        print(f"\n  Thomas final ncdm params: "
              + ", ".join(f"{k}={pT_final[k]}" for k in sorted(ncdm_keys)))
        ncdm_keys = [k for k in pM_final if any(x in k for x in
                     ["ncdm","N_ur","deg","G_eff","m_ncdm","T_ncdm"])]
        print(f"  Mine   final ncdm params: "
              + ", ".join(f"{k}={pM_final[k]}" for k in sorted(ncdm_keys)))

        # Budget readout
        bT = read_budget(cT, f"Thomas deg={deg}")
        bM = read_budget(cM, f"Mine   deg={deg}")
        print("\n  === Energy budget at z=0 ===")
        print_budget(bT, deg, N_ur)
        print_budget(bM, deg, N_ur)

        # Cross-build comparison
        print(f"\n  === Cross-build comparison (Mine/Thomas - 1) ===")
        for qty in ["rho_ur", "rho_ncdm", "rho_g", "rho_crit", "Omega_ur", "Omega_ncdm"]:
            vT = bT[qty]; vM = bM[qty]
            if np.isfinite(vT) and np.isfinite(vM) and vT != 0:
                rel = (vM - vT) / abs(vT)
                flag = "  ← DIFF" if abs(rel) > 1e-3 else ""
                print(f"    {qty:<14s}  T={vT:.4e}  M={vM:.4e}  rel={rel:+.3e}{flag}")

        budget_store[(deg, "T")] = bT
        budget_store[(deg, "M")] = bM

        # P(k)
        pk_store[("T", deg)] = get_pk(cT)
        pk_store[("M", deg)] = get_pk(cM)

        cT.struct_cleanup(); cT.empty()
        cM.struct_cleanup(); cM.empty()

    # ── Summary: rho_ncdm scaling with deg ───────────────────────
    print(f"\n{'═'*65}")
    print("  SCALING CHECK: does rho_ncdm scale linearly with deg?")
    print(f"{'═'*65}")
    for build, label in [("T", "Thomas"), ("M", "Mine")]:
        r1 = budget_store[(DEG_VALUES[0], build)]["rho_ncdm"]
        r2 = budget_store[(DEG_VALUES[1], build)]["rho_ncdm"]
        expected_ratio = DEG_VALUES[1] / DEG_VALUES[0]
        actual_ratio   = r2 / r1 if r1 > 0 else np.nan
        print(f"  {label}: rho_ncdm(deg={DEG_VALUES[1]})/rho_ncdm(deg={DEG_VALUES[0]}) = "
              f"{actual_ratio:.4f}  (expected ~{expected_ratio:.4f} if linear)")

    # ── P(k) plots ───────────────────────────────────────────────
    h_ref = 0.6736
    k_h   = K_LIST / h_ref

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(f"Degeneracy diagnostic  G_eff={G_EFF_ON:.1e}", fontsize=11)

    for col, deg in enumerate(DEG_VALUES):
        pkT = pk_store[("T", deg)] * h_ref**3
        pkM = pk_store[("M", deg)] * h_ref**3
        ratio = pkM / pkT

        axes[0, col].loglog(k_h, pkT, label="Thomas")
        axes[0, col].loglog(k_h, pkM, ls="--", label="Mine")
        axes[0, col].set_ylabel(r"$P(k)$  $[(\mathrm{Mpc}/h)^3]$")
        axes[0, col].set_title(f"deg = {deg}")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, which="both", ls=":")

        axes[1, col].semilogx(k_h, ratio)
        axes[1, col].axhline(1, ls=":", c="k")
        axes[1, col].set_ylabel("Mine / Thomas")
        axes[1, col].set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
        axes[1, col].grid(True, which="both", ls=":")
        print(f"\n  P(k) max|ratio-1| at deg={deg}: "
              f"{np.max(np.abs(ratio-1)):.3e}  "
              f"(linear k<0.2: {np.max(np.abs(ratio[k_h<0.2]-1)):.3e})")

    plt.tight_layout()
    plt.savefig("deg_budget_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()