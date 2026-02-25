#!/usr/bin/env python3
"""
compare_pk_interacting_ncdm_apples_to_apples.py

Purpose
-------
Compare Thomas (CLASSpp wrapper: classy) vs Mine (modified NEDE/DRMD-CLASS wrapper: classy_NEDE)
in a controlled "apples-to-apples" setup where ALL massive ncdm species are interacting.

We compute the linear matter power spectrum P(k,z=0) on the same physical k [1/Mpc] grid,
convert to the same (Mpc/h)^3 convention using a common reference h, and plot:
  - P(k) overlay
  - ratio Mine/Thomas
We also print max |ratio-1|.

Key design
----------
#1 Define base ΛCDM parameters
#2 Define Thomas interacting-ncdm parameters (known working keys)
#3 Define Mine interacting-ncdm parameters (split ncdm + known working keys)
#4 Compute both
#5 Plot + compare
"""

import numpy as np
import matplotlib.pyplot as plt

from classy_tobias import Class as ClassThomas
from classy_NEDE import Class as ClassMine


# ============================================================
# 0) Knobs
# ============================================================
L_MAX = 2500

# "Clearly interacting" but not absurd
G_EFF_NCDM_ON = 1e1   # in same units used by Thomas's branch (you already validated this works)
G_EFF_UR_ON   = 0.0    # keep UR interactions off here unless you explicitly want UR interacting too

# Massive neutrino content (one effective species)
N_NCDM_INT = 1
M_NCDM_EV  = 0.06
DEG_NCDM   = 1.0   # effective degeneracy carried by the massive species

# P(k) probe grid (physical k in 1/Mpc)
K_1MPC = np.logspace(-4, 1.0, 250)
Z_PK   = 0.0


# ============================================================
# 1) Base ΛCDM parameters (shared)
# ============================================================
def base_params(lmax: int) -> dict:
    return {
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "A_s": 2.1e-9,
        "n_s": 0.9649,
        "tau_reio": 0.0543,

        # need mPk since we call pk()
        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": lmax,

        # make sure pk tables cover our range
        "P_k_max_1/Mpc": float(np.max(K_1MPC) * 1.05),

        "T_cmb": 2.7255,
        "Omega_k": 0.0,

        # consistent recombination in both (safe default)
        "recombination": "recfast",
        "reio_parametrization": "reio_camb",
    }


def pk_safe(cosmo, k_1Mpc: float, z: float) -> float:
    """Prefer pk_lin if available, otherwise pk."""
    if hasattr(cosmo, "pk_lin"):
        return cosmo.pk_lin(k_1Mpc, z)
    return cosmo.pk(k_1Mpc, z)


def run(Cls, params: dict, label: str):
    print(f"Running {label}...")
    c = Cls()
    c.set(params)
    c.compute()
    return c


# ============================================================
# 2) Thomas parameters: ALL ncdm interacting (known working keys)
# ============================================================
def params_thomas_all_interacting(lmax: int) -> dict:
    p = base_params(lmax)

    # Radiation budget: keep closure consistent when adding massive species.
    # If you want "fixed total Neff-like budget": reduce N_ur by DEG_NCDM.
    p["N_ur"] = max(0.0, 3.046 - float(DEG_NCDM))

    # UR interactions (optional)
    if G_EFF_UR_ON > 0.0:
        p["G_eff_ur"] = float(G_EFF_UR_ON)
    else:
        p["G_eff_ur"] = 0.0

    # Thomas/CLASSpp interacting-ncdm interface:
    p.update({
        "N_ncdm_interacting": int(N_NCDM_INT),
        "m_ncdm_interacting": float(M_NCDM_EV),
        "deg_ncdm_interacting": float(DEG_NCDM),
        "G_eff_ncdm_interacting": float(G_EFF_NCDM_ON),
    })
    return p


# ============================================================
# 3) Mine parameters: split ncdm, ALL interacting (known working keys)
# ============================================================
def params_mine_all_interacting(lmax: int) -> dict:
    p = base_params(lmax)

    # Same radiation budget convention
    p["N_ur"] = max(0.0, 3.046 - float(DEG_NCDM))

    # UR interactions (optional)
    if G_EFF_UR_ON > 0.0:
        p["G_eff_ur"] = float(G_EFF_UR_ON)
    else:
        p["G_eff_ur"] = 0.0

    # Use your split keys: DO NOT set legacy N_ncdm when split keys are present
    p.update({
        "N_ncdm_standard": 0,
        "N_ncdm_interacting": int(N_NCDM_INT),

        # For your branch, masses are typically still read via legacy m_ncdm list
        # (length = Ntot). Here Ntot=1.
        "m_ncdm": float(M_NCDM_EV),

        # Coupling for interacting subset (you validated this key works)
        "G_eff_ncdm_interacting": float(G_EFF_NCDM_ON),

        # If your code uses deg list for ncdm internally, you can also set:
        # "deg_ncdm": float(DEG_NCDM),
        # but only do this if your input parser supports it in this branch.
    })
    return p


# ============================================================
# 4) Compute + compare
# ============================================================
def main():
    # ---- define parameters
    pT = params_thomas_all_interacting(L_MAX)
    pM = params_mine_all_interacting(L_MAX)

    # ---- run
    cosmo_T = run(ClassThomas, pT, "Thomas (CLASSpp) : interacting ncdm")
    cosmo_M = run(ClassMine,   pM, "Mine (split ncdm) : interacting ncdm")

    # ---- compute P(k) on common physical k [1/Mpc]
    Pk_T_1Mpc = np.array([pk_safe(cosmo_T, k, Z_PK) for k in K_1MPC])
    Pk_M_1Mpc = np.array([pk_safe(cosmo_M, k, Z_PK) for k in K_1MPC])

    # ---- convert to common k[h/Mpc] + P(k)[(Mpc/h)^3] using Thomas h as reference
    h_ref = float(cosmo_T.h())
    k_hMpc = K_1MPC / h_ref
    Pk_T = Pk_T_1Mpc * h_ref**3
    Pk_M = Pk_M_1Mpc * h_ref**3

    # ---- plot overlay
    plt.figure(figsize=(7, 5))
    plt.loglog(k_hMpc, Pk_T, label="Thomas")
    plt.loglog(k_hMpc, Pk_M, "--", label="Mine")
    plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
    plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
    plt.title(r"$P(k)$ – LCDM + interacting ncdm (linear, $z=0$)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    # ---- ratio
    ratio = Pk_M / Pk_T
    plt.figure(figsize=(7, 3.8))
    plt.semilogx(k_hMpc, ratio)
    plt.axhline(1.0, ls=":")
    plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
    plt.ylabel("Mine / Thomas")
    plt.title(r"$P(k)$ ratio (Mine/Thomas)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    # ---- prints
    print("h(Thomas) =", cosmo_T.h(), " h(Mine) =", cosmo_M.h())
    print("max |ratio-1| =", np.max(np.abs(ratio - 1.0)))

    # ---- cleanup
    cosmo_T.struct_cleanup(); cosmo_T.empty()
    cosmo_M.struct_cleanup(); cosmo_M.empty()


if __name__ == "__main__":
    main()