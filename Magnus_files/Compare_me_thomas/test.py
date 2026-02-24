#!/usr/bin/env python3
"""
Degeneracy budget check (2-minute settle-it test)
=================================================
Purpose:
  For Thomas (CLASSpp/classy) and Mine (classy_NEDE), run the SAME cosmology twice:
    - deg = 1.0
    - deg = 3.046
  with ALL ncdm interacting and with N_ur forced to N_ur = max(0, 3.046 - deg).

Then print, for each backend and each degeneracy:
  - rho_ur(z~0) and rho_ncdm(z~0) from get_background()
  - Omega0_ur, Omega0_ncdm_tot, Neff (if exposed) from get_current_derived_parameters()
  - the intended N_ur value used in the input dict

This settles whether the mismatch is due to inconsistent degeneracy / radiation-budget mapping.
"""

import numpy as np
import matplotlib.pyplot as plt

from classy      import Class as ClassThomas
from classy_NEDE import Class as ClassMine


# ============================================================
# User knobs
# ============================================================
DEG_LIST   = [1.0, 3.046]
M_NCDM_EV  = 0.06
G_EFF_ON   = 1e-1   # use a clearly-nonzero interacting value (same in both)
PK_Z       = 0.0
K_1MPC     = np.logspace(-4, 1.0, 250)  # physical k [1/Mpc]
PK_MAX_1MPC = 10.0

PLOT = True


# ============================================================
# Helpers
# ============================================================
def pk_safe(cosmo, k, z):
    """Prefer linear pk if wrapper exposes it."""
    if hasattr(cosmo, "pk_lin"):
        return cosmo.pk_lin(k, z)
    return cosmo.pk(k, z)

def base_params():
    """Common LCDM baseline used for BOTH backends."""
    return {
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "A_s": 2.1e-9,
        "n_s": 0.9649,
        "tau_reio": 0.0543,

        # We only need P(k) + background for this diagnostic
        "output": "mPk",
        "lensing": "no",
        "P_k_max_1/Mpc": PK_MAX_1MPC,

        "T_cmb": 2.7255,
        "Omega_k": 0.0,

        # keep recombination simple/robust across forks
        "recombination": "recfast",
    }

def run(ClassObj, params, label):
    print(f"\nRunning {label}...")
    c = ClassObj()
    c.set(params)
    c.compute()
    return c

def dump_budget(cosmo, label, intended_Nur):
    print("\n" + "-"*80)
    print(f"[BUDGET] {label}")
    print(f"  intended N_ur = {intended_Nur}")

    bg = cosmo.get_background()
    z = np.asarray(bg["z"])
    i0 = int(np.argmin(np.abs(z - 0.0)))

    def pick(colnames):
        for c in colnames:
            if c in bg:
                return c
        return None

    col_rho_g    = pick(["rho_g", "rho_gamma"])
    col_rho_ur   = pick(["rho_ur", "rho_ur+rho_dr", "rho_rel", "rho_r"])
    col_rho_ncdm = pick(["rho_ncdm", "rho_ncdm_tot", "rho_ncdm[0]"])

    print(f"  z0 used       = {z[i0]:.8e}")

    if col_rho_g:
        print(f"  {col_rho_g:14s} = {bg[col_rho_g][i0]: .10e}")
    else:
        print("  rho_g          = <MISSING>")

    if col_rho_ur:
        print(f"  {col_rho_ur:14s} = {bg[col_rho_ur][i0]: .10e}")
    else:
        print("  rho_ur         = <MISSING>")

    if col_rho_ncdm:
        print(f"  {col_rho_ncdm:14s} = {bg[col_rho_ncdm][i0]: .10e}")
    else:
        # try summing per-species columns if present
        ncdm_cols = [k for k in bg.keys() if k.startswith("rho_ncdm") and k not in ["rho_ncdm", "rho_ncdm_tot"]]
        if ncdm_cols:
            rho_sum = 0.0
            for k in ncdm_cols:
                rho_sum += float(bg[k][i0])
            print(f"  rho_ncdm(sum)  = {rho_sum: .10e}   (from {len(ncdm_cols)} cols)")
            print(f"    example cols: {ncdm_cols[:6]}{'...' if len(ncdm_cols)>6 else ''}")
        else:
            print("  rho_ncdm       = <MISSING>")

    # Derived parameters (names vary)
    candidates = [
        "Neff", "N_eff",
        "Omega0_ur", "Omega_ur", "Omega_r",
        "Omega0_ncdm_tot", "Omega_ncdm",
        "Omega_m", "Omega_Lambda",
        "h", "H0",
    ]
    try:
        dp = cosmo.get_current_derived_parameters(candidates)
    except Exception as e:
        dp = {}
        print(f"\n  [DERIVED] FAILED to query derived parameters: {e}")

    print("\n  [DERIVED]")
    for k in candidates:
        if k in dp:
            print(f"  {k:16s} = {dp[k]}")
        else:
            print(f"  {k:16s} = <MISSING>")

    print("-"*80)

def build_thomas_params(deg):
    """
    Thomas side (CLASSpp) interacting ncdm interface.
    IMPORTANT: do NOT assume log10 keys exist; use linear G_eff_ncdm_interacting.
    """
    p = base_params()

    # Force intended radiation budget
    Nur = max(0.0, 3.046 - float(deg))
    p["N_ur"] = Nur

    # All ncdm interacting (Thomas expects these split keys)
    p.update({
        "N_ncdm_interacting": 1,
        "m_ncdm_interacting": M_NCDM_EV,
        "deg_ncdm_interacting": float(deg),
        "G_eff_ncdm_interacting": G_EFF_ON,
    })

    return p, Nur

def build_mine_params(deg):
    """
    Mine side (DRMD-CLASS / classy_NEDE) using your split + per-species interaction logic.
    IMPORTANT: do NOT set legacy N_ncdm when using split keys.
    """
    p = base_params()

    # Force intended radiation budget
    Nur = max(0.0, 3.046 - float(deg))
    p["N_ur"] = Nur

    # All ncdm interacting via split keys
    p.update({
        "N_ncdm_standard": 0,
        "N_ncdm_interacting": 1,
        "m_ncdm_interacting": M_NCDM_EV,
        "deg_ncdm_interacting": float(deg),

        # Use the key you already verified works
        "G_eff_ncdm_interacting": G_EFF_ON,
    })

    return p, Nur

def compute_pk_in_common_units(cosmo_T, cosmo_M):
    """
    Evaluate P(k,z=0) on common physical k [1/Mpc], then convert to (Mpc/h)^3
    using Thomas h as the common reference (same trick as your earlier scripts).
    """
    Pk_T_1Mpc = np.array([pk_safe(cosmo_T, k, PK_Z) for k in K_1MPC])
    Pk_M_1Mpc = np.array([pk_safe(cosmo_M, k, PK_Z) for k in K_1MPC])

    h_ref = float(cosmo_T.h())
    k_hMpc = K_1MPC / h_ref
    Pk_T = Pk_T_1Mpc * h_ref**3
    Pk_M = Pk_M_1Mpc * h_ref**3
    return k_hMpc, Pk_T, Pk_M, h_ref

def main():
    print("\n=== Degeneracy budget check ===")
    print(f"DEG_LIST   = {DEG_LIST}")
    print(f"M_NCDM_EV  = {M_NCDM_EV} eV")
    print(f"G_EFF_ON   = {G_EFF_ON}")
    print("Intended: N_ur = max(0, 3.046 - deg)\n")

    store = {}

    for deg in DEG_LIST:
        # --- build params ---
        pT, NurT = build_thomas_params(deg)
        pM, NurM = build_mine_params(deg)
        assert abs(NurT - NurM) < 1e-15

        # --- run ---
        cosmo_T = run(ClassThomas, pT, f"Thomas (deg={deg})")
        dump_budget(cosmo_T, f"Thomas (deg={deg})", intended_Nur=NurT)

        cosmo_M = run(ClassMine,   pM, f"Mine   (deg={deg})")
        dump_budget(cosmo_M, f"Mine   (deg={deg})", intended_Nur=NurM)

        # --- pk compare ---
        k_hMpc, Pk_T, Pk_M, h_ref = compute_pk_in_common_units(cosmo_T, cosmo_M)
        ratio = Pk_M / Pk_T
        max_dev = float(np.max(np.abs(ratio - 1.0)))
        print(f"\n  [deg={deg}] h_ref(Thomas) = {h_ref:.8g}")
        print(f"  [deg={deg}] max |Mine/Thomas - 1| = {max_dev:.3e}")

        store[deg] = (k_hMpc, Pk_T, Pk_M, ratio)

        # cleanup
        cosmo_T.struct_cleanup(); cosmo_T.empty()
        cosmo_M.struct_cleanup(); cosmo_M.empty()

        # plots for this deg
        if PLOT:
            plt.figure(figsize=(7.5, 5.2))
            plt.loglog(k_hMpc, Pk_T, label="Thomas")
            plt.loglog(k_hMpc, Pk_M, "--", label="Mine")
            plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
            plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
            plt.title(f"Degeneracy test (all ncdm interacting)\nP(k) overlay (deg={deg})")
            plt.grid(True, which="both", ls=":")
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7.5, 4.3))
            plt.semilogx(k_hMpc, ratio)
            plt.axhline(1.0, ls=":")
            plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
            plt.ylabel("Mine / Thomas")
            plt.title(f"Degeneracy test (all ncdm interacting)\nP(k) ratio Mine/Thomas (deg={deg})")
            plt.grid(True, which="both", ls=":")
            plt.tight_layout()
            plt.show()

    # Optional: show degeneracy response inside each backend (deg=3.046 vs 1.0)
    if len(DEG_LIST) >= 2 and (1.0 in store) and (3.046 in store):
        k1, PkT1, PkM1, _ = store[1.0]
        k3, PkT3, PkM3, _ = store[3.046]

        # same k grid by construction
        plt.figure(figsize=(7.5, 4.3))
        plt.semilogx(K_1MPC, PkT3 / PkT1)
        plt.axhline(1.0, ls=":")
        plt.xlabel(r"$k\,[1/\mathrm{Mpc}]$")
        plt.ylabel(r"$P_k(\mathrm{deg}=3.046) / P_k(\mathrm{deg}=1.0)$")
        plt.title("Degeneracy test (all ncdm interacting)\nDegeneracy response in Thomas")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7.5, 4.3))
        plt.semilogx(K_1MPC, PkM3 / PkM1)
        plt.axhline(1.0, ls=":")
        plt.xlabel(r"$k\,[1/\mathrm{Mpc}]$")
        plt.ylabel(r"$P_k(\mathrm{deg}=3.046) / P_k(\mathrm{deg}=1.0)$")
        plt.title("Degeneracy test (all ncdm interacting)\nDegeneracy response in Mine")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.show()

    print("\n=== Done ===")
    print("Interpretation tip:")
    print("  - For deg=3.046, intended N_ur=0 => rho_ur(z=0) should be ~0.")
    print("  - If rho_ncdm(z=0) and Omega0_ncdm_tot scale differently between backends when deg changes,")
    print("    then the degeneracy key is being interpreted differently (or applied to different species).")

if __name__ == "__main__":
    main()