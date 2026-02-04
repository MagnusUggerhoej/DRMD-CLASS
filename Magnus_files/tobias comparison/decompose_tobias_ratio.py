#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from classy_NEDE import Class

HERE = os.path.dirname(os.path.abspath(__file__))

def load_tobias_ratio():
    path = os.path.join(HERE, "mPk_tobias.txt")
    data = np.loadtxt(path)
    k = data[:,0]
    ratio = data[:,1] / data[:,2]
    return k, ratio

def run_pk(params, kgrid, z):
    M = Class()
    M.set(params)
    M.compute()
    pk = np.array([M.pk(float(k), float(z)) for k in kgrid])
    M.struct_cleanup()
    M.empty()
    return pk

def main():
    z = 3.0
    k, ratio_tob = load_tobias_ratio()

    # ===============================
    # BASE COSMOLOGY (NO CMB OUTPUT)
    # ===============================
    base = {
        "output": "mPk",
        "z_max_pk": 4.0,
        "H0": 67.5,
        "N_ur": 2.0328,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "tau_reio": 0.054,
        "ln10^{10}A_s": 3.0,
        "n_s": 0.965,
    }

    # Force exact hierarchy like Tobias
    base_exact = dict(base)
    base_exact["ncdm_fluid_approximation"] = 3

    # ===============================
    # DRMD PARAMETERS (Tobias)
    # ===============================
    drmd = {
        "G_over_aH_drmd_ini": 1e-3,
        "delta_Neff_drmd": 0.1,
        "f_idm_drmd": 0.1,
        "z_stop": 5000,
    }

    # ===============================
    # NCDM SETTINGS (mapped to your fork)
    # ===============================
    ncdm_present = {
        "N_ncdm": 1,
        "deg_ncdm": "3.0",
        "m_ncdm": "1e-2",
    }

    # ===============================
    # DEFINE THE FOUR MODELS
    # ===============================

    # M0: Plain LCDM (no DRMD, no ncdm)
    M0 = dict(base_exact)
    M0["N_ncdm"] = 0

    # M1: DRMD only
    M1 = dict(base_exact)
    M1.update(drmd)
    M1["N_ncdm"] = 0

    # M2: DRMD + neutrinos present but Geff = 0 (collisionless)
    M2 = dict(base_exact)
    M2.update(drmd)
    M2.update(ncdm_present)
    M2["G_eff_ncdm_species"] = "0.0"

    # M3: DRMD + neutrinos + interaction
    M3 = dict(base_exact)
    M3.update(drmd)
    M3.update(ncdm_present)
    M3["G_eff_ncdm_species"] = "1e-3"

    # ===============================
    # RUN MODELS
    # ===============================
    print("Computing M0 (baseline LCDM)...")
    pk0 = run_pk(M0, k, z)

    print("Computing M1 (DRMD only)...")
    pk1 = run_pk(M1, k, z)

    print("Computing M2 (DRMD + ncdm, Geff=0)...")
    pk2 = run_pk(M2, k, z)

    print("Computing M3 (DRMD + ncdm + interaction)...")
    pk3 = run_pk(M3, k, z)

    # ===============================
    # RATIOS
    # ===============================
    ratio_total_like_tobias = pk3 / pk0
    ratio_drmd_only         = pk1 / pk0
    ratio_add_ncdm          = pk2 / pk1
    ratio_pure_interaction  = pk3 / pk2

    # ===============================
    # PLOT
    # ===============================
    plt.figure(figsize=(9,5))
    plt.semilogx(k, ratio_tob, "k--", lw=2.5, label="Tobias file ratio")
    plt.semilogx(k, ratio_total_like_tobias, lw=2, label="Your total (DRMD+ncdm+Geff) / baseline")
    plt.semilogx(k, ratio_drmd_only, lw=2, label="DRMD only")
    plt.semilogx(k, ratio_add_ncdm, lw=2, label="Add ncdm (Geff=0)")
    plt.semilogx(k, ratio_pure_interaction, lw=2, label="Pure interaction (Geff on/off)")
    plt.axhline(1.0, color="k", lw=1)
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("ratio")
    plt.title(f"Decomposition of Tobias suppression at z={z}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===============================
    # PRINT SUMMARY
    # ===============================
    print("\n=== Ratios at largest k ===")
    print(f"Tobias ratio (file)            = {ratio_tob[-1]:.6f}")
    print(f"Your total-like-Tobias         = {ratio_total_like_tobias[-1]:.6f}")
    print(f"DRMD-only                      = {ratio_drmd_only[-1]:.6f}")
    print(f"Add-ncdm (Geff=0)              = {ratio_add_ncdm[-1]:.6f}")
    print(f"Pure interaction effect        = {ratio_pure_interaction[-1]:.6f}")

if __name__ == "__main__":
    main()
