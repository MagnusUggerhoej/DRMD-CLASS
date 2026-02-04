#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from classy_NEDE import Class

HERE = os.path.dirname(os.path.abspath(__file__))

def load_tobias():
    data = np.loadtxt(os.path.join(HERE, "mPk_tobias.txt"))
    k = data[:, 0]
    pk_classpp = data[:, 1]   # header says: Pk_CLASS++
    pk_ref = data[:, 2]       # header says: Pk_reference
    return k, pk_classpp / pk_ref

def run_pk(params, kgrid, z):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    pk = np.array([cosmo.pk(float(k), float(z)) for k in kgrid])
    cosmo.struct_cleanup()
    cosmo.empty()
    return pk

def fnu_at_z(params, z):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    bg = cosmo.get_background()
    i = int(np.argmin(np.abs(bg["z"] - z)))

    rho_crit = bg["(.)rho_crit"][i]
    rho_b    = bg["(.)rho_b"][i]
    rho_cdm  = bg["(.)rho_cdm"][i]

    rho_ncdm = 0.0
    n = 0
    while f"(.)rho_ncdm[{n}]" in bg:
        rho_ncdm += bg[f"(.)rho_ncdm[{n}]"][i]
        n += 1

    Omega_m = (rho_b + rho_cdm + rho_ncdm) / rho_crit
    Omega_ncdm = rho_ncdm / rho_crit
    fnu = Omega_ncdm / Omega_m

    cosmo.struct_cleanup()
    cosmo.empty()
    return float(fnu), float(Omega_ncdm), float(Omega_m)

def main():
    z = 3.0
    k, ratio_tob = load_tobias()

    base = {
        "output": "mPk",
        "z_max_pk": 4.0,
        "H0": 67.5,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "tau_reio": 0.054,
        "ln10^{10}A_s": 3.0,
        "n_s": 0.965,
        "N_ur": 2.0328,
    }

    # ------------------------------------------------------------
    # Scenario A: "Reference = no ncdm", "Model = has ncdm"
    # ------------------------------------------------------------
    params_ref_no_ncdm = dict(base)
    params_ref_no_ncdm.update({
        "N_ncdm": 0,
    })

    params_model_with_ncdm = dict(base)
    params_model_with_ncdm.update({
        "N_ncdm": 2,
        "m_ncdm": "0.06,0.6",
        "deg_ncdm": "1.0,1.0",
        # choose whether you want "interacting" or just "present"
        "G_eff_ncdm_species": "0.0,1e-3",
    })

    fnu_model, Onu_model, Om_model = fnu_at_z(params_model_with_ncdm, z)
    print("\nScenario A (no ncdm reference):")
    print(f"  model f_nu(z={z}) = {fnu_model:.4f}, Omega_ncdm={Onu_model:.4f}, Omega_m={Om_model:.4f}")

    pk_refA = run_pk(params_ref_no_ncdm, k, z)
    pk_modA = run_pk(params_model_with_ncdm, k, z)
    ratio_A = pk_modA / pk_refA

    # ------------------------------------------------------------
    # Scenario B (your previous): same ncdm content, toggle interactions
    # ------------------------------------------------------------
    params_ref_same = dict(base)
    params_ref_same.update({
        "N_ncdm": 2,
        "m_ncdm": "0.06,0.6",
        "deg_ncdm": "1.0,1.0",
        "G_eff_ncdm_species": "0.0,0.0",
    })

    params_int_same = dict(base)
    params_int_same.update({
        "N_ncdm": 2,
        "m_ncdm": "0.06,0.6",
        "deg_ncdm": "1.0,1.0",
        "G_eff_ncdm_species": "0.0,1e-3",
    })

    pk_refB = run_pk(params_ref_same, k, z)
    pk_intB = run_pk(params_int_same, k, z)
    ratio_B = pk_intB / pk_refB

    # ------------------------------------------------------------
    # Plot all
    # ------------------------------------------------------------
    plt.figure(figsize=(9,5))
    plt.semilogx(k, ratio_tob, "k--", lw=2.5, label="Tobias: Pk_CLASS++ / Pk_reference")
    plt.semilogx(k, ratio_A, lw=2, label="Scenario A: (with ncdm+Geff) / (no ncdm)")
    plt.semilogx(k, ratio_B, lw=2, label="Scenario B: (with ncdm+Geff) / (with ncdm, Geff=0)")
    plt.axhline(1.0, color="k", lw=1)
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("ratio")
    plt.title(f"Reference-definition diagnostic at z={z}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
