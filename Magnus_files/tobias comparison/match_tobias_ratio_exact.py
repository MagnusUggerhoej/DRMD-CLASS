#!/usr/bin/env python3
"""
match_tobias_ratio_exact.py  (WORKING: no lensing required)

Purpose:
  Compare Tobias ratio from mPk_tobias.txt to several denominator definitions in YOUR build.

Fix vs your error:
  We do NOT set lensing=yes, because output=mPk only.

Run:
  conda activate CLASS
  python match_tobias_ratio_exact.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from classy_NEDE import Class

here = os.path.dirname(os.path.abspath(__file__))


def load_tobias_file(fname="mPk_tobias.txt"):
    path = os.path.join(here, fname)
    data = np.loadtxt(path)
    if data.shape[1] < 3:
        raise ValueError(f"{fname} must have >=3 columns: k, Pk_CLASS++, Pk_reference")
    k = data[:, 0]
    pk_num = data[:, 1]
    pk_den = data[:, 2]
    return k, pk_num, pk_den


def run_pk(params, kgrid, z, which="pk"):
    """
    which: "pk" or "pk_cb"
    Falls back from pk_cb -> pk if pk_cb isn't available for that model.
    """
    # HARD SAFETY: ensure lensing isn't accidentally on when output=mPk
    params = dict(params)  # copy
    if "lensing" in params:
        # force it off unless user explicitly wants to pay for CMB spectra
        params["lensing"] = "no"

    c = Class()
    c.set(params)
    c.compute()

    out = []
    for kk in kgrid:
        if which == "pk_cb":
            try:
                val = c.pk_cb(float(kk), float(z))
            except Exception:
                val = c.pk(float(kk), float(z))
        else:
            val = c.pk(float(kk), float(z))
        out.append(val)

    c.struct_cleanup()
    c.empty()
    return np.array(out, dtype=float)


def score_ratio(r_you, r_tob, kgrid):
    diff = r_you - r_tob
    rms = np.sqrt(np.mean(diff**2))
    imax = int(np.argmax(np.abs(diff)))
    return {
        "rms": float(rms),
        "maxabs": float(np.max(np.abs(diff))),
        "worst_k": float(kgrid[imax]),
        "tob_worst": float(r_tob[imax]),
        "you_worst": float(r_you[imax]),
    }


def report(name, r_you, r_tob, kgrid):
    s = score_ratio(r_you, r_tob, kgrid)
    print(f"\n=== {name} ===")
    print("RMS difference     = %.6e" % s["rms"])
    print("Max abs difference = %.6e" % s["maxabs"])
    print("Worst k            = %.6e 1/Mpc" % s["worst_k"])
    print("Tobias ratio       = %.6e" % s["tob_worst"])
    print("Your ratio         = %.6e" % s["you_worst"])
    return s


def main():
    # 1) Load Tobias
    k_tob, pk_tob_num, pk_tob_den = load_tobias_file("mPk_tobias.txt")
    ratio_tob = pk_tob_num / pk_tob_den

    print("Loaded mPk_tobias.txt")
    print("At smallest k: ratio = %.6f" % ratio_tob[0])
    print("At largest  k: ratio = %.6f" % ratio_tob[-1])
    print("")

    z = 3.0

    # 2) Base cosmology (Tobias-like, but P(k) only)
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

        # keep the verbosity knobs (harmless)
        "perturbations_verbose": "0",
        "thermodynamics_verbose": "0",

        # IMPORTANT: do NOT set lensing=yes when output=mPk
        "lensing": "no",
    }

    # 3) DRMD block
    drmd = {
        "G_over_aH_drmd_ini": 1e-3,
        "delta_Neff_drmd": 0.1,
        "f_idm_drmd": 0.1,
        "z_stop": 5000,
    }

    # 4) ncdm block (explicit T_ncdm!)
    ncdm_block = {
        "N_ncdm": 1,
        "deg_ncdm": "3.0",
        "m_ncdm": "1e-2",
        "T_ncdm": "0.71611",
        "ncdm_fluid_approximation": 3,
    }

    # -------------------------
    # Models
    # -------------------------
    # Numerator: DRMD + ncdm + Geff
    num = dict(base)
    num.update(drmd)
    num.update(ncdm_block)
    num["G_eff_ncdm_species"] = "1e-3"

    # DenB: Tobias-like baseline (no DRMD, no ncdm)
    denB = dict(base)

    # DenA: DRMD baseline (with DRMD, no ncdm)
    denA = dict(base)
    denA.update(drmd)

    # DenC: DRMD + ncdm with Geff=0   (the key “on/off” definition)
    denC = dict(base)
    denC.update(drmd)
    denC.update(ncdm_block)
    denC["G_eff_ncdm_species"] = "0.0"

    # -------------------------
    # Compute P(k)
    # -------------------------
    print("Computing numerator once (DRMD + ncdm + Geff)...")
    pk_num_tot = run_pk(num, k_tob, z, which="pk")
    pk_num_cb = run_pk(num, k_tob, z, which="pk_cb")

    print("Computing denominator B: no DRMD, no ncdm...")
    pk_denB_tot = run_pk(denB, k_tob, z, which="pk")

    print("Computing denominator A: DRMD, no ncdm...")
    pk_denA_tot = run_pk(denA, k_tob, z, which="pk")

    print("Computing denominator C (NEW): DRMD + ncdm, Geff=0 ...")
    pk_denC_tot = run_pk(denC, k_tob, z, which="pk")
    pk_denC_cb = run_pk(denC, k_tob, z, which="pk_cb")

    # -------------------------
    # Ratios
    # -------------------------
    ratio_you_B_tot = pk_num_tot / pk_denB_tot
    ratio_you_A_tot = pk_num_tot / pk_denA_tot
    ratio_you_C_tot = pk_num_tot / pk_denC_tot

    ratio_you_B_cbnum = pk_num_cb / pk_denB_tot
    ratio_you_A_cbnum = pk_num_cb / pk_denA_tot
    ratio_you_C_cb = pk_num_cb / pk_denC_cb  # cb/cb

    # -------------------------
    # Scores
    # -------------------------
    report("Ratio B (Num / (no-DRMD,no-ncdm)) using pk()", ratio_you_B_tot, ratio_tob, k_tob)
    report("Ratio A (Num / (DRMD,no-ncdm)) using pk()", ratio_you_A_tot, ratio_tob, k_tob)
    report("Ratio C (NEW: (DRMD+ncdm+Geff)/(DRMD+ncdm Geff=0)) using pk()", ratio_you_C_tot, ratio_tob, k_tob)

    report("Ratio B using pk_cb() numerator (cb / total-denB)", ratio_you_B_cbnum, ratio_tob, k_tob)
    report("Ratio A using pk_cb() numerator (cb / total-denA)", ratio_you_A_cbnum, ratio_tob, k_tob)
    report("Ratio C (NEW cb/cb): pk_cb on/off", ratio_you_C_cb, ratio_tob, k_tob)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(9, 6))
    plt.semilogx(k_tob, ratio_tob, "k--", lw=3, label="Tobias file ratio")

    plt.semilogx(k_tob, ratio_you_B_tot, lw=2, label="You: Num/(no-DRMD,no-ncdm) [pk]")
    plt.semilogx(k_tob, ratio_you_A_tot, lw=2, label="You: Num/(DRMD,no-ncdm) [pk]")
    plt.semilogx(k_tob, ratio_you_C_tot, lw=2, label="You NEW: (DRMD+ncdm+Geff)/(DRMD+ncdm Geff=0) [pk]")

    plt.semilogx(k_tob, ratio_you_C_cb, lw=2, label="You NEW cb/cb: pk_cb on/off")

    plt.axhline(1.0, color="k", lw=1)
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("P_num / P_den")
    plt.title(f"Denominator-definition test at z={z}")
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    print("\nIf Ratio C is the closest to Tobias, then Tobias' 'Pk_reference' is very likely")
    print("a *same-DRMD+same-ncdm but Geff=0* reference (an on/off definition).")


if __name__ == "__main__":
    main()
