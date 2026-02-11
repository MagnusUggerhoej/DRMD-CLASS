#!/usr/bin/env python3
"""
match_tobias_ratio_exact.py

Goal:
  Test the key missing hypothesis:
    Tobias ratio ~= P(DRMD + ncdm + Geff) / P(DRMD + ncdm, Geff=0)
  (and compare to other denominator choices)

What this script does:
  1) Loads Tobias file mPk_tobias.txt (k, Pk_CLASS++, Pk_reference) and forms Tobias ratio.
  2) Runs YOUR CLASS_NEDE build on the SAME k-grid at z=3.0 for a set of models:
        - Num:  DRMD + ncdm + Geff
        - DenA: DRMD baseline (no ncdm)                      [old "B" denominator]
        - DenB: Tobias-like baseline (no DRMD, no ncdm)      [old "A" denominator]
        - DenC: DRMD + ncdm with Geff=0                      [NEW: the important one]
     and forms ratios for each denominator.
  3) Tries both total-matter P(k) (pk) and cb-only P(k) (pk_cb) when available.
  4) Explicitly sets T_ncdm (default 0.71611) so we don't drift.

How to run:
  Put this script in the same folder as mPk_tobias.txt and run:
    conda activate CLASS
    python match_tobias_ratio_exact.py

Notes:
  - pk_cb() is only defined when there are massive neutrinos (N_ncdm>0). If absent, the script falls back gracefully.
  - Adjust DRMD parameters block below if you want a different DRMD setup.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from classy_NEDE import Class

here = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# I/O helpers
# -----------------------
def load_tobias_file(fname="mPk_tobias.txt"):
    path = os.path.join(here, fname)
    data = np.loadtxt(path)
    if data.shape[1] < 3:
        raise ValueError(f"{fname} must have >=3 columns: k, Pk_CLASS++, Pk_reference")
    k = data[:, 0]
    pk_num = data[:, 1]
    pk_den = data[:, 2]
    return k, pk_num, pk_den

# -----------------------
# CLASS wrapper
# -----------------------
def run_pk(params, kgrid, z, which="pk"):
    """
    which: "pk" or "pk_cb"
    Returns numpy array P(k,z) evaluated on kgrid.
    Falls back from pk_cb->pk if pk_cb not available.
    """
    c = Class()
    c.set(params)
    c.compute()

    out = []
    for kk in kgrid:
        if which == "pk_cb":
            try:
                val = c.pk_cb(float(kk), float(z))
            except Exception:
                # If not computed (e.g. no massive neutrinos), fall back to total pk
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

# -----------------------
# Main
# -----------------------
def main():
    # ---------
    # Load Tobias
    # ---------
    k_tob, pk_tob_num, pk_tob_den = load_tobias_file("mPk_tobias.txt")
    ratio_tob = pk_tob_num / pk_tob_den

    print("Loaded mPk_tobias.txt")
    print("At smallest k: ratio = %.6f" % ratio_tob[0])
    print("At largest  k: ratio = %.6f" % ratio_tob[-1])
    print("")

    z = 3.0

    # ------------------------------------
    # Base cosmology (Tobias dict_general-ish)
    # ------------------------------------
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

        # turn off lensing for P(k)-only runs
        "lensing": "no",

        "start_large_k_at_tau_h_over_tau_k": 0.01,
        "perturbations_verbose": "0",
        "thermodynamics_verbose": "0",
        "l_max_scalars": 2500,
    }


    # --------------------------
    # DRMD parameters (as used)
    # --------------------------
    drmd = {
        "G_over_aH_drmd_ini": 1e-3,
        "delta_Neff_drmd": 0.1,
        "f_idm_drmd": 0.1,
        "z_stop": 5000,
    }

    # --------------------------
    # ncdm parameters (as used)
    # --------------------------
    # IMPORTANT: explicitly set T_ncdm to avoid silent mismatch.
    # Default CLASS convention for standard neutrinos is often 0.71611.
    ncdm_block = {
        "N_ncdm": 1,
        "deg_ncdm": "3.0",
        "m_ncdm": "1e-2",
        "T_ncdm": "0.71611",
        # Keep your chosen approximation setting (you previously used 3 to match Tobias' comment)
        "ncdm_fluid_approximation": 3,
    }

    # ------------------------------------
    # Build models
    # ------------------------------------
    # Numerator: DRMD + ncdm + interactions
    num = dict(base)
    num.update(drmd)
    num.update(ncdm_block)
    num["G_eff_ncdm_species"] = "1e6"

    # Denominator B (Tobias dict_general): no DRMD, no ncdm
    den_no_drmd_no_ncdm = dict(base)
    # (no drmd, no ncdm keys)

    # Denominator A (DRMD baseline): DRMD, but no ncdm
    den_drmd_no_ncdm = dict(base)
    den_drmd_no_ncdm.update(drmd)

    # NEW Denominator C: DRMD + ncdm with Geff=0  (the crucial test)
    den_drmd_with_ncdm_geff0 = dict(base)
    den_drmd_with_ncdm_geff0.update(drmd)
    den_drmd_with_ncdm_geff0.update(ncdm_block)
    den_drmd_with_ncdm_geff0["G_eff_ncdm_species"] = "0.0"

    # ------------------------------------
    # Compute P(k) once for each model
    # ------------------------------------
    print("Computing numerator once (DRMD + ncdm + Geff)...")
    pk_num_tot = run_pk(num, k_tob, z, which="pk")
    pk_num_cb  = run_pk(num, k_tob, z, which="pk_cb")  # will fall back pointwise if needed

    print("Computing denominator B: no DRMD, no ncdm (Tobias dict_general)...")
    pk_denB_tot = run_pk(den_no_drmd_no_ncdm, k_tob, z, which="pk")

    print("Computing denominator A: WITH DRMD, no ncdm (DRMD baseline)...")
    pk_denA_tot = run_pk(den_drmd_no_ncdm, k_tob, z, which="pk")

    print("Computing denominator C (NEW): DRMD + ncdm, Geff=0 ...")
    pk_denC_tot = run_pk(den_drmd_with_ncdm_geff0, k_tob, z, which="pk")
    pk_denC_cb  = run_pk(den_drmd_with_ncdm_geff0, k_tob, z, which="pk_cb")

    # ------------------------------------
    # Form ratios (total-matter)
    # ------------------------------------
    ratio_you_B_tot = pk_num_tot / pk_denB_tot
    ratio_you_A_tot = pk_num_tot / pk_denA_tot
    ratio_you_C_tot = pk_num_tot / pk_denC_tot

    # cb numerators (denominators are still total unless we explicitly use cb den)
    ratio_you_B_cbnum = pk_num_cb / pk_denB_tot
    ratio_you_A_cbnum = pk_num_cb / pk_denA_tot
    ratio_you_C_cb    = pk_num_cb / pk_denC_cb  # cb / cb (best-defined cb-only ratio)

    # ------------------------------------
    # Score
    # ------------------------------------
    def report(name, r):
        s = score_ratio(r, ratio_tob, k_tob)
        print(f"\n=== {name} ===")
        print("RMS difference     = %.6e" % s["rms"])
        print("Max abs difference = %.6e" % s["maxabs"])
        print("Worst k            = %.6e 1/Mpc" % s["worst_k"])
        print("Tobias ratio       = %.6e" % s["tob_worst"])
        print("Your ratio         = %.6e" % s["you_worst"])
        return s

    report("Ratio B (Tobias denom: no DRMD, no ncdm) using pk()", ratio_you_B_tot)
    report("Ratio A (DRMD denom: DRMD baseline, no ncdm) using pk()", ratio_you_A_tot)
    report("Ratio C (NEW: DRMD+ncdm Geff on/off) using pk()", ratio_you_C_tot)

    report("Ratio B using pk_cb() numerator (cb / total-denB)", ratio_you_B_cbnum)
    report("Ratio A using pk_cb() numerator (cb / total-denA)", ratio_you_A_cbnum)
    report("Ratio C (NEW, cb/cb): pk_cb() on/off", ratio_you_C_cb)

    # ------------------------------------
    # Plot
    # ------------------------------------
    plt.figure(figsize=(9, 6))
    plt.semilogx(k_tob, ratio_tob, "k--", lw=3, label="Tobias file ratio (Pk_CLASS++ / Pk_reference)")

    plt.semilogx(k_tob, ratio_you_B_tot, lw=2, label="You: Num/(no-DRMD,no-ncdm) [pk]")
    plt.semilogx(k_tob, ratio_you_A_tot, lw=2, label="You: Num/(DRMD,no-ncdm) [pk]")
    plt.semilogx(k_tob, ratio_you_C_tot, lw=2, label="You NEW: (DRMD+ncdm+Geff)/(DRMD+ncdm Geff=0) [pk]")

    # cb definitions
    plt.semilogx(k_tob, ratio_you_C_cb, lw=2, label="You NEW cb/cb: pk_cb on/off")

    plt.axhline(1.0, color="k", lw=1)
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("P_num / P_den")
    plt.title(f"Denominator-definition test at z={z}")
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # ------------------------------------
    # Interpretation hint (printed)
    # ------------------------------------
    print("\n\n---------------- INTERPRETATION ----------------")
    print("The NEW test (Ratio C) answers: is Tobias 'Pk_reference' already including the same")
    print("massive component (ncdm) and DRMD, but with Geff=0?")
    print("If Ratio C matches Tobias best, then your earlier mismatch was largely a *reference-definition* mismatch.")
    print("If Ratio C is still systematically too low/high, then the remaining gap is likely in:")
    print("  - T_ncdm / energy-density mapping,")
    print("  - approximation switching (fluid trigger / l_max_ncdm / q grid), or")
    print("  - Tobias file produced by a different build/fork than the repo you cloned.")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    main()
