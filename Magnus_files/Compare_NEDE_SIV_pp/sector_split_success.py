#!/usr/bin/env python3
"""
Minimal (but heavily commented) test of your new ncdm sector split / interaction knob
using wrapper: classy_NEDE (DRMD-CLASS build).



What we test is the interacting sector's effect on the matter power spectrum P(k) at z=3, which is a clean observablely affected by the new interaction/damping terms in perturbations.c. We run two cosmologies that are identical in every way except for the ncdm self-interaction strength:
------------
We run TWO cosmologies that are identical in every way EXCEPT for the ncdm
self-interaction strength:

  (A) Reference:    N_ncdm=1, same mass & degeneracy, but G_eff_ncdm_species = 0
  (B) Interacting:  N_ncdm=1, same mass & degeneracy, but G_eff_ncdm_species = 1e-3

Because the massive neutrino sector is otherwise identical, any deviation in P(k)
is attributable to your new interaction/damping implementation in perturbations.c.

Important units / conventions
-----------------------------
- model.pk(k, z) expects k in [1/Mpc] (NOT h/Mpc).
- P(k) returned is in [Mpc^3] (CLASS convention).
- We choose a wide k-grid and automatically stop before CLASS's internal k_max.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- YOUR wrapper (DRMD-CLASS / NEDE build) ---
import classy_NEDE as classy_pp


def safe_pk_array(model, ks, z):
    """
    Evaluate P(k,z) on the array ks, but stop cleanly before CLASS's internal k_max.

    Why this exists:
    - CLASS only precomputes P(k,z) up to some k_max depending on settings like
      P_k_max_h/Mpc, k_output_values, etc.
    - If you call pk(k,z) above that, the wrapper may raise an exception.
    - We "walk upward in k" and stop at the first failure to keep the script robust.

    Returns:
        k_valid : np.ndarray  (subset of ks that worked)
        pk_valid: np.ndarray  (P(k,z) for those k)
    """
    k_list = []
    pk_list = []

    for k in ks:
        try:
            pk = model.pk(float(k), float(z))
            k_list.append(float(k))
            pk_list.append(float(pk))
        except Exception:
            # Once we hit out-of-range (or other pk failure), stop.
            break

    if len(k_list) < 5:
        raise RuntimeError(
            "pk() failed too early. Check you included 'mPk' in output and set z_max_pk >= z. "
            "You may also need to raise P_k_max_h/Mpc if k-range is too small."
        )

    return np.array(k_list), np.array(pk_list)


def interp_loglog(x_src, y_src, x_new):
    """
    Log-log interpolation.

    This is the standard safe choice for spectra because:
    - P(k) is positive
    - P(k) can vary by many orders of magnitude
    - log-log interpolation avoids distortions from linear interpolation
    """
    return np.exp(np.interp(np.log(x_new), np.log(x_src), np.log(y_src)))


def main():
    # ------------------------------------------------------------
    # 1) Base cosmology (shared by both runs)
    # ------------------------------------------------------------
    # Keep this as simple as possible while still enabling pk():
    base = {
        # Request matter power spectrum; pk() only works if mPk is in output
        "output": "mPk",

        # We will query P(k,z) at z=3, so CLASS must compute P(k,z) up to at least z=3
        "z_max_pk": 4.0,

        # Background / primordial parameters (standard-ish)
        "H0": 67.5,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "tau_reio": 0.054,
        "n_s": 0.965,
        "ln10^{10}A_s": 3.0,

        # Radiation content (as in Tobias-style setups)
        "N_ur": 2.0328,

        # Optional: reduce spam
        "perturbations_verbose": "0",
        "thermodynamics_verbose": "0",

        # Optional: widen accessible k-range for pk() (uncomment if pk() stops too early)
        # "P_k_max_h/Mpc": 10.0,
    }

    # ------------------------------------------------------------
    # 2) Massive neutrino / ncdm sector settings (identical in both)
    # ------------------------------------------------------------
    # We model "three degenerate neutrinos" as one ncdm species with degeneracy=3.
    # This mirrors Tobias' idea of "deg_ncdm_interacting = 3" with 1 species.
    ncdm_common = {
        "N_ncdm": 1,
        "deg_ncdm": "3.0",   # one-entry list as string (CLASS parser style)
        "m_ncdm": "1e-2",    # one-entry list as string (eV)
        # "T_ncdm": "0.71611",  # optional; defaults are usually fine
    }

    # ------------------------------------------------------------
    # 3) Build the TWO runs: reference vs interacting
    # ------------------------------------------------------------
    # Reference: SAME ncdm sector, but interactions turned OFF
    params_ref = dict(base)
    params_ref.update(ncdm_common)
    params_ref.update({
        "G_eff_ncdm_species": "0",      # <-- key point: no interactions
        # (If your code also supports scalar ppt->G_eff_ncdm, keep it off too)
        # "G_eff_ncdm": 0.0,
    })

    # Interacting: SAME ncdm sector, interactions turned ON
    params_int = dict(base)
    params_int.update(ncdm_common)
    params_int.update({
        "G_eff_ncdm_species": "1e-3",   # <-- key point: interactions ON
        # "G_eff_ncdm": 1e-3,
    })

    # ------------------------------------------------------------
    # 4) Compute both cosmologies
    # ------------------------------------------------------------
    # Use separate Class instances so there is zero shared internal state.
    model_ref = classy_pp.Class()
    model_ref.set(params_ref)
    model_ref.compute()

    model_int = classy_pp.Class()
    model_int.set(params_int)
    model_int.compute()

    # ------------------------------------------------------------
    # 5) Evaluate P(k,z) for both and align on a common k-grid
    # ------------------------------------------------------------
    z = 3.0

    # k grid in 1/Mpc (not h/Mpc). Wide on purpose; safe_pk_array will clip.
    ks_req = np.logspace(-4, 1, 800)

    # Safely evaluate until CLASS's internal k_max is reached
    k_r, pk_r = safe_pk_array(model_ref, ks_req, z)
    k_i, pk_i = safe_pk_array(model_int, ks_req, z)

    # Use only the range where BOTH runs have valid P(k)
    kmax_common = min(k_r[-1], k_i[-1])
    k_common = ks_req[ks_req <= 0.999 * kmax_common]

    # Interpolate both spectra onto the exact same k grid (log-log)
    pk_r_i = interp_loglog(k_r, pk_r, k_common)
    pk_i_i = interp_loglog(k_i, pk_i, k_common)

    # Ratio isolates "interaction effect" because everything else is identical
    ratio = pk_i_i / pk_r_i

    # ------------------------------------------------------------
    # 6) Plot absolute spectra
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.loglog(k_common, pk_r_i, label="Reference: G_eff=0")
    plt.loglog(k_common, pk_i_i, label="Interacting: G_eff=1e-3")
    plt.xlabel(r"$k\;[1/\mathrm{Mpc}]$")
    plt.ylabel(r"$P(k,z)\;[\mathrm{Mpc}^3]$")
    plt.title(f"classy_NEDE: P(k) comparison at z={z}")
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------
    # 7) Plot ratio
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.semilogx(k_common, ratio)
    plt.axhline(1.0)
    plt.xlabel(r"$k\;[1/\mathrm{Mpc}]$")
    plt.ylabel(r"$P_{\mathrm{int}}(k)/P_{\mathrm{ref}}(k)$")
    plt.title("Interaction-only effect (same ncdm sector, different G_eff)")
    plt.tight_layout()

    plt.show()

    # ------------------------------------------------------------
    # 8) Cleanup: free CLASS internal memory
    # ------------------------------------------------------------
    model_ref.struct_cleanup()
    model_ref.empty()
    model_int.struct_cleanup()
    model_int.empty()


if __name__ == "__main__":
    main()
