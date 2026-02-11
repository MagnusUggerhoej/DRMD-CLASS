import numpy as np
import matplotlib.pyplot as plt

import classy_SIN as classy_pp


dict_general = {
    "k_output_values": "0.05, 0.002",
    "output": "mPk",
    "z_max_pk": 4.0,

    "H0": 67.5,
    "N_ur": 2.0328,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "ln10^{10}A_s": 3.0,
    "n_s": 0.965,

    # IMPORTANT: avoid uninitialized/garbage default in this fork
    "f_idm_drmd": 0.0,
}

dict_specific = {
    **dict_general,

    # --- Interacting NCDM (as Tobias) ---
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,
    "deg_ncdm_interacting": 3,
    "m_ncdm_interacting": 1e-2,
    "ncdm_fluid_approximation": 3,
    "G_eff_ncdm_interacting": 1e-3,

    # --- DRMD block (as Tobias) ---
    "G_over_aH_drmd_ini": 1e-3,
    "delta_Neff_drmd": 0.1,
    "f_idm_drmd": 0.1,   # override baseline 0.0 for the interacting case
    "z_stop": 5000,
}

def mPk(model, klist_hmpc):
    """
    Tobias convention:
      - choose klist in h/Mpc
      - convert to 1/Mpc by multiplying by h()
      - convert P(k) back to (Mpc/h)^3 by multiplying h^3
    """
    h = model.h()
    return np.array([model.pk(k*h, 0.0) * h**3 for k in klist_hmpc])

def main():
    # Build BOTH models by passing dict to constructor (important for this wrapper)
    model_specific = classy_pp.Class(dict_specific)
    model_specific.compute()

    model_ref = classy_pp.Class(dict_general)
    model_ref.compute()

    # k in h/Mpc (like Tobias)
    kmax = 1.0
    klist = np.logspace(-4, np.log10(kmax), 1000)

    pk_specific = mPk(model_specific, klist)
    pk_ref = mPk(model_ref, klist)

    # cleanup (good habit)
    model_specific.struct_cleanup()
    model_specific.empty()
    model_ref.struct_cleanup()
    model_ref.empty()

    # Plot P(k)
    plt.figure()
    plt.loglog(klist, pk_specific, label="SIN: interacting model")
    plt.loglog(klist, pk_ref, label="SIN: reference (no interaction)")
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("P(k) [(Mpc/h)^3]")
    plt.title("classy_SIN P(k) smoke+comparison")
    plt.legend()
    plt.show()

    # Plot ratio
    plt.figure()
    plt.semilogx(klist, pk_specific / pk_ref)
    plt.axhline(1.0)
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("P(k)_model / P(k)_ref")
    plt.title("Ratio")
    plt.show()

if __name__ == "__main__":
    main()
