#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --- use YOUR wrapper (DRMD-CLASS / NEDE build) ---
import classy_NEDE as classy_pp


def safe_pk_array(model, ks, z):
    """
    Evaluate P(k,z) on ks, but stop cleanly before CLASS's internal k_max.
    Returns (k_valid, pk_valid).
    """
    pk_list = []
    k_list = []
    for k in ks:
        try:
            pk = model.pk(float(k), float(z))
            k_list.append(float(k))
            pk_list.append(float(pk))
        except Exception:
            # once we hit out-of-bounds, stop
            break

    if len(k_list) < 5:
        raise RuntimeError(
            "pk() failed too early. You may need output='mPk' and z_max_pk >= z."
        )

    return np.array(k_list), np.array(pk_list)


def interp_loglog(x_src, y_src, x_new):
    """Log-log interpolation (safe for positive spectra)."""
    return np.exp(np.interp(np.log(x_new), np.log(x_src), np.log(y_src)))


def main():
    # --- mimic Tobias' two dictionaries, but mapped to YOUR implementation ---
    dict_general = {
        "k_output_values": "0.05, 0.002",
        "output": "tCl,pCl,lCl,mPk,mTk",
        "lensing": "yes",
        "start_large_k_at_tau_h_over_tau_k": 0.01,
        "perturbations_verbose": "0",
        "thermodynamics_verbose": "0",
        "z_max_pk": 4.0,
        "H0": 67.5,
        "N_ur": 2.0328,
        # Tobias sets standard ncdm sector to 0. In YOUR build, just don't add ncdm here.
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "tau_reio": 0.054,
        "ln10^{10}A_s": 3,
        "l_max_scalars": 2500,
        "n_s": 0.965,
    }

    dict_specific = dict(dict_general)

    # ---- Tobias "interacting sector" translated to YOUR "per-species coupling" ----
    # Tobias: N_ncdm_interacting=1, deg=3, m=1e-2  -> in YOUR build: N_ncdm=1 with deg=3
    dict_specific.update({
        "N_ncdm": 1,
        "deg_ncdm": "3.0",
        "m_ncdm": "1e-2",
        # (T_ncdm is optional; leaving it default is usually fine)
        # Your new knob:
        "G_eff_ncdm_species": "1e-3",   # length must equal N_ncdm (=1)
        # brute-force fallback (optional; remove later)
        # "G_eff_ncdm": 1e-3,
    })

    # ---- DRMD params (keep exactly like Tobias if your build supports them) ----
    dict_specific.update({
        "G_over_aH_drmd_ini": 1e-3,
        "delta_Neff_drmd": 0.1,
        "f_idm_drmd": 0.1,
        "z_stop": 5000,
    })

    # --- run models ---
    model_pp = classy_pp.Class()
    model_pp.set(dict_specific)
    model_pp.compute()

    model_ref = classy_pp.Class()
    model_ref.set(dict_general)
    model_ref.compute()

    # --- compute spectra on a requested k-grid, then safely intersect ---
    z = 3.0
    ks_req = np.logspace(-4, 1, 800)  # wide; will clip automatically

    k_pp, pk_pp = safe_pk_array(model_pp, ks_req, z)
    k_rf, pk_rf = safe_pk_array(model_ref, ks_req, z)

    kmax_common = min(k_pp[-1], k_rf[-1])
    k_common = ks_req[ks_req <= kmax_common * 0.999]

    pk_pp_i = interp_loglog(k_pp, pk_pp, k_common)
    pk_rf_i = interp_loglog(k_rf, pk_rf, k_common)

    # --- plot 1: absolute spectra ---
    plt.figure()
    plt.loglog(k_common, pk_pp_i, label="CLASS_NEDE: interacting (mapped)")
    plt.loglog(k_common, pk_rf_i, label="CLASS_NEDE: reference (G_eff=0)")
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("mP(k)  [(Mpc)^3]")  # label like Tobias; actual is P(k)
    plt.title("Comparison between CLASS_NEDE runs")
    plt.legend()
    plt.tight_layout()

    # --- plot 2: ratio ---
    plt.figure()
    plt.semilogx(k_common, pk_pp_i / pk_rf_i)
    plt.axhline(1.0)
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("mP(k)_int / mP(k)_ref")
    plt.title("Ratio: interacting / reference")
    plt.tight_layout()

    plt.show()

    # --- cleanup ---
    model_pp.struct_cleanup()
    model_pp.empty()
    model_ref.struct_cleanup()
    model_ref.empty()


if __name__ == "__main__":
    main()
