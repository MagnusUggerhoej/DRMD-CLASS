#!/usr/bin/env python3
"""
match_tobias_style_three_wrappers.py

Purpose:
- Match Tobias' parameter style as closely as possible
- Run Magnus (classy_NEDE), Tobias (classy_tobias), and Thomas (classy)
- Plot delta_ncdm[0] and theta_ncdm[0] from scalar block 0

Notes:
- Uses Tobias-style cosmology choices:
    * k_output_values = "0.05, 0.002"
    * N_ur = 2.0328
    * m_ncdm_interacting = 1e-2
    * H0/h split by wrapper style
- Keeps wrapper-specific precision keys
- Keeps DRMD off only for Magnus/Tobias because Thomas does not know those keys
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import classy_NEDE as classy_Magnus
import classy_tobias as classy_Tobias
from classy import Class as ClassThomas


# --------------------------------------------------------------------
# 0) Tobias-style shared setup
# --------------------------------------------------------------------
KVAL = "0.05, 0.002"
H0_MINE_THOMAS = 67.5
H_TOBIAS = 0.67

params_general_mine_thomas = {
    "k_output_values": KVAL,
    "output": "tCl,pCl,lCl,mPk,mTk",
    "lensing": "yes",
    "start_large_k_at_tau_h_over_tau_k": 0.01,
    "perturbations_verbose": "0",
    "thermodynamics_verbose": "0",
    "z_max_pk": 4.0,
    "H0": float(H0_MINE_THOMAS),
    "N_ur": 2.0328,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "ln10^{10}A_s": 3.0,
    "l_max_scalars": 2500,
    "n_s": 0.965,
    "gauge": "synchronous",
}

params_general_tobias = {
    "k_output_values": KVAL,
    "output": "tCl,pCl,lCl,mPk,mTk",
    "lensing": "yes",
    "start_large_k_at_tau_h_over_tau_k": 0.01,
    "perturbations_verbose": "0",
    "thermodynamics_verbose": "0",
    "z_max_pk": 4.0,
    "h": float(H_TOBIAS),
    "N_ur": 2.0328,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "ln10^{10}A_s": 3.0,
    "l_max_scalars": 2500,
    "n_s": 0.965,
    "gauge": "synchronous",
}


# --------------------------------------------------------------------
# 1) Tobias-style interacting ncdm sector
# --------------------------------------------------------------------
params_ncdm_interacting = {
    "quadrature_strategy_ncdm_interacting": 0,
    "N_momentum_bins_ncdm_interacting": 5,
    "maximum_q_ncdm_interacting": 15.0,
    "T_ncdm_interacting": 0.71611,
    "ksi_ncdm_interacting": 0.0,
    "N_ncdm_interacting": 1,
    "deg_ncdm_interacting": 1.0,
    "m_ncdm_interacting": 1e-2,
    "ncdm_fluid_approximation": 3,
    "G_eff_ncdm_interacting": 1e-3,
    "Omega_ncdm_interacting": 0.0,
}


# --------------------------------------------------------------------
# 2) Wrapper-specific precision keys
# --------------------------------------------------------------------
params_precision_mine = {
    "background_integration_stepsize": 7e-4,
    "thermo_integration_stepsize": 7e-4,
    "perturbations_integration_stepsize": 7e-4,
}

params_precision_tobias = {
    "back_integration_stepsize": 7e-4,
    # "thermo_integration_stepsize": 7e-4,  # Tobias had this commented out
    "perturb_integration_stepsize": 7e-4,
}


# --------------------------------------------------------------------
# 3) DRMD off for Magnus/Tobias only
# --------------------------------------------------------------------
params_drmd_off_mine = {
    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
}

params_drmd_off_tobias = {
    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 1e-30,  # tends to stabilize Tobias branch
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
}


# --------------------------------------------------------------------
# 4) Final parameter dicts
# --------------------------------------------------------------------
params_mine = dict(params_general_mine_thomas)
params_mine.update(params_precision_mine)
params_mine.update(params_ncdm_interacting)
params_mine.update(params_drmd_off_mine)

params_tobias = dict(params_general_tobias)
params_tobias.update(params_precision_tobias)
params_tobias.update(params_ncdm_interacting)
params_tobias.update(params_drmd_off_tobias)

params_thomas = dict(params_general_mine_thomas)
params_thomas.update(params_ncdm_interacting)
# No DRMD keys for Thomas
# No forced precision keys for Thomas


# --------------------------------------------------------------------
# 5) Runner
# --------------------------------------------------------------------
def run_cosmo(ClassObj, params, label, tries=1):
    last_err = None
    for _ in range(tries):
        m = None
        try:
            m = ClassObj()
            m.set(dict(params))
            m.compute()
            print(f"[{label}] computed.")
            return m
        except Exception as e:
            last_err = e
            try:
                if m is not None:
                    m.struct_cleanup()
                    m.empty()
            except Exception:
                pass
    raise last_err


# --------------------------------------------------------------------
# 6) Sanity prints
# --------------------------------------------------------------------
def sanity(label, p, want_drmd=False):
    print(f"\n[{label}] sanity:")
    print("  H0 =", p.get("H0"))
    print("  h  =", p.get("h"))
    print("  k_output_values =", p.get("k_output_values"))
    print("  N_ur =", p.get("N_ur"))
    print("  N_ncdm_standard =", p.get("N_ncdm_standard"))
    print("  N_ncdm_interacting =", p.get("N_ncdm_interacting"))
    print("  m_ncdm_interacting =", p.get("m_ncdm_interacting"))
    print("  G_eff_ncdm_interacting =", p.get("G_eff_ncdm_interacting"))
    if want_drmd:
        print("  f_idm_drmd =", p.get("f_idm_drmd"))
        print("  delta_Neff_drmd =", p.get("delta_Neff_drmd"))


print("Magnus so :", getattr(classy_Magnus, "__file__", "unknown"))
print("Tobias so :", getattr(classy_Tobias, "__file__", "unknown"))

sanity("Magnus", params_mine, want_drmd=True)
sanity("Tobias", params_tobias, want_drmd=True)
sanity("Thomas", params_thomas, want_drmd=False)


# --------------------------------------------------------------------
# 7) Run models
# --------------------------------------------------------------------
model_M = run_cosmo(classy_Magnus.Class, params_mine, "Magnus", tries=1)
model_T = run_cosmo(classy_Tobias.Class, params_tobias, "Tobias", tries=8)
model_TH = run_cosmo(ClassThomas, params_thomas, "Thomas", tries=1)


# --------------------------------------------------------------------
# 8) Helpers for perturbations
# --------------------------------------------------------------------
def ensure_increasing(a, block):
    a = np.array(a, dtype=float)
    if a.size >= 2 and a[0] > a[-1]:
        a = a[::-1]
        for k in list(block.keys()):
            block[k] = np.array(block[k])[::-1]
    return a, block


def get_scalar_blocks(model, label):
    pert = model.get_perturbations()
    scal = pert.get("scalar", [])
    if len(scal) < 1:
        raise RuntimeError(f"[{label}] No scalar blocks. Keys: {list(pert.keys())}")
    return scal


def print_scalar_block_info(model, label):
    scal = get_scalar_blocks(model, label)
    print(f"\n[{label}] scalar blocks: {len(scal)}")
    for i, blk in enumerate(scal):
        print(f"  block {i}: keys = {list(blk.keys())[:8]} ...")
        for k in blk.keys():
            lk = k.lower()
            if lk == "k" or "k (" in lk or lk.startswith("k_"):
                print(f"    {k} = {blk[k]}")


# --------------------------------------------------------------------
# 9) Inspect blocks and select block 0 (matching Tobias' style)
# --------------------------------------------------------------------
print_scalar_block_info(model_M, "Magnus")
print_scalar_block_info(model_T, "Tobias")
print_scalar_block_info(model_TH, "Thomas")

pert_M = get_scalar_blocks(model_M, "Magnus")[0]
pert_T = get_scalar_blocks(model_T, "Tobias")[0]
pert_TH = get_scalar_blocks(model_TH, "Thomas")[0]

a_M, pert_M = ensure_increasing(pert_M["a"], pert_M)
a_T, pert_T = ensure_increasing(pert_T["a"], pert_T)
a_TH, pert_TH = ensure_increasing(pert_TH["a"], pert_TH)


# --------------------------------------------------------------------
# 10) Plot delta_ncdm[0] and theta_ncdm[0]
# --------------------------------------------------------------------
keys_to_plot = ["delta_ncdm[0]", "theta_ncdm[0]"]

for key in keys_to_plot:
    for (lbl, blk) in [("Magnus", pert_M), ("Tobias", pert_T), ("Thomas", pert_TH)]:
        if key not in blk:
            raise KeyError(f"Missing '{key}' in {lbl}. Available keys:\n{list(blk.keys())}")

    y_M = np.array(pert_M[key], dtype=float)
    y_T = np.array(pert_T[key], dtype=float)
    y_TH = np.array(pert_TH[key], dtype=float)

    y_Ti = interp1d(a_T, y_T, kind="cubic", fill_value="extrapolate")(a_M)
    y_THi = interp1d(a_TH, y_TH, kind="cubic", fill_value="extrapolate")(a_M)

    eps = 1e-300
    rel_T = (y_M - y_Ti) / np.where(np.abs(y_M) > eps, y_M, np.nan)
    rel_TH = (y_M - y_THi) / np.where(np.abs(y_M) > eps, y_M, np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))

    axs[0].plot(a_M, y_M, label="Magnus", ls="solid")
    axs[0].plot(a_T, y_T, label="Tobias", ls="dashed")
    axs[0].plot(a_TH, y_TH, label="Thomas", ls="dotted")
    axs[0].set_xlabel("a")
    axs[0].set_ylabel(key)
    axs[0].set_xscale("log")
    axs[0].set_title(f"Comparison for {key}")
    axs[0].legend()

    axs[1].plot(a_M, rel_T, label="(Magnus - Tobias) / Magnus", ls="solid")
    axs[1].plot(a_M, rel_TH, label="(Magnus - Thomas) / Magnus", ls="dashed")
    axs[1].set_xlabel("a")
    axs[1].set_ylabel(f"Relative difference for {key}")
    axs[1].set_xscale("log")
    axs[1].axhline(0.0, ls=":")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 11) Cleanup
# --------------------------------------------------------------------
for m in (model_M, model_T, model_TH):
    try:
        m.struct_cleanup()
        m.empty()
    except Exception:
        pass

print("\nDone.")