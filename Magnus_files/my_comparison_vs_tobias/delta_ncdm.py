#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import classy_NEDE as classy_Magnus
import classy_tobias as classy_Tobias


dict_general_pp={
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk,mTk',
        'lensing':'yes',
        'start_large_k_at_tau_h_over_tau_k': 0.01,
        'perturbations_verbose':'0',
        'thermodynamics_verbose':'0',
        'z_max_pk': 4.,
        'h': 0.67,
        'N_ur' : 2.0328,
        #'N_ncdm': 1,        
        #'deg_ncdm': 3, #Three degenerate interacting neutrinos with
        #'m_ncdm': 1e-2,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,
        'back_integration_stepsize': 7e-4,
        #'thermo_integration_stepsize': 7e-4,
        'perturb_integration_stepsize': 7e-4,
        'gauge': 'synchronous',

    }


dict_general={
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk,mTk',
        'lensing':'yes',
        'start_large_k_at_tau_h_over_tau_k': 0.01,
        'perturbations_verbose':'0',
        'thermodynamics_verbose':'0',
        'z_max_pk': 4.,
        'H0': 67.5,
        'N_ur' : 2.0328,
        #'N_ncdm': 1,        
        #'deg_ncdm': 3, #Three degenerate interacting neutrinos with
        #'m_ncdm': 1e-2,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,
        'background_integration_stepsize': 7e-4,
        'thermo_integration_stepsize': 7e-4,
        'perturbations_integration_stepsize': 7e-4,
        'gauge': 'synchronous',
    }

dict_ncdm={
        'quadrature_strategy_ncdm_interacting': 0,
        'N_momentum_bins_ncdm_interacting': 5,
        'maximum_q_ncdm_interacting': 15.0,
        'T_ncdm_interacting': 0.71611, #T_ncdm_default
        'ksi_ncdm_interacting': 0.0, #ksi_ncdm_default
        'N_ncdm_interacting': 1,        
        'deg_ncdm_interacting':1, #Three degenerate interacting neutrinos with
        'm_ncdm_interacting': 1e-2,
        'ncdm_fluid_approximation': 3,
        'G_eff_ncdm_interacting': 1e-3,
        'Omega_ncdm_interacting': 0.0,
    }


dict_drmd={ 
        'G_over_aH_drmd_ini':1e-3,      
        'delta_Neff_drmd':1.0,
        'f_idm_drmd':0.5,
        'z_stop':5000,
}


drmd_off = {
    "f_idm_drmd": 1e-5,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
}

def get_dict_general():
    return dict_general

def get_dict_general_pp():
    return dict_general_pp

def get_dict_ncdm():
    dict_out = dict_general.copy()
    dict_out.update(drmd_off)   # <-- keep this
    dict_out.update(dict_ncdm)
    return dict_out

def get_dict_ncdm_pp():
    dict_out = dict_general_pp.copy()
    dict_out.update(drmd_off)   # <-- keep this
    dict_out.update(dict_ncdm)
    return dict_out

def get_dict_drmd():
    dict_out = dict_general.copy()
    dict_out.update(dict_drmd)
    return dict_out

def get_dict_drmd_pp():
    dict_out = dict_general_pp.copy()
    dict_out.update(dict_drmd)
    return dict_out

# ------------------------------------------------------------
# You MUST have these two functions defined exactly as in your
# working setup (or replace them with explicit dict literals).
#   - get_dict_ncdm()     -> dict for Magnus wrapper
#   - get_dict_ncdm_pp()  -> dict for Tobias wrapper
# ------------------------------------------------------------
# from your_file_with_dicts import get_dict_ncdm, get_dict_ncdm_pp


def run_cosmo(classy_version, params):
    m = classy_version.Class()
    m.set(params)
    m.compute()
    return m


# -----------------------------
# 1) Run both cosmologies
# -----------------------------
model_Magnus_ncdm = run_cosmo(classy_Magnus, get_dict_ncdm())
p = get_dict_ncdm_pp()
print("Tobias params contain f_idm_drmd?", "f_idm_drmd" in p, "value =", p.get("f_idm_drmd"))
print("Tobias params contain delta_Neff_drmd?", "delta_Neff_drmd" in p, p.get("delta_Neff_drmd"))
print("Tobias params contain z_stop?", "z_stop" in p, p.get("z_stop"))
print("Tobias params contain G_over_aH_drmd_ini?", "G_over_aH_drmd_ini" in p, p.get("G_over_aH_drmd_ini"))
model_Tobias_ncdm = run_cosmo(classy_Tobias, get_dict_ncdm_pp())

# -----------------------------
# 2) Extract scalar perturbations (k-index 0)
# -----------------------------
pert_M = model_Magnus_ncdm.get_perturbations()["scalar"][0]
pert_T = model_Tobias_ncdm.get_perturbations()["scalar"][0]

a_M = np.array(pert_M["a"], dtype=float)
a_T = np.array(pert_T["a"], dtype=float)

# Safety: ensure increasing a for interpolation
# (CLASS usually returns increasing a, but let's not assume)
if a_T[0] > a_T[-1]:
    a_T = a_T[::-1]
    for k in list(pert_T.keys()):
        pert_T[k] = np.array(pert_T[k])[::-1]

if a_M[0] > a_M[-1]:
    a_M = a_M[::-1]
    for k in list(pert_M.keys()):
        pert_M[k] = np.array(pert_M[k])[::-1]

# -----------------------------
# 3) Plot only the two keys
# -----------------------------
keys_to_plot = ["delta_ncdm[0]", "theta_ncdm[0]"]

for key in keys_to_plot:
    if key not in pert_M or key not in pert_T:
        print(f"Key '{key}' missing. Available keys in Magnus are:\n{list(pert_M.keys())}\n")
        raise KeyError(key)

    y_M = np.array(pert_M[key], dtype=float)
    y_T = np.array(pert_T[key], dtype=float)

    # Interpolate Tobias onto Magnus a-grid
    interp_T = interp1d(a_T, y_T, kind="cubic", fill_value="extrapolate")
    y_Ti = interp_T(a_M)

    # Relative difference: (M - T)/M
    # (Guard against division by ~0)
    eps = 1e-300
    rel = (y_M - y_Ti) / np.where(np.abs(y_M) > eps, y_M, np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(a_M, y_M, label="Magnus", ls="solid")
    axs[0].plot(a_T, y_T, label="Tobias", ls="dashed")
    axs[0].set(xlabel="a", ylabel=key, xscale="log")
    axs[0].set_title(f"Comparison for {key}")
    axs[0].legend()

    axs[1].plot(a_M, rel, label="(Magnus - Tobias) / Magnus", ls="solid")
    axs[1].set(xlabel="a", ylabel=f"Relative difference for {key}", xscale="log")
    axs[1].axhline(0.0, ls=":")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# 4) Cleanup
# -----------------------------
for m in (model_Magnus_ncdm, model_Tobias_ncdm):
    try:
        m.struct_cleanup()
        m.empty()
    except Exception:
        pass

print("Done.")