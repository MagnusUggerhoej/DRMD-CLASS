#!/usr/bin/env python3
"""
test_three_wrappers_single_k.py

Goal:
- Run the same NCDM-interacting setup in:
    1) Magnus (classy_NEDE)
    2) Tobias (classy_tobias)
    3) Thomas/supervisor (classy)

Key points:
- Uses canonical CLASS keys (H0, omega_b, omega_cdm, etc.)
- Disables DRMD in Magnus/Tobias safely (but does NOT pass DRMD keys to Thomas)
- Avoids the "sometimes crashes in classy_tobias __init__" issue by:
    - calling set() immediately after construction
    - retrying Tobias constructor+compute a few times if the branch has uninitialized defaults
- Plots ONLY delta_ncdm[0] and theta_ncdm[0] for scalar block 0 (single k_output_values)

Run:
  conda activate CLASS
  python test_three_wrappers_single_k.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import classy_NEDE as classy_Magnus
import classy_tobias as classy_Tobias
from classy import Class as ClassThomas


# --------------------------------------------------------------------
# 0) Minimal shared cosmology + one-k perturbation output
# --------------------------------------------------------------------
#KVAL = "0.05"  # single k block
KVAL = '0.05, 0.002'
H0 = 67.0
h = H0 / 100.0 #tobias

shared_common = {
    "output": "tCl,pCl,lCl,mPk,mTk",
    "lensing": "yes",
    "k_output_values": KVAL,
    "z_max_pk": 4.0,
    "H0": float(H0),
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "ln10^{10}A_s": 3.0,
    "n_s": 0.965,
    "gauge": "synchronous",
    "start_large_k_at_tau_h_over_tau_k": 0.01,
    "perturbations_verbose": "0",
    "thermodynamics_verbose": "0",
    # Neff budget consistent with your setup
    "N_ur": 2.0328,
}

# stepsizes:
# - Magnus accepts background_integration_stepsize + perturbations_integration_stepsize
# - Tobias accepts back_integration_stepsize + perturb_integration_stepsize
# - Thomas varies by branch; we keep it minimal and don't force stepsizes
steps_Magnus = {
    "background_integration_stepsize": 7e-4,
    "perturbations_integration_stepsize": 7e-4,
}
steps_Tobias = {
    "back_integration_stepsize": 7e-4,
    "perturb_integration_stepsize": 7e-4,
}

# --------------------------------------------------------------------
# 1) Interacting ncdm sector (single interacting species)
# --------------------------------------------------------------------
ncdm_interacting = {
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,
    "T_ncdm_interacting": 0.71611,
    "ksi_ncdm_interacting": 0.0,
    "m_ncdm_interacting": 1e-2,
    "deg_ncdm_interacting": 1.0,
    "ncdm_fluid_approximation": 3,
    "G_eff_ncdm_interacting": 1e-3,
    "Omega_ncdm_interacting": 0.0,
    "quadrature_strategy_ncdm_interacting": 0,
    "N_momentum_bins_ncdm_interacting": 5,
    "maximum_q_ncdm_interacting": 15.0,
}

# --------------------------------------------------------------------
# 2) DRMD off
# IMPORTANT: Thomas does NOT know DRMD keys -> never pass them to Thomas.
# Tobias branch appears to require these to be present (even if off),
# but has uninitialized defaults in some builds; keep them explicit.
# --------------------------------------------------------------------
drmd_off_M = {
    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
}
# Tobias sometimes behaves better with a tiny nonzero delta_Neff_drmd
drmd_off_T = {
    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 1e-30,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
}

# --------------------------------------------------------------------
# 3) Build params per wrapper
# --------------------------------------------------------------------
params_M = {}
params_M.update(shared_common)
params_M.update(steps_Magnus)
params_M.update(drmd_off_M)
params_M.update(ncdm_interacting)

params_T = {}
params_T.update(shared_common)
params_T.update(steps_Tobias)
params_T.update(drmd_off_T)
params_T.update(ncdm_interacting)

params_TH = {}
params_TH.update(shared_common)
# no stepsizes forced, no DRMD keys
params_TH.update(ncdm_interacting)


# --------------------------------------------------------------------
# 4) Runner with retries (needed for Tobias if branch has uninitialized defaults)
# --------------------------------------------------------------------
def run_cosmo(ClassObj, params, label, tries=1):
    last_err = None
    for i in range(tries):
        m = None
        try:
            m = ClassObj()
            m.set(params)
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
# 5) Sanity prints
# --------------------------------------------------------------------
def sanity(label, p, want_drmd=False):
    print(f"\n[{label}] sanity:")
    print("  H0              =", p.get("H0"))
    print("  k_output_values =", p.get("k_output_values"))
    print("  Nstd/Nint       =", p.get("N_ncdm_standard"), p.get("N_ncdm_interacting"))
    print("  Geff            =", p.get("G_eff_ncdm_interacting"))
    if want_drmd:
        print("  f_idm_drmd      =", p.get("f_idm_drmd"))
        print("  delta_Neff_drmd =", p.get("delta_Neff_drmd"))


print("Magnus so :", getattr(classy_Magnus, "__file__", "unknown"))
print("Tobias so :", getattr(classy_Tobias, "__file__", "unknown"))

sanity("Params Magnus", params_M, want_drmd=True)
sanity("Params Tobias", params_T, want_drmd=True)
sanity("Params Thomas", params_TH, want_drmd=False)


# --------------------------------------------------------------------
# 6) Run three models
# --------------------------------------------------------------------
model_M = run_cosmo(classy_Magnus.Class, params_M, "Magnus", tries=1)
model_T = run_cosmo(classy_Tobias.Class, params_T, "Tobias", tries=8)
model_TH = run_cosmo(ClassThomas, params_TH, "Thomas", tries=1)


# --------------------------------------------------------------------
# 7) Extract perturbations (scalar block 0) and plot delta/theta for ncdm[0]
# --------------------------------------------------------------------
def get_scalar_block(model, label):
    pert = model.get_perturbations()
    scal = pert.get("scalar", [])
    if len(scal) < 1:
        raise RuntimeError(f"[{label}] No scalar perturbation blocks. Keys: {list(pert.keys())}")
    return scal[0]


pert_M = get_scalar_block(model_M, "Magnus")
pert_T = get_scalar_block(model_T, "Tobias")
pert_TH = get_scalar_block(model_TH, "Thomas")

print("\n[Perturbation blocks]")
print("  Magnus scalar blocks:", len(model_M.get_perturbations().get("scalar", [])))
print("  Tobias scalar blocks:", len(model_T.get_perturbations().get("scalar", [])))
print("  Thomas scalar blocks:", len(model_TH.get_perturbations().get("scalar", [])))
print("  (Expected 1 block each because k_output_values is a single value.)")

a_M = np.array(pert_M["a"], dtype=float)
a_T = np.array(pert_T["a"], dtype=float)
a_TH = np.array(pert_TH["a"], dtype=float)

def ensure_increasing(a, block):
    if a[0] > a[-1]:
        a = a[::-1]
        for k in list(block.keys()):
            block[k] = np.array(block[k])[::-1]
    return a, block

a_M, pert_M = ensure_increasing(a_M, pert_M)
a_T, pert_T = ensure_increasing(a_T, pert_T)
a_TH, pert_TH = ensure_increasing(a_TH, pert_TH)

keys_to_plot = ["delta_ncdm[0]", "theta_ncdm[0]"]

for key in keys_to_plot:
    for (lbl, blk) in [("Magnus", pert_M), ("Tobias", pert_T), ("Thomas", pert_TH)]:
        if key not in blk:
            raise KeyError(f"Missing '{key}' in {lbl}. Available keys:\n{list(blk.keys())}")

    y_M = np.array(pert_M[key], dtype=float)
    y_T = np.array(pert_T[key], dtype=float)
    y_TH = np.array(pert_TH[key], dtype=float)

    # Interpolate Tobias and Thomas onto Magnus a-grid
    y_Ti = interp1d(a_T, y_T, kind="cubic", fill_value="extrapolate")(a_M)
    y_THi = interp1d(a_TH, y_TH, kind="cubic", fill_value="extrapolate")(a_M)

    eps = 1e-300
    rel_T = (y_M - y_Ti) / np.where(np.abs(y_M) > eps, y_M, np.nan)
    rel_TH = (y_M - y_THi) / np.where(np.abs(y_M) > eps, y_M, np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))

    axs[0].plot(a_M, y_M, label="Magnus", ls="solid")
    axs[0].plot(a_T, y_T, label="Tobias", ls="dashed")
    axs[0].plot(a_TH, y_TH, label="Thomas", ls="dotted")
    axs[0].set(xlabel="a", ylabel=key, xscale="log")
    axs[0].set_title(f"Comparison for {key}")
    axs[0].legend()

    axs[1].plot(a_M, rel_T, label="(Magnus - Tobias) / Magnus", ls="solid")
    axs[1].plot(a_M, rel_TH, label="(Magnus - Thomas) / Magnus", ls="dashed")
    axs[1].set(xlabel="a", ylabel=f"Relative difference for {key}", xscale="log")
    axs[1].axhline(0.0, ls=":")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 8) Cleanup
# --------------------------------------------------------------------
for m in (model_M, model_T, model_TH):
    try:
        m.struct_cleanup()
        m.empty()
    except Exception:
        pass

print("\nDone.")