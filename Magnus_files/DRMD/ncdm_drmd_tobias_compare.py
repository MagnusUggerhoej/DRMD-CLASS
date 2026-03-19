#!/usr/bin/env python3

'''
The script does the following three things:
    1) builds a true ΛCDM baseline with no massive ncdm at all

    2) builds the DRMD + interacting ncdm model for Mine and Tobias

    3) plots:

        - top: ΛCDM, Mine new model, Tobias new model

        - bottom: percent deviation relative to ΛCDM

        - and an extra comparison plot of Mine vs Tobias for the new model only

So this lets you show both:

    - that the new model differs from standard ΛCDM

    - and that Mine and Tobias agree on the new model

'''


import numpy as np
import matplotlib.pyplot as plt

from classy_NEDE import Class as ClassMine
from classy_tobias import Class as ClassTobias


# =============================================================================
# Helpers
# =============================================================================
def pk_safe(cosmo, k_1Mpc_arr, z):
    fn = cosmo.pk_lin if hasattr(cosmo, "pk_lin") else cosmo.pk
    return np.array([fn(float(k), float(z)) for k in k_1Mpc_arr])

def get_derived_safe(cosmo):
    out = {}
    for key in ["h", "Omega_m", "Omega_r", "z_eq"]:
        try:
            out[key] = cosmo.get_current_derived_parameters([key])[key]
        except Exception:
            pass

    try:
        out["h_direct"] = cosmo.h()
    except Exception:
        pass

    try:
        out["Neff"] = cosmo.get_current_derived_parameters(["Neff"])["Neff"]
    except Exception:
        pass

    return out

def run_model(ClassObj, params, label):
    c = ClassObj()
    c.set(params)
    c.compute()
    print(f"[{label}] computed.")
    d = get_derived_safe(c)
    if d:
        print(f"[{label}] derived: {d}")
    return c

def cleanup(*cosmos):
    for c in cosmos:
        try:
            c.struct_cleanup()
            c.empty()
        except Exception:
            pass

def print_param_block(name, p):
    print(f"\n--- Parameters: {name} ---")
    for k, v in p.items():
        print(f"{k}: {v}")


# =============================================================================
# Shared cosmology
# =============================================================================
H0        = 67.32
omega_b   = 0.02238
omega_cdm = 0.1201
A_s       = 2.101e-9
n_s       = 0.9660
tau_reio  = 0.0543
k_pivot   = 0.05
T_cmb     = 2.7255
Omega_k   = 0.0

K_1MPC = np.logspace(-4, 1.0, 250)
Z_PK   = 0.0
L_MAX  = 2500

DEG_NCDM  = 1.0
M_NCDM_EV = 0.06
T_NCDM    = 0.71611
N_ur_run  = max(0.0, 3.046 - DEG_NCDM)


# =============================================================================
# DRMD + interacting ncdm test point
# =============================================================================
DELTA_NEFF_ON = 0.8
F_IDM_ON      = 0.03
Z_STOP_ON     = 5.0e4
G_AH_ON       = 1.0e9
G_EFF_NCDM_ON = 1e-1


# =============================================================================
# Base params
# =============================================================================
def base_params():
    return {
        "H0": float(H0),
        "omega_b": float(omega_b),
        "omega_cdm": float(omega_cdm),
        "A_s": float(A_s),
        "n_s": float(n_s),
        "tau_reio": float(tau_reio),
        "k_pivot": float(k_pivot),
        "T_cmb": float(T_cmb),
        "Omega_k": float(Omega_k),

        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": int(L_MAX),
        "P_k_max_1/Mpc": float(np.max(K_1MPC) * 1.05),

        "recombination": "recfast",
        "reio_parametrization": "reio_camb",
    }


# =============================================================================
# True LCDM baseline (no massive ncdm)
# =============================================================================
# Tobias LCDM
pT_lcdm = base_params()
pT_lcdm.update({
    "N_ur": 3.046,
    "N_ncdm": 0,
})
for key in [
    "m_ncdm", "deg_ncdm", "T_ncdm",
    "N_ncdm_interacting", "m_ncdm_interacting", "deg_ncdm_interacting",
    "G_eff_ncdm_interacting", "log10G_eff_ncdm_interacting",
    "N_ncdm_standard"
]:
    pT_lcdm.pop(key, None)

# Mine LCDM
pM_lcdm = base_params()
pM_lcdm.update({
    "N_ur": 3.046,
})
for key in [
    "N_ncdm", "N_ncdm_standard", "N_ncdm_interacting",
    "m_ncdm", "deg_ncdm", "T_ncdm",
    "m_ncdm_interacting", "deg_ncdm_interacting",
    "G_eff_ncdm_interacting", "log10G_eff_ncdm_interacting",
    "G_eff_ncdm", "log10_G_eff_ncdm",
    "ncdm_fluid_approximation",
]:
    pM_lcdm.pop(key, None)


# =============================================================================
# DRMD ON + interacting ncdm
# =============================================================================
# Tobias: interacting interface only
pT_on = base_params()
pT_on.update({
    "N_ur"                   : float(N_ur_run),

    "N_ncdm_interacting"     : 1,
    "m_ncdm_interacting"     : float(M_NCDM_EV),
    "deg_ncdm_interacting"   : float(DEG_NCDM),
    "G_eff_ncdm_interacting" : float(G_EFF_NCDM_ON),

    "f_idm_drmd"             : float(F_IDM_ON),
    "delta_Neff_drmd"        : float(DELTA_NEFF_ON),
    "z_stop"                 : float(Z_STOP_ON),
    "G_over_aH_drmd_ini"     : float(G_AH_ON),
})
for key in ["N_ncdm", "m_ncdm", "deg_ncdm", "T_ncdm"]:
    pT_on.pop(key, None)

# Mine: split keys + legacy arrays for m/deg
pM_on = base_params()
pM_on.update({
    "N_ur"                   : float(N_ur_run),

    "N_ncdm_standard"        : 0,
    "N_ncdm_interacting"     : 1,
    "m_ncdm"                 : float(M_NCDM_EV),
    "deg_ncdm"               : float(DEG_NCDM),
    "G_eff_ncdm_interacting" : float(G_EFF_NCDM_ON),

    "f_idm_drmd"             : float(F_IDM_ON),
    "delta_Neff_drmd"        : float(DELTA_NEFF_ON),
    "z_stop"                 : float(Z_STOP_ON),
    "G_over_aH_drmd_ini"     : float(G_AH_ON),
})
pM_on.pop("N_ncdm", None)


# =============================================================================
# Print setup
# =============================================================================
print("\n========================================")
print("LCDM vs DRMD + interacting ncdm")
print("========================================")

print_param_block("Mine   LCDM", pM_lcdm)
print_param_block("Tobias LCDM", pT_lcdm)
print_param_block("Mine   ON",   pM_on)
print_param_block("Tobias ON",   pT_on)


# =============================================================================
# Run all four cases
# =============================================================================
cM_lcdm = run_model(ClassMine,   pM_lcdm, "Mine   LCDM")
cT_lcdm = run_model(ClassTobias, pT_lcdm, "Tobias LCDM")

cM_on   = run_model(ClassMine,   pM_on,   "Mine   ON")
cT_on   = run_model(ClassTobias, pT_on,   "Tobias ON")


# =============================================================================
# Compute P(k)
# =============================================================================
h_ref  = float(cT_lcdm.h())
k_hMpc = K_1MPC / h_ref

PkM_lcdm = pk_safe(cM_lcdm, K_1MPC, Z_PK) * h_ref**3
PkT_lcdm = pk_safe(cT_lcdm, K_1MPC, Z_PK) * h_ref**3

PkM_on   = pk_safe(cM_on,   K_1MPC, Z_PK) * h_ref**3
PkT_on   = pk_safe(cT_on,   K_1MPC, Z_PK) * h_ref**3

# Use Tobias LCDM as the displayed baseline
Pk_ref = PkT_lcdm

ratio_M_vs_LCDM = PkM_on / Pk_ref
ratio_T_vs_LCDM = PkT_on / Pk_ref
ratio_M_vs_T    = PkM_on / PkT_on

diff_M_vs_LCDM = 100.0 * (ratio_M_vs_LCDM - 1.0)
diff_T_vs_LCDM = 100.0 * (ratio_T_vs_LCDM - 1.0)
diff_M_vs_T    = 100.0 * (ratio_M_vs_T    - 1.0)


# =============================================================================
# Plot 1: LCDM vs new model
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# Top panel: absolute spectra
ax1.loglog(k_hMpc, PkT_lcdm, label="LCDM baseline")
ax1.loglog(k_hMpc, PkM_on,   label="Mine: DRMD + interacting ncdm")
ax1.loglog(k_hMpc, PkT_on, "--", label="Tobias: DRMD + interacting ncdm")
ax1.set_title("LCDM vs DRMD + interacting ncdm")
ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.grid(True, which="both", ls=":")
ax1.legend()

# Bottom panel: deviations from LCDM
ax2.semilogx(k_hMpc, diff_M_vs_LCDM, label="Mine / LCDM - 1")
ax2.semilogx(k_hMpc, diff_T_vs_LCDM, "--", label="Tobias / LCDM - 1")
ax2.axhline(0.0, ls=":")
ax2.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax2.set_ylabel("[%]")
ax2.grid(True, which="both", ls=":")
ax2.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# Plot 2: Mine vs Tobias only for the new model
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.loglog(k_hMpc, PkM_on, label="Mine ON")
ax1.loglog(k_hMpc, PkT_on, "--", label="Tobias ON")
ax1.set_title("DRMD + interacting ncdm: Mine vs Tobias")
ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.grid(True, which="both", ls=":")
ax1.legend()

ax2.semilogx(k_hMpc, diff_M_vs_T, label=r"$100\times(P_{\rm Mine}/P_{\rm Tobias}-1)$")
ax2.axhline(0.0, ls=":")
ax2.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax2.set_ylabel("[%]")
ax2.grid(True, which="both", ls=":")
ax2.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# Printed summary
# =============================================================================
print("\n========================================")
print("Summary")
print("========================================")

print("\nEffect of new model relative to LCDM:")
print("  Mine:   max |P_new / P_LCDM - 1| =",
      np.max(np.abs(ratio_M_vs_LCDM - 1.0)))
print("  Tobias: max |P_new / P_LCDM - 1| =",
      np.max(np.abs(ratio_T_vs_LCDM - 1.0)))

print("\nMine vs Tobias for new model:")
print("  max |Mine_ON / Tobias_ON - 1| =",
      np.max(np.abs(ratio_M_vs_T - 1.0)))
print("  min deviation [%] =",
      np.min(diff_M_vs_T))
print("  max deviation [%] =",
      np.max(diff_M_vs_T))

print("\nInterpretation:")
print("  - First figure shows that the DRMD + interacting-ncdm model differs from standard LCDM.")
print("  - The lower panel quantifies the deviation of each code relative to LCDM.")
print("  - Second figure checks agreement between Mine and Tobias for the new model itself.")
print("  - If Mine and Tobias lie close together while both differ clearly from LCDM,")
print("    then the new physics effect is visible and consistently reproduced in both codes.")

cleanup(cM_lcdm, cT_lcdm, cM_on, cT_on)
print("\nDone.")