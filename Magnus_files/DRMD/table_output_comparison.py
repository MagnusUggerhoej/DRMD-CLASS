#!/usr/bin/env python3

import numpy as np

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

def pct_diff(a, b):
    return 100.0 * (a / b - 1.0)

def fmt_float(x, ndp=6):
    return f"{x:.{ndp}f}"

def fmt_sci(x, ndp=6):
    return f"{x:.{ndp}e}"

def print_separator(width=142):
    print("-" * width)


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

Z_PK   = 0.0
L_MAX  = 2500

DEG_NCDM  = 1.0
M_NCDM_EV = 0.06
T_NCDM    = 0.71611
N_ur_run  = max(0.0, 3.046 - DEG_NCDM)


# =============================================================================
# Benchmark model: DRMD + interacting ncdm
# =============================================================================
DELTA_NEFF_ON = 0.8
F_IDM_ON      = 0.03
Z_STOP_ON     = 5.0e4
G_AH_ON       = 1.0e9
G_EFF_NCDM_ON = 1e-1


# =============================================================================
# k values to print
# =============================================================================
# These are physical k in 1/Mpc.
K_PRINT_1MPC = np.array([
    1e-3,
    1e-2,
    5e-2,
    1e-1,
    2e-1,
    5e-1,
    1.0,
], dtype=float)


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
        "P_k_max_1/Mpc": float(np.max(K_PRINT_1MPC) * 1.20),

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
print("\n============================================================")
print("Table benchmark: LCDM vs DRMD + interacting ncdm")
print("============================================================")

print_param_block("Mine   LCDM", pM_lcdm)
print_param_block("Tobias LCDM", pT_lcdm)
print_param_block("Mine   ON",   pM_on)
print_param_block("Tobias ON",   pT_on)

print("\nSelected k values [1/Mpc]:")
print("  " + ", ".join(fmt_sci(k, 2) for k in K_PRINT_1MPC))


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
h_ref = float(cT_lcdm.h())

PkM_lcdm = pk_safe(cM_lcdm, K_PRINT_1MPC, Z_PK) * h_ref**3
PkT_lcdm = pk_safe(cT_lcdm, K_PRINT_1MPC, Z_PK) * h_ref**3
PkM_on   = pk_safe(cM_on,   K_PRINT_1MPC, Z_PK) * h_ref**3
PkT_on   = pk_safe(cT_on,   K_PRINT_1MPC, Z_PK) * h_ref**3

k_print_hMpc = K_PRINT_1MPC / h_ref


# =============================================================================
# Print derived summary table
# =============================================================================
print("\n")
print("=" * 92)
print("Derived quantities")
print("=" * 92)

dM_lcdm = get_derived_safe(cM_lcdm)
dT_lcdm = get_derived_safe(cT_lcdm)
dM_on   = get_derived_safe(cM_on)
dT_on   = get_derived_safe(cT_on)

rows = [
    ("h",        dM_lcdm.get("h", np.nan),     dT_lcdm.get("h", np.nan),     dM_on.get("h", np.nan),     dT_on.get("h", np.nan)),
    ("Omega_m",  dM_lcdm.get("Omega_m", np.nan), dT_lcdm.get("Omega_m", np.nan), dM_on.get("Omega_m", np.nan), dT_on.get("Omega_m", np.nan)),
    ("z_eq",     dM_lcdm.get("z_eq", np.nan),  dT_lcdm.get("z_eq", np.nan),  dM_on.get("z_eq", np.nan),  dT_on.get("z_eq", np.nan)),
    ("Neff",     dM_lcdm.get("Neff", np.nan),  dT_lcdm.get("Neff", np.nan),  dM_on.get("Neff", np.nan),  dT_on.get("Neff", np.nan)),
]

print(f"{'Quantity':<14} {'Mine LCDM':>18} {'Tobias LCDM':>18} {'Mine ON':>18} {'Tobias ON':>18}")
print_separator(92)
for name, a, b, c, d in rows:
    def f(x):
        return "n/a" if np.isnan(x) else fmt_float(x, 6)
    print(f"{name:<14} {f(a):>18} {f(b):>18} {f(c):>18} {f(d):>18}")


# =============================================================================
# Print main comparison table
# =============================================================================
print("\n")
print("=" * 162)
print("Matter power spectrum comparison table")
print("=" * 162)
print(
    f"{'k [1/Mpc]':>11} "
    f"{'k [h/Mpc]':>11} "
    f"{'P_LCDM(Mine)':>16} "
    f"{'P_LCDM(Tob)':>16} "
    f"{'P_ON(Mine)':>16} "
    f"{'P_ON(Tob)':>16} "
    f"{'Mine/LCDM-1 [%]':>16} "
    f"{'Tob/LCDM-1 [%]':>16} "
    f"{'Mine/Tob-1 [%]':>15}"
)
print_separator(162)

for i, k in enumerate(K_PRINT_1MPC):
    print(
        f"{fmt_sci(k, 2):>11} "
        f"{fmt_float(k_print_hMpc[i], 6):>11} "
        f"{fmt_sci(PkM_lcdm[i], 6):>16} "
        f"{fmt_sci(PkT_lcdm[i], 6):>16} "
        f"{fmt_sci(PkM_on[i], 6):>16} "
        f"{fmt_sci(PkT_on[i], 6):>16} "
        f"{fmt_float(pct_diff(PkM_on[i], PkT_lcdm[i]), 6):>16} "
        f"{fmt_float(pct_diff(PkT_on[i], PkT_lcdm[i]), 6):>16} "
        f"{fmt_float(pct_diff(PkM_on[i], PkT_on[i]), 6):>15}"
    )


# =============================================================================
# Compact supervisor-friendly summary
# =============================================================================
mine_vs_lcdm = PkM_on / PkT_lcdm - 1.0
tob_vs_lcdm  = PkT_on / PkT_lcdm - 1.0
mine_vs_tob  = PkM_on / PkT_on   - 1.0

print("\n")
print("=" * 92)
print("Compact summary")
print("=" * 92)
print(f"Using Tobias LCDM as displayed LCDM baseline and h_ref = {h_ref:.6f}")
print(f"max |Mine ON / Tobias ON - 1|      = {np.max(np.abs(mine_vs_tob)):.6e}")
print(f"max |Mine ON / LCDM - 1|           = {np.max(np.abs(mine_vs_lcdm)):.6e}")
print(f"max |Tobias ON / LCDM - 1|         = {np.max(np.abs(tob_vs_lcdm)):.6e}")

imax_mt = int(np.argmax(np.abs(mine_vs_tob)))
imax_ml = int(np.argmax(np.abs(mine_vs_lcdm)))
imax_tl = int(np.argmax(np.abs(tob_vs_lcdm)))

print("\nWhere the largest effects occur among the selected k values:")
print(f"  Largest Mine/Tobias mismatch at k = {K_PRINT_1MPC[imax_mt]:.3e} 1/Mpc "
      f"({k_print_hMpc[imax_mt]:.6f} h/Mpc), "
      f"deviation = {100.0 * mine_vs_tob[imax_mt]:.6f}%")
print(f"  Largest Mine/LCDM effect   at k = {K_PRINT_1MPC[imax_ml]:.3e} 1/Mpc "
      f"({k_print_hMpc[imax_ml]:.6f} h/Mpc), "
      f"deviation = {100.0 * mine_vs_lcdm[imax_ml]:.6f}%")
print(f"  Largest Tobias/LCDM effect at k = {K_PRINT_1MPC[imax_tl]:.3e} 1/Mpc "
      f"({k_print_hMpc[imax_tl]:.6f} h/Mpc), "
      f"deviation = {100.0 * tob_vs_lcdm[imax_tl]:.6f}%")

print("\nInterpretation:")
print("  - The table gives exact P(k) values at selected physically relevant k values.")
print("  - Mine/LCDM and Tobias/LCDM quantify how strongly the new model differs from standard LCDM.")
print("  - Mine/Tobias quantifies code-to-code agreement for the new model.")
print("  - This is the table version of the benchmark plots, suitable for direct validation discussions.")

cleanup(cM_lcdm, cT_lcdm, cM_on, cT_on)
print("\nDone.")