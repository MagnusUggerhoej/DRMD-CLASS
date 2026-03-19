#!/usr/bin/env python3

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

DEG      = 1.0
M_NCDM   = 0.06
N_ur_run = max(0.0, 3.046 - DEG)


# =============================================================================
# DRMD ON test point
# =============================================================================
DELTA_NEFF_ON = 0.8
F_IDM_ON      = 0.03
Z_STOP_ON     = 5.0e4
G_AH_ON       = 1.0e9


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
# DRMD ON params only
# =============================================================================
p_on = base_params()
p_on.update({
    "N_ur": float(N_ur_run),
    "N_ncdm": 1,
    "m_ncdm": float(M_NCDM),
    "deg_ncdm": float(DEG),
    "T_ncdm": 0.71611,

    "f_idm_drmd": float(F_IDM_ON),
    "delta_Neff_drmd": float(DELTA_NEFF_ON),
    "z_stop": float(Z_STOP_ON),
    "G_over_aH_drmd_ini": float(G_AH_ON),
})


# =============================================================================
# Print setup
# =============================================================================
print("\n========================================")
print("Mine vs Tobias DRMD comparison (ON only)")
print("========================================")

print_param_block("DRMD ON", p_on)


# =============================================================================
# Run ON in both codes
# =============================================================================
cM_on = run_model(ClassMine,   p_on, "Mine   ON")
cT_on = run_model(ClassTobias, p_on, "Tobias ON")


# =============================================================================
# Compute P(k)
# =============================================================================
h_ref  = float(cT_on.h())
k_hMpc = K_1MPC / h_ref

PkM_on = pk_safe(cM_on, K_1MPC, Z_PK) * h_ref**3
PkT_on = pk_safe(cT_on, K_1MPC, Z_PK) * h_ref**3

ratio_on = PkM_on / PkT_on
diff_pct = 100.0 * (ratio_on - 1.0)


# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# Top panel: absolute spectra
ax1.loglog(k_hMpc, PkM_on, label="Mine ON")
ax1.loglog(k_hMpc, PkT_on, "--", label="Tobias ON")
ax1.set_title("DRMD ON: absolute $P(k)$")
ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.grid(True, which="both", ls=":")
ax1.legend()

# Bottom panel: percent deviation
ax2.semilogx(k_hMpc, diff_pct, label=r"$100\times(P_{\rm Mine}/P_{\rm Tobias}-1)$")
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

print("  max |Mine_ON / Tobias_ON - 1| =",
      np.max(np.abs(ratio_on - 1.0)))
print("  min deviation [%] =",
      np.min(diff_pct))
print("  max deviation [%] =",
      np.max(diff_pct))

print("\nInterpretation:")
print("  - This compares the DRMD-on matter power spectrum directly between Mine and Tobias.")
print("  - The lower panel shows the percent deviation of Mine relative to Tobias.")
print("  - A curve close to zero means the two implementations agree well at this DRMD point.")

cleanup(cM_on, cT_on)
print("\nDone.")