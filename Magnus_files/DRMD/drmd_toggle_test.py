#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from classy_NEDE import Class as ClassMine


# =============================================================================
# Helpers
# =============================================================================
def pk_safe(cosmo, k_1Mpc_arr, z):
    fn = cosmo.pk_lin if hasattr(cosmo, "pk_lin") else cosmo.pk
    return np.array([fn(float(k), float(z)) for k in k_1Mpc_arr])

def run_mine(params, label):
    c = ClassMine()
    c.set(params)
    c.compute()
    print(f"[{label}] computed.")
    try:
        d = c.get_current_derived_parameters(
            ["h", "Omega_m", "Omega_r", "z_eq"]
        )
        print(f"[{label}] derived:", d)
    except Exception:
        try:
            print(f"[{label}] h = {c.h():.6f}")
        except Exception:
            pass
    return c

def cleanup(*cosmos):
    for c in cosmos:
        try:
            c.struct_cleanup()
            c.empty()
        except Exception:
            pass


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
# DRMD settings
# =============================================================================
# OFF case
F_IDM_OFF      = 0.0
DELTA_NEFF_OFF = 0.0
Z_STOP_OFF     = 0.0
G_AH_OFF       = 0.0

# ON case (taken close to the shipped DRMD.ini for a visible signal)
F_IDM_ON       = 0.03
DELTA_NEFF_ON  = 0.8
Z_STOP_ON      = 5.0e4
G_AH_ON        = 1.0e9


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

        "input_verbose": 1,
        "background_verbose": 1,
    }


# =============================================================================
# Baseline: DRMD OFF
# =============================================================================
p_off = base_params()
p_off.update({
    "N_ur": float(N_ur_run),
    "N_ncdm": 1,
    "m_ncdm": float(M_NCDM),
    "deg_ncdm": float(DEG),
    "T_ncdm": 0.71611,

    "f_idm_drmd": float(F_IDM_OFF),
    "delta_Neff_drmd": float(DELTA_NEFF_OFF),
    "z_stop": float(Z_STOP_OFF),
    "G_over_aH_drmd_ini": float(G_AH_OFF),
})


# =============================================================================
# DRMD ON
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
# Print params
# =============================================================================
print("\n==============================")
print("DRMD toggle test")
print("==============================")

print("\n--- Parameters: OFF ---")
for k, v in p_off.items():
    print(f"{k}: {v}")

print("\n--- Parameters: ON ---")
for k, v in p_on.items():
    print(f"{k}: {v}")


# =============================================================================
# Run
# =============================================================================
c_off = run_mine(p_off, "DRMD OFF")
c_on  = run_mine(p_on,  "DRMD ON")

h_ref  = float(c_off.h())
k_hMpc = K_1MPC / h_ref

Pk_off = pk_safe(c_off, K_1MPC, Z_PK) * h_ref**3
Pk_on  = pk_safe(c_on,  K_1MPC, Z_PK) * h_ref**3

ratio = Pk_on / Pk_off
diff_pct = 100.0 * (ratio - 1.0)


# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.loglog(
    k_hMpc,
    Pk_off,
    label="DRMD OFF"
)
ax1.loglog(
    k_hMpc,
    Pk_on,
    "--",
    label=(
        f"DRMD ON: "
        f"f_idm={F_IDM_ON}, "
        f"ΔNeff={DELTA_NEFF_ON}, "
        f"z_stop={Z_STOP_ON:.0e}, "
        f"G/(aH)={G_AH_ON:.0e}"
    )
)
ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.set_title("DRMD toggle diagnostic")
ax1.grid(True, which="both", ls=":")
ax1.legend()

ax2.semilogx(
    k_hMpc,
    diff_pct,
    "--",
    label=r"$100\times(P_{\rm ON}/P_{\rm OFF}-1)$"
)
ax2.axhline(0.0, ls=":")
ax2.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax2.set_ylabel("[%]")
ax2.grid(True, which="both", ls=":")
ax2.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# Summary
# =============================================================================
print("\nSummary:")
print("  max |Pk_on/Pk_off - 1| =", np.max(np.abs(ratio - 1.0)))
print("  min diff [%] =", np.min(diff_pct))
print("  max diff [%] =", np.max(diff_pct))

print("\nInterpretation:")
print("  - If DRMD ON differs visibly from DRMD OFF, the DRMD path is alive.")
print("  - This first script is only a toggle/plumbing test.")
print("  - Next step is a one-parameter scan, keeping the others fixed.")

cleanup(c_off, c_on)
print("\nDone.")