#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from classy_NEDE import Class as ClassMine
from classy_Florian import Class as ClassFlorian


# =============================================================================
# Helpers
# =============================================================================
def pk_safe(cosmo, k_1Mpc_arr, z):
    fn = cosmo.pk_lin if hasattr(cosmo, "pk_lin") else cosmo.pk
    return np.array([fn(float(k), float(z)) for k in k_1Mpc_arr])

def run_model(ClassObj, params, label):
    c = ClassObj()
    c.set(params)
    c.compute()
    print(f"[{label}] computed.")
    try:
        d = c.get_current_derived_parameters(["h", "Omega_m", "Omega_r", "z_eq"])
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

# ON case
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
print("\n========================================")
print("DRMD toggle test: Mine vs Florian")
print("========================================")

print("\n--- Parameters: OFF ---")
for k, v in p_off.items():
    print(f"{k}: {v}")

print("\n--- Parameters: ON ---")
for k, v in p_on.items():
    print(f"{k}: {v}")


# =============================================================================
# Run
# =============================================================================
cM_off = run_model(ClassMine,    p_off, "Mine OFF")
cF_off = run_model(ClassFlorian, p_off, "Florian OFF")

cM_on  = run_model(ClassMine,    p_on,  "Mine ON")
cF_on  = run_model(ClassFlorian, p_on,  "Florian ON")


# Use Florian OFF as reference h for plotting
h_ref  = float(cF_off.h())
k_hMpc = K_1MPC / h_ref

PkM_off = pk_safe(cM_off, K_1MPC, Z_PK) * h_ref**3
PkF_off = pk_safe(cF_off, K_1MPC, Z_PK) * h_ref**3

PkM_on  = pk_safe(cM_on,  K_1MPC, Z_PK) * h_ref**3
PkF_on  = pk_safe(cF_on,  K_1MPC, Z_PK) * h_ref**3

ratio_off = PkM_off / PkF_off
ratio_on  = PkM_on  / PkF_on

diff_off_pct = 100.0 * (ratio_off - 1.0)
diff_on_pct  = 100.0 * (ratio_on  - 1.0)


# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')

ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# OFF absolute
ax1.loglog(k_hMpc, PkM_off, label="Mine OFF")
ax1.loglog(k_hMpc, PkF_off, "--", label="Florian OFF")
ax1.set_title("DRMD OFF: absolute $P(k)$")
ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.grid(True, which="both", ls=":")
ax1.legend()

# ON absolute
ax2.loglog(k_hMpc, PkM_on, label="Mine ON")
ax2.loglog(
    k_hMpc,
    PkF_on,
    "--",
    label=(
        f"Florian ON: "
        f"f_idm={F_IDM_ON}, "
        f"ΔNeff={DELTA_NEFF_ON}, "
        f"z_stop={Z_STOP_ON:.0e}, "
        f"G/(aH)={G_AH_ON:.0e}"
    )
)
ax2.set_title("DRMD ON: absolute $P(k)$")
ax2.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax2.grid(True, which="both", ls=":")
ax2.legend()

# OFF deviation
ax3.semilogx(
    k_hMpc,
    diff_off_pct,
    label=r"$100\times(P_{\rm Mine}^{OFF}/P_{\rm Florian}^{OFF}-1)$"
)
ax3.axhline(0.0, ls=":")
ax3.set_title("DRMD OFF: Mine vs Florian")
ax3.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax3.set_ylabel("[%]")
ax3.grid(True, which="both", ls=":")
ax3.legend()

# ON deviation
ax4.semilogx(
    k_hMpc,
    diff_on_pct,
    label=r"$100\times(P_{\rm Mine}^{ON}/P_{\rm Florian}^{ON}-1)$"
)
ax4.axhline(0.0, ls=":")
ax4.set_title("DRMD ON: Mine vs Florian")
ax4.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax4.set_ylabel("[%]")
ax4.grid(True, which="both", ls=":")
ax4.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# Summary
# =============================================================================
print("\nSummary:")
print("  OFF: max |Mine/Florian - 1| =", np.max(np.abs(ratio_off - 1.0)))
print("  OFF: min diff [%] =", np.min(diff_off_pct))
print("  OFF: max diff [%] =", np.max(diff_off_pct))

print("  ON : max |Mine/Florian - 1| =", np.max(np.abs(ratio_on - 1.0)))
print("  ON : min diff [%] =", np.min(diff_on_pct))
print("  ON : max diff [%] =", np.max(diff_on_pct))

print("\nInterpretation:")
print("  - Top row compares the absolute matter power spectra for Mine and Florian.")
print("  - Bottom row shows the percent deviation of Mine relative to Florian.")
print("  - Compare both OFF and ON to see whether any mismatch is already present")
print("    in the baseline or only appears once DRMD is switched on.")

cleanup(cM_off, cF_off, cM_on, cF_on)
print("\nDone.")