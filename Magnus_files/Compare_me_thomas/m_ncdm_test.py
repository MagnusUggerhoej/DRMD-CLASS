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
        d = c.get_current_derived_parameters(["h", "Omega_m", "Omega_r"])
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
# Shared setup
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
N_ur_run = max(0.0, 3.046 - DEG)
G_EFF    = 1e-3

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

        # turn on your new debug prints in input.c
        "input_verbose": 1,
    }


# 


# =============================================================================
# Case A: legacy mass input (reference)
# =============================================================================
p_legacy = base_params()
p_legacy.update({
    "N_ur": float(N_ur_run),
    "N_ncdm": 1,
    "m_ncdm": 1e-2,
    "deg_ncdm": float(DEG),
    "T_ncdm": 0.71611,
})


# ============================================================================= 
# Case B: split interacting, target mass = 1e-2
# =============================================================================
p_split_hi = base_params()
p_split_hi.update({
    "N_ur": float(N_ur_run),
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,

    "m_ncdm_interacting": 1e-2,
    "deg_ncdm_interacting": float(DEG),
    "T_ncdm_interacting": 0.71611,
    "ksi_ncdm_interacting": 0.0,

    "G_eff_ncdm_interacting": float(G_EFF),
    "ncdm_fluid_approximation": 3,
    "quadrature_strategy_ncdm_interacting": 0,
    "N_momentum_bins_ncdm_interacting": 5,
    "maximum_q_ncdm_interacting": 15.0,

    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
})
p_split_hi.pop("N_ncdm", None)


# =============================================================================
# Case C: split interacting, control mass = 1e-5
# =============================================================================
p_split_lo = base_params()
p_split_lo.update({
    "N_ur": float(N_ur_run),
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,

    "m_ncdm_interacting": 1e-5,
    "deg_ncdm_interacting": float(DEG),
    "T_ncdm_interacting": 0.71611,
    "ksi_ncdm_interacting": 0.0,

    "G_eff_ncdm_interacting": float(G_EFF),
    "ncdm_fluid_approximation": 3,
    "quadrature_strategy_ncdm_interacting": 0,
    "N_momentum_bins_ncdm_interacting": 5,
    "maximum_q_ncdm_interacting": 15.0,

    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
})
p_split_lo.pop("N_ncdm", None)


# =============================================================================
# Run
# =============================================================================
print("\n==============================")
print("Mine-only mass diagnostic")
print("==============================")

print("\n--- Parameters: legacy ---")
for k, v in p_legacy.items():
    print(f"{k}: {v}")

print("\n--- Parameters: split_hi ---")
for k, v in p_split_hi.items():
    print(f"{k}: {v}")

print("\n--- Parameters: split_lo ---")
for k, v in p_split_lo.items():
    print(f"{k}: {v}")

c_legacy   = run_mine(p_legacy,   "Legacy m_ncdm = 1e-2")
c_split_hi = run_mine(p_split_hi, "Split m_ncdm_interacting = 1e-2")
c_split_lo = run_mine(p_split_lo, "Split m_ncdm_interacting = 1e-5")

h_ref  = float(c_legacy.h())
k_hMpc = K_1MPC / h_ref

Pk_legacy   = pk_safe(c_legacy,   K_1MPC, Z_PK) * h_ref**3
Pk_split_hi = pk_safe(c_split_hi, K_1MPC, Z_PK) * h_ref**3
Pk_split_lo = pk_safe(c_split_lo, K_1MPC, Z_PK) * h_ref**3

r_hi = Pk_split_hi / Pk_legacy
r_lo = Pk_split_lo / Pk_legacy


# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)
 
ax1.loglog(k_hMpc, Pk_legacy,   label="Legacy: m_ncdm = 1e-2")
ax1.loglog(k_hMpc, Pk_split_hi, "--", label="Split: m_ncdm_interacting = 1e-2")
ax1.loglog(k_hMpc, Pk_split_lo, "-.", label="Split: m_ncdm_interacting = 1e-5")
ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.set_title("Mine-only diagnostic: does m_ncdm_interacting change the physics?")
ax1.grid(True, which="both", ls=":")
ax1.legend()

ax2.semilogx(k_hMpc, r_hi, "--", label="Split(1e-2) / Legacy(1e-2)")
ax2.semilogx(k_hMpc, r_lo, "-.", label="Split(1e-5) / Legacy(1e-2)")
ax2.axhline(1.0, ls=":")
ax2.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax2.set_ylabel("ratio")
ax2.grid(True, which="both", ls=":")
ax2.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# Print summary
# =============================================================================
print("\nSummary:")
print("  max |Split(1e-2)/Legacy(1e-2) - 1| =",
      np.max(np.abs(r_hi - 1.0)))
print("  max |Split(1e-5)/Legacy(1e-2) - 1| =",
      np.max(np.abs(r_lo - 1.0)))

print("\nInterpretation:")
print("  - Watch the terminal for:")
print("      DEBUG before split(ncdm_props)")
print("      DEBUG m override flags")
print("      DEBUG after split(ncdm_props)")
print("  - If split(1e-2) and split(1e-5) behave the same, m_ncdm_interacting is likely ignored.")
print("  - If DEBUG after split(ncdm_props) still prints m=1e-5 when you pass 1e-2, the overwrite bug is confirmed.")

cleanup(c_legacy, c_split_hi, c_split_lo)
print("\nDone.")