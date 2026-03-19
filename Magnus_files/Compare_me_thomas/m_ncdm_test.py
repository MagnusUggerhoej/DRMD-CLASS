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

def fmt_mass(x):
    return f"{x:.0e}"


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

# -----------------------------------------------------------------------------
# Change only these
# -----------------------------------------------------------------------------
REF_MASS   = 1e-2
TEST_MASS_1 = 1e-2
TEST_MASS_2 = 1e-5

# For pure mass-plumbing comparison keep this at 0.
# Set nonzero only if you explicitly want interacting physics switched on.
G_EFF = 0.0


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

        "input_verbose": 0,
    }


# =============================================================================
# Build three apples-to-apples interacting-interface runs
# =============================================================================
# Reference run: interacting interface, reference mass
p_ref = base_params()
p_ref.update({
    "N_ur": float(N_ur_run),
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,

    # Use legacy arrays for mass/deg, exactly like your Mine INT snippet
    "m_ncdm": float(REF_MASS),
    "deg_ncdm": float(DEG),

    "G_eff_ncdm_interacting": float(G_EFF),

    # Keep these explicit since you were already using them
    "ncdm_fluid_approximation": 3,
    "quadrature_strategy_ncdm_interacting": 0,
    "N_momentum_bins_ncdm_interacting": 5,
    "maximum_q_ncdm_interacting": 15.0,

    "f_idm_drmd": 0.0,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
})
for key in ["N_ncdm", "m_ncdm_interacting", "deg_ncdm_interacting", "T_ncdm", "T_ncdm_interacting", "ksi_ncdm_interacting"]:
    p_ref.pop(key, None)

# Test run 1: interacting interface, mass 1
p_test_1 = base_params()
p_test_1.update({
    "N_ur": float(N_ur_run),
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,

    # same interface as p_ref
    "m_ncdm": float(TEST_MASS_1),
    "deg_ncdm": float(DEG),

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
for key in ["N_ncdm", "m_ncdm_interacting", "deg_ncdm_interacting", "T_ncdm", "T_ncdm_interacting", "ksi_ncdm_interacting"]:
    p_test_1.pop(key, None)

# Test run 2: interacting interface, mass 2
p_test_2 = base_params()
p_test_2.update({
    "N_ur": float(N_ur_run),
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,

    # same interface as p_ref
    "m_ncdm": float(TEST_MASS_2),
    "deg_ncdm": float(DEG),

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
for key in ["N_ncdm", "m_ncdm_interacting", "deg_ncdm_interacting", "T_ncdm", "T_ncdm_interacting", "ksi_ncdm_interacting"]:
    p_test_2.pop(key, None)


# =============================================================================
# Print parameters
# =============================================================================
print("\n==============================")
print("Mine-only interacting-interface mass diagnostic")
print("==============================")

print("\n--- Parameters: reference ---")
for k, v in p_ref.items():
    print(f"{k}: {v}")

print("\n--- Parameters: test_1 ---")
for k, v in p_test_1.items():
    print(f"{k}: {v}")

print("\n--- Parameters: test_2 ---")
for k, v in p_test_2.items():
    print(f"{k}: {v}")


# =============================================================================
# Run
# =============================================================================
label_ref    = f"Reference interacting mass = {fmt_mass(REF_MASS)}"
label_test_1 = f"Interacting mass = {fmt_mass(TEST_MASS_1)}"
label_test_2 = f"Interacting mass = {fmt_mass(TEST_MASS_2)}"

c_ref    = run_mine(p_ref,    label_ref)
c_test_1 = run_mine(p_test_1, label_test_1)
c_test_2 = run_mine(p_test_2, label_test_2)

h_ref  = float(c_ref.h())
k_hMpc = K_1MPC / h_ref

Pk_ref    = pk_safe(c_ref,    K_1MPC, Z_PK) * h_ref**3
Pk_test_1 = pk_safe(c_test_1, K_1MPC, Z_PK) * h_ref**3
Pk_test_2 = pk_safe(c_test_2, K_1MPC, Z_PK) * h_ref**3

r_ref    = Pk_ref / Pk_ref
r_test_1 = Pk_test_1 / Pk_ref
r_test_2 = Pk_test_2 / Pk_ref

d_ref_pct    = 100.0 * (r_ref - 1.0)
d_test_1_pct = 100.0 * (r_test_1 - 1.0)
d_test_2_pct = 100.0 * (r_test_2 - 1.0)


# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# Top panel
ax1.loglog(
    k_hMpc,
    Pk_ref,
    label=f"Reference: interacting mass = {fmt_mass(REF_MASS)}"
)
ax1.loglog(
    k_hMpc,
    Pk_test_1,
    "--",
    label=f"Test 1: interacting mass = {fmt_mass(TEST_MASS_1)}"
)
ax1.loglog(
    k_hMpc,
    Pk_test_2,
    "-.",
    label=f"Test 2: interacting mass = {fmt_mass(TEST_MASS_2)}"
)

ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.set_title("Mine-only diagnostic: interacting-interface mass test")
ax1.grid(True, which="both", ls=":")
ax1.legend()

# Bottom panel
ax2.semilogx(
    k_hMpc,
    d_ref_pct,
    ":",
    label=f"Ref({fmt_mass(REF_MASS)}) / Ref({fmt_mass(REF_MASS)})"
)
ax2.semilogx(
    k_hMpc,
    d_test_1_pct,
    "--",
    label=f"Test1({fmt_mass(TEST_MASS_1)}) / Ref({fmt_mass(REF_MASS)})"
)
ax2.semilogx(
    k_hMpc,
    d_test_2_pct,
    "-.",
    label=f"Test2({fmt_mass(TEST_MASS_2)}) / Ref({fmt_mass(REF_MASS)})"
)

ax2.axhline(0.0, ls=":")
ax2.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax2.set_ylabel(r"$100\times(P/P_{\rm ref}-1)$ [%]")
ax2.grid(True, which="both", ls=":")
ax2.legend()


plt.tight_layout()
plt.show()


# =============================================================================
# Summary
# =============================================================================
print("\nSummary:")
print(f"  max |Test1({fmt_mass(TEST_MASS_1)})/Ref({fmt_mass(REF_MASS)}) - 1| =",
      np.max(np.abs(r_test_1 - 1.0)))
print(f"  max |Test2({fmt_mass(TEST_MASS_2)})/Ref({fmt_mass(REF_MASS)}) - 1| =",
      np.max(np.abs(r_test_2 - 1.0)))

print("\nInterpretation:")
print("  - All three runs now use the interacting interface.")
print("  - Changing REF_MASS, TEST_MASS_1, TEST_MASS_2 updates both")
print("    the run parameters and the legend labels automatically.")
print("  - If TEST_MASS_1 == REF_MASS, that curve should lie on zero")
print("    up to numerical precision.")
print("  - Keep G_EFF = 0.0 for a pure mass-plumbing test.")

cleanup(c_ref, c_test_1, c_test_2)
print("\nDone.")