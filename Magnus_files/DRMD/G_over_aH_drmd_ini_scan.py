#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from classy_NEDE import Class as ClassMine

'''
his scan asks:

How sensitive is the DRMD signal to the initial interaction strength between interacting dark matter and dark radiation?

Physically:

- small G_over_aH_drmd_ini means the drag is weak from the start,

- large G_over_aH_drmd_ini means the two fluids are tightly coupled early on,

- once you are deep enough in tight coupling, the result should start to saturate.

So this scan is useful for checking whether your usual choice, like 1e9, is already in the saturated regime.

'''


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
    return out

def run_mine(params, label):
    c = ClassMine()
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
# DRMD scan settings
# =============================================================================
# Keep these fixed during the G_over_aH scan
DELTA_NEFF_FIXED = 0.8
F_IDM_FIXED      = 0.03
Z_STOP_FIXED     = 5.0e4

# Scan values
G_AH_VALUES = [1e-3, 1e0, 1e3, 1e6, 1e9]


def fmt_gah(x):
    return f"{x:.0e}"


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
# Build parameter sets
# =============================================================================
param_sets = []
for G_AH in G_AH_VALUES:
    p = base_params()
    p.update({
        "N_ur": float(N_ur_run),
        "N_ncdm": 1,
        "m_ncdm": float(M_NCDM),
        "deg_ncdm": float(DEG),
        "T_ncdm": 0.71611,

        "f_idm_drmd": float(F_IDM_FIXED),
        "delta_Neff_drmd": float(DELTA_NEFF_FIXED),
        "z_stop": float(Z_STOP_FIXED),
        "G_over_aH_drmd_ini": float(G_AH),
    })
    param_sets.append((G_AH, p))


# =============================================================================
# Print scan setup
# =============================================================================
print("\n==============================")
print("DRMD G_over_aH_drmd_ini scan")
print("==============================")
print(f"Fixed delta_Neff_drmd = {DELTA_NEFF_FIXED}")
print(f"Fixed f_idm_drmd      = {F_IDM_FIXED}")
print(f"Fixed z_stop          = {Z_STOP_FIXED}")
print(f"Scanned G/(aH)_ini    = {[fmt_gah(x) for x in G_AH_VALUES]}")

for G_AH, p in param_sets:
    print(f"\n--- Parameters: G_over_aH_drmd_ini = {G_AH} ---")
    for k, v in p.items():
        print(f"{k}: {v}")


# =============================================================================
# Run models
# =============================================================================
models = []
for G_AH, p in param_sets:
    label = f"G_over_aH_drmd_ini = {fmt_gah(G_AH)}"
    c = run_mine(p, label)
    models.append((G_AH, c))


# =============================================================================
# Compute P(k)
# =============================================================================
h_ref = float(models[0][1].h())
k_hMpc = K_1MPC / h_ref

pk_data = {}
for G_AH, c in models:
    pk_data[G_AH] = pk_safe(c, K_1MPC, Z_PK) * h_ref**3

Pk_ref = pk_data[G_AH_VALUES[0]]


# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Top panel
for i, G_AH in enumerate(G_AH_VALUES):
    ax1.loglog(
        k_hMpc,
        pk_data[G_AH],
        ls=linestyles[i % len(linestyles)],
        label=fr"$G/(aH)_{{\rm ini}} = {fmt_gah(G_AH)}$"
    )

ax1.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
ax1.set_title(r"DRMD scan: varying $G/(aH)_{\rm ini}$")
ax1.grid(True, which="both", ls=":")
ax1.legend()

# Bottom panel
for i, G_AH in enumerate(G_AH_VALUES):
    ratio = pk_data[G_AH] / Pk_ref
    diff_pct = 100.0 * (ratio - 1.0)
    ax2.semilogx(
        k_hMpc,
        diff_pct,
        ls=linestyles[i % len(linestyles)],
        label=f"{fmt_gah(G_AH)} / {fmt_gah(G_AH_VALUES[0])}"
    )

ax2.axhline(0.0, ls=":")
ax2.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
ax2.set_ylabel(r"$100\times(P/P_{\rm ref}-1)$ [%]")
ax2.grid(True, which="both", ls=":")
ax2.legend(title=r"$G/(aH)_{\rm ini}$ ratio")

plt.tight_layout()
plt.show()


# =============================================================================
# Summary
# =============================================================================
print(f"\nSummary relative to baseline G_over_aH_drmd_ini = {fmt_gah(G_AH_VALUES[0])}:")
for G_AH in G_AH_VALUES[1:]:
    ratio = pk_data[G_AH] / Pk_ref
    diff_pct = 100.0 * (ratio - 1.0)
    print(
        f"  G_over_aH_drmd_ini = {fmt_gah(G_AH)}: "
        f"max |ratio - 1| = {np.max(np.abs(ratio - 1.0)):.6e}, "
        f"min diff [%] = {np.min(diff_pct):.6f}, "
        f"max diff [%] = {np.max(diff_pct):.6f}"
    )

print("\nInterpretation:")
print("  - This scan isolates the effect of the initial DR-IDM coupling strength.")
print("  - If the curves stop changing much at large G/(aH)_ini, you are in the tight-coupling saturation regime.")
print("  - Then using a very large value such as 1e9 is justified and effectively equivalent to other sufficiently large values.")

cleanup(*[c for _, c in models])
print("\nDone.")