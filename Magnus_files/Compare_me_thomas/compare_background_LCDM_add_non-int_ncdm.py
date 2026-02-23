"""
This script performs a controlled consistency check between two independent
CLASS-based cosmology implementations: a reference CLASS build (Thomas)
and a modified NEDE / interacting-ncdm branch (Mine).

It runs both codes with identical ΛCDM + 1 massive neutrino parameters
(interactions explicitly disabled) to verify that the modified branch
reproduces standard free-streaming ncdm physics.

The script evaluates the linear matter power spectrum P(k) at z=0,
sampling the same physical k [1/Mpc] grid in both runs to avoid h-scaling artifacts.

It then converts both spectra to a common (Mpc/h)^3 convention and
computes the ratio Mine/Thomas across k.

Agreement at the ~1e-3 level or better confirms that the modified
ncdm implementation reduces correctly to standard ΛCDM in the
non-interacting limit and preserves baseline cosmological behavior.
"""



from classy_NEDE_jan import Class as ClassMine #if swithcing to feb version (CLASS_NEDE), remember changes in input so that you need to explicitly give the splitting. =>input_read_parameters_species(L:2860) :condition (flag_legacy_ncdm == _TRUE_) is true; Do not set both N_ncdm and N_ncdm_standard/N_ncdm_interacting. 
from classy import Class as ClassThomas

import numpy as np
import matplotlib.pyplot as plt

LOG10_G_EFF_ZERO = -50.0  # effectively zero

def common_base(lmax=2500):
    return {
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "A_s": 2.1e-9,
        "n_s": 0.9649,
        "tau_reio": 0.0543,

        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": lmax,

        "input_verbose": 2,
        "background_verbose": 1,


        # ensure pk table exists and is wide enough
        "P_k_max_1/Mpc": 10.0,

        "T_cmb": 2.7255,
        "Omega_k": 0.0,

        "N_ur": 3.046,
        "N_ncdm": 1,
        "m_ncdm": 0.06,
    }

def run(Cls, p, label):
    print(f"Running {label}...")
    c = Cls()
    c.set(p)
    c.compute()
    return c

def pk_safe(cosmo, k, z):
    # avoid nonlinear path if pk_lin exists
    if hasattr(cosmo, "pk_lin"):
        return cosmo.pk_lin(k, z)
    return cosmo.pk(k, z)

params_thomas = common_base(lmax=2500)

params_mine = common_base(lmax=2500)
params_mine["N_ncdm_standard"] = 1
params_mine["N_ncdm_interacting"] = 0
params_mine["log10_G_eff_ur"] = LOG10_G_EFF_ZERO
params_mine["log10_G_eff_ncdm"] = LOG10_G_EFF_ZERO

cosmo_T = run(ClassThomas, params_thomas, "Thomas")
cosmo_M = run(ClassMine,   params_mine,   "Mine")

# same physical k [1/Mpc]
k_1Mpc = np.logspace(-4, 1.0, 250)

Pk_T_1Mpc = np.array([pk_safe(cosmo_T, k, 0.0) for k in k_1Mpc])
Pk_M_1Mpc = np.array([pk_safe(cosmo_M, k, 0.0) for k in k_1Mpc])

h_ref = cosmo_T.h()
k_hMpc = k_1Mpc / h_ref
Pk_T = Pk_T_1Mpc * h_ref**3
Pk_M = Pk_M_1Mpc * h_ref**3

plt.figure(figsize=(7, 5))
plt.loglog(k_hMpc, Pk_T, label="Thomas")
plt.loglog(k_hMpc, Pk_M, ls="--", label="Mine")
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("P(k) – LCDM + 1 massive ncdm (linear)")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

ratio = Pk_M / Pk_T
plt.figure(figsize=(7, 3.8))
plt.semilogx(k_hMpc, ratio)
plt.axhline(1.0, ls=":")
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
plt.ylabel("Mine / Thomas")
plt.title("P(k) ratio")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

print("h(Thomas) =", cosmo_T.h(), " h(Mine) =", cosmo_M.h())
print("max |ratio-1| =", np.max(np.abs(ratio - 1)))

cosmo_T.struct_cleanup(); cosmo_T.empty()
cosmo_M.struct_cleanup(); cosmo_M.empty()
