from classy_NEDE_jan import Class as ClassMine
from classy import Class as ClassThomas

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Shared LCDM parameters (from your base_params)
# -------------------------
def base_params(lmax: int) -> dict:
    p = {
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "A_s": 2.1e-9,
        "n_s": 0.9649,
        "tau_reio": 0.0543,

        # need mPk since we call pk()
        "output": "tCl,lCl,pCl,mPk",
        "lensing": "yes",
        "l_max_scalars": lmax,

        "T_cmb": 2.7255,
        "Omega_k": 0.0,

        # keep these only if K_PIVOT / RECOMBINATION / REIO_PARAM are defined in your script/env
        # "k_pivot": K_PIVOT,
        # "recombination": RECOMBINATION,
        # "reio_parametrization": REIO_PARAM,
    }
    return p

params = base_params(lmax=2500)

# -------------------------
# Thomas
# -------------------------
print("Running Thomas's CLASS...")
cosmo_T = ClassThomas()

derived_T = cosmo_T.get_current_derived_parameters(['h', 'Omega_m'])
print("Thomas derived parameters:", derived_T)

cosmo_T.set(params)
cosmo_T.compute()


# -------------------------
# Mine
# -------------------------
print("Running my CLASS...")
cosmo_M = ClassMine()

derived_M = cosmo_M.get_current_derived_parameters(['h', 'Omega_m'])
print("Mine derived parameters:", derived_M)

cosmo_M.set(params)
cosmo_M.compute()

# -------------------------
# Compute P(k)
# -------------------------
k_hMpc = np.logspace(-3, 0.0, 200)

h_T = cosmo_T.h()
h_M = cosmo_M.h()

Pk_T = np.array([cosmo_T.pk(kk * h_T, 0.0) * h_T**3 for kk in k_hMpc])
Pk_M = np.array([cosmo_M.pk(kk * h_M, 0.0) * h_M**3 for kk in k_hMpc])

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(7, 5))
plt.loglog(k_hMpc, Pk_T, label="Thomas")
plt.loglog(k_hMpc, Pk_M, ls="--", label="Mine")

plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("Matter Power Spectrum – LCDM")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

rel = np.max(np.abs(Pk_T - Pk_M) / Pk_T)
print("Max relative difference:", rel)

# optional cleanup (avoids memory issues in repeated runs)
cosmo_T.struct_cleanup(); cosmo_T.empty()
cosmo_M.struct_cleanup(); cosmo_M.empty()
