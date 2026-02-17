#!/usr/bin/env python3
"""
BARE MINIMUM sanity test for classy_tobias.

Goal:
1) Run a *vanilla* (no interacting sector) cosmology.
2) Plot matter power spectrum P(k) at z=0.

Notes:
- Uses Tobia's parameter style (z_max_pk, ln10^{10}A_s, etc.)
- Avoids ANY interacting keys (N_ncdm_interacting, G_eff_ncdm_interacting, ...)
"""

import numpy as np
import matplotlib.pyplot as plt

import classy_tobias as classy_pp


# ------------------------------------------------------------
# 1) Minimal vanilla ΛCDM + mPk output
# ------------------------------------------------------------
params_vanilla = {
    # outputs: just what we need
    "output": "mPk",
    "z_max_pk": 0.0,          # only need z=0
    "P_k_max_h/Mpc": 5.0,     # max k in h/Mpc for interpolation

    # baseline cosmology
    "H0": 67.5,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "n_s": 0.965,

    # Tobia often uses ln10^{10}A_s (CLASS accepts this)
    "ln10^{10}A_s": 3.0,

    # neutrino content (massless + no massive sector)
    "N_ur": 2.0328,
    "N_ncdm_standard": 0,     # <-- explicitly no massive neutrinos here
}


# ------------------------------------------------------------
# 2) Run CLASS
# ------------------------------------------------------------
cosmo = classy_pp.Class(params_vanilla)
cosmo.compute()


# ------------------------------------------------------------
# 3) Sample P(k) at z=0
#    pk() expects k in 1/Mpc (not h/Mpc), so we convert:
#      k_Mpc = k_hMpc * h
#    and convert P(k) back to (Mpc/h)^3 by multiplying by h^3
# ------------------------------------------------------------
z = 0.0
h = cosmo.h()

k_hMpc = np.logspace(-3, 0.7, 200)  # k in h/Mpc
Pk = np.array([cosmo.pk(kk * h, z) * h**3 for kk in k_hMpc])


# ------------------------------------------------------------
# 4) Plot
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.loglog(k_hMpc, Pk)
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("classy_tobias vanilla mPk (z=0)")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5) Cleanup
# ------------------------------------------------------------
cosmo.struct_cleanup()
cosmo.empty()
