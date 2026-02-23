from classy_tobias import Class as ClassThomas
from classy_NEDE import Class as ClassMine   # <-- change if your wrapper has another name

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Shared LCDM parameters
# -------------------------
params = {
    'output':'tCl,pCl,lCl,mPk',
    'lensing':'yes',
    'H0': 67.5,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'tau_reio': 0.054,
    'ln10^{10}A_s': 3.0,
    'n_s': 0.965
}

# -------------------------
# Thomas
# -------------------------
cosmo_T = ClassThomas()
cosmo_T.set(params)
cosmo_T.compute()

# -------------------------
# Mine (NEDE branch, but LCDM parameters)
# -------------------------
cosmo_M = ClassMine()
cosmo_M.set(params)
cosmo_M.compute()

# -------------------------
# Compute P(k)
# -------------------------
k_hMpc = np.logspace(-3, 0.0, 200)

h_T = cosmo_T.h()
h_M = cosmo_M.h()

Pk_T = np.array([cosmo_T.pk(kk*h_T, 0.0) * h_T**3 for kk in k_hMpc])
Pk_M = np.array([cosmo_M.pk(kk*h_M, 0.0) * h_M**3 for kk in k_hMpc])

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(7,5))
plt.loglog(k_hMpc, Pk_T, label="Thomas")
plt.loglog(k_hMpc, Pk_M, ls="--", label="Mine")

plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("Matter Power Spectrum – LCDM")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

# Optional: relative difference (quick sanity check)
rel = np.max(np.abs(Pk_T - Pk_M) / Pk_T)
print("Max relative difference:", rel)
