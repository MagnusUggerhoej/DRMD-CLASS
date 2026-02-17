from classy_NEDE import Class
import matplotlib.pyplot as plt
import numpy as np
cosmo = Class()
cosmo.set({
    'output':'tCl,pCl,lCl,mPk',
    'lensing':'yes',
    'H0': 67.5,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'tau_reio': 0.054,
    'ln10^{10}A_s': 3.0,
    'n_s': 0.965
})
cosmo.compute()

plt.figure(figsize=(7, 5))
k_hMpc = np.logspace(-3, 0.7, 200)  # k in h/Mpc
h = cosmo.h()
Pk = np.array([cosmo.pk(kk * h, 0.0) * h**3 for kk in k_hMpc])
plt.loglog(k_hMpc, Pk)
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("Matter Power Spectrum LCDM")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

