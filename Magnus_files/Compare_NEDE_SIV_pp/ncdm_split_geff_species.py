from classy_NEDE import Class
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1) BASE ΛCDM PARAMETERS (reference model)
# ============================================================

params_lcdm = {
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'H0': 67.5,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'tau_reio': 0.054,
    'ln10^{10}A_s': 3.0,
    'n_s': 0.965
}


# ============================================================
# 2) INTERACTING NCDM MODEL
# ============================================================

params_interacting = params_lcdm.copy()

params_interacting.update({
    'N_ncdm_standard': 0,
    'N_ncdm_interacting': 1,
    'm_ncdm_interacting': 1e0,
    'deg_ncdm_interacting': 3,
    'G_eff_ncdm_interacting': 1e6,
})


# ============================================================
# 3) COMPUTE REFERENCE MODEL
# ============================================================

cosmo_lcdm = Class()
cosmo_lcdm.set(params_lcdm)

print("Computing LCDM...")
cosmo_lcdm.compute()
print("LCDM done.")
bg_lcdm = cosmo_lcdm.get_background()

# look for any rho_ncdm columns
print("LCDM bg keys containing 'ncdm':", [k for k in bg_lcdm.keys() if "ncdm" in k])




# ============================================================
# 4) COMPUTE INTERACTING MODEL
# ============================================================

cosmo_int = Class()
cosmo_int.set(params_interacting)

print("Computing interacting model...")
cosmo_int.compute()
print("Interacting model done.")



# ============================================================
# 5) MATTER POWER SPECTRUM
# ============================================================

k_hMpc = np.logspace(-3, 0.0, 300)

h_lcdm = cosmo_lcdm.h()
h_int  = cosmo_int.h()

Pk_lcdm = np.array([cosmo_lcdm.pk(k * h_lcdm, 0.0) * h_lcdm**3 for k in k_hMpc])
Pk_int  = np.array([cosmo_int.pk(k * h_int,  0.0) * h_int**3  for k in k_hMpc])


# ============================================================
# 6) PLOT BOTH SPECTRA
# ============================================================

plt.figure(figsize=(7, 5))
plt.loglog(k_hMpc, Pk_lcdm, label="ΛCDM")
plt.loglog(k_hMpc, Pk_int, label="Interacting ncdm")
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("Matter Power Spectrum")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()


# ============================================================
# 7) RATIO PLOT (to see wiggles clearly)
# ============================================================

plt.figure(figsize=(7, 4))
plt.plot(k_hMpc, Pk_int / Pk_lcdm)
plt.xscale("log")
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
plt.ylabel(r"$P_{\rm int}/P_{\Lambda CDM}$")
plt.title("Ratio: Interacting / ΛCDM")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()


