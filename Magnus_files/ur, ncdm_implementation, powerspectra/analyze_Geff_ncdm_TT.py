import os, sys
# Absolute path to your DRMD-CLASS/python directory:
classy_path = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/python"
sys.path.insert(0, classy_path)

import matplotlib.pyplot as plt
from classy_NEDE import Class

# --- Parameters ---
lmax = 3000
G_vals = [0.0, 1e-5, 1e-3, 1e2]   # You can adjust as needed
cls_results = {}

def compute_cls(Geff):
    """Compute lensed CMB Cls for a given G_eff_ncdm"""
    cosmo = Class()
    cosmo.set({
        'h': 0.67,
        'Omega_b': 0.05,
        'Omega_cdm': 0.25,
        'A_s': 2.1e-9,
        'n_s': 0.965,
        'tau_reio': 0.054,
        'output': 'tCl,lCl',
        'lensing': 'yes',
        'modes': 's',                # Scalars only
        'l_max_scalars': lmax,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'G_eff_ncdm': Geff
    })
    cosmo.compute()
    cl = cosmo.lensed_cl(lmax)
    cosmo.struct_cleanup()
    print(f"✅ Run successful for G_eff_ncdm = {Geff:.1e}")
    return cl

# --- Compute spectra for each coupling ---
for val in G_vals:
    cls_results[val] = compute_cls(val)

# --- Reference model (free-streaming neutrinos) ---
ref_cl = cls_results[0.0]

# --- Plot only TT ratio ---
plt.figure(figsize=(8, 6))
for val in G_vals:
    if val == 0.0:
        continue  # Skip ratio to itself
    ell = cls_results[val]['ell'][2:]
    ratio_TT = cls_results[val]['tt'][2:] / ref_cl['tt'][2:]
    plt.plot(ell, ratio_TT, label=fr'$G_{{\mathrm{{eff}}}}={val:.0e}$')

plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{TT}(G_{\mathrm{eff}})/C_\ell^{TT}(G_{\mathrm{eff}}=0)$')
plt.title(r"Effect of neutrino self-interaction on CMB Temperature Spectrum")
plt.legend(title="Interaction strength")
plt.tight_layout()
plt.show()
