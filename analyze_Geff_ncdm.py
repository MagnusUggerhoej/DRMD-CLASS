import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

import matplotlib.pyplot as plt
from classy_NEDE import Class

lmax = 3000
G_vals = [0.0, 1e2]
cls_results = {}

def compute_cls(Geff):
    cosmo = Class()
    cosmo.set({
        'h': 0.67,
        'Omega_b': 0.05,
        'Omega_cdm': 0.25,
        'A_s': 2.1e-9,
        'n_s': 0.965,
        'tau_reio': 0.054,
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'modes': 's',              # ✅ required for scalar modes
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

# Compute spectra for each coupling
for val in G_vals:
    cls_results[val] = compute_cls(val)

# Reference model
ref_cl = cls_results[0.0]

# --- Plot TT, EE ratios ---
plt.figure(figsize=(8,6))
for val in G_vals:
    ell = cls_results[val]['ell'][2:]
    ratio_TT = cls_results[val]['tt'][2:] / ref_cl['tt'][2:]
    if 'ee' in cls_results[val]:
        ratio_EE = cls_results[val]['ee'][2:] / ref_cl['ee'][2:]
        plt.plot(ell, ratio_EE, '--', label=f'EE ratio (G_eff={val:.0e})')
    plt.plot(ell, ratio_TT, label=f'TT ratio (G_eff={val:.0e})')

plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell / C_\ell(G_{\mathrm{eff}}=0)$')
plt.legend()
plt.title(r"Relative effect of neutrino self-interaction on CMB spectra")
plt.tight_layout()
plt.show()
