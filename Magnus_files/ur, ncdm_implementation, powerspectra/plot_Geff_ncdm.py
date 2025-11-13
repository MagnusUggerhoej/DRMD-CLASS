import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

import matplotlib.pyplot as plt
from classy_NEDE import Class

G_vals = [0.0, 1e-2, 1e0, 1e2]

cls_results = {}

for val in G_vals:
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
        'l_max_scalars': 2500,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'G_eff_ncdm': val
    })
    cosmo.compute()
    cls_results[val] = cosmo.lensed_cl(2500)   # ✅ Store result
    cosmo.struct_cleanup()                     # ✅ Free memory
    print(f"✅ Run successful for G_eff_ncdm = {val:.1e}")

# --- Plot TT spectra ---
plt.figure(figsize=(8, 6))
for val in G_vals:
    ell = cls_results[val]['ell'][2:]
    clTT = cls_results[val]['tt'][2:]
    plt.plot(ell, ell * (ell + 1) * clTT / (2 * 3.14159), label=f"G_eff_ncdm={val:.0e}")

plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$')
plt.legend()
plt.title(r"CMB Temperature Power Spectrum vs $G_{\mathrm{eff,ncdm}}$")
plt.tight_layout()
plt.show()
