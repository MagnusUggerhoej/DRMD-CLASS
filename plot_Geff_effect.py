from classy import Class
import matplotlib.pyplot as plt
import numpy as np

G_eff_values = [0.0, 1e-2, 1e+0, 1e+20]

base = {
    'h': 0.67,
    'T_cmb': 2.7255,
    'Omega_b': 0.05,
    'Omega_cdm': 0.25,
    'N_ur': 3.046,
    'output': 'tCl',          # temperature Cls
    'l_max_scalars': 2500,
    # primordial power params (needed for Cls)
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.054,
}

res = []
for G in G_eff_values:
    pars = base.copy()
    pars['G_eff_ur'] = G
    cosmo = Class()
    cosmo.set(pars)
    cosmo.compute()
    cl = cosmo.raw_cl(2500)       # <- unlensed
    res.append((G, cl))
    cosmo.struct_cleanup()
    cosmo.empty()

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
for G, cl in res:
    ell = cl['ell'][2:]
    DlTT = cl['tt'][2:] * ell*(ell+1)/(2*np.pi)
    plt.plot(ell, DlTT, label=fr'$G_{{\rm eff}}={G:.0e}$')

plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$\ell$'); plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi\ [\mu{\rm K}^2]$')
plt.title(r'Unlensed $TT$: effect of $G_{\rm eff}$'); plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.tight_layout(); plt.show()
