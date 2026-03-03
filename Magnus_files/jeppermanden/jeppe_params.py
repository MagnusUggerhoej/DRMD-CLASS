#!/usr/bin/env python3

from classy_NEDE import Class

# Instantiate CLASS
cosmo = Class()

# Set ONLY your parameter set
cosmo.set({
    'output': 'tCl,pCl,lCl,mPk',
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'h': 0.67,
    'A_s': 2.1e-9,
    'n_s': 0.96,
    'tau_reio': 0.06,

    'N_ncdm_interacting': 1,
    'm_ncdm_interacting': 0.1,
    'deg_ncdm_interacting': 1,
    'log10G_eff_ncdm_interacting': -2,

    'N_ncdm_standard': 2.0,
    'm_ncdm_standard': 0.0,
    'deg_ncdm_standard': 1,

    'N_ur': 0.0,

    'r': 0.1,
    'modes': 's,t',
    'k_pivot': 0.05,

    'ncdm_fluid_approximation': 3,

    'P_k_max_h/Mpc': 1.0,
    'lensing': 'yes'
})

# Run computation
cosmo.compute()

# Optional sanity print
try:
    derived = cosmo.get_current_derived_parameters(
        ["h", "Omega_m", "Omega_r", "sigma8", "Neff"]
    )
    print("Derived parameters:", derived)
except:
    print("Computation finished successfully.")

# Clean up (good practice)
cosmo.struct_cleanup()
cosmo.empty()

print("Done.")