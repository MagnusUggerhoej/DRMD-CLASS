import numpy as np
import matplotlib.pyplot as plt

# >>> CHANGE THIS if your wrapper module name differs
import classy_NEDE as classy_pp

# ----------------------------
# File with Tobias output
# ----------------------------
TOBIAS_FILE = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/Magnus_files/tobias comparison/mPk_tobias.txt"

# ----------------------------
# Settings
# ----------------------------
z_plot = 3.0
kmax = 1.0
Nk = 1000
klist = np.logspace(-4, np.log10(kmax), Nk)

def mPk(perts, z):
    """Return P(k,z) on klist (k in 1/Mpc), using CLASS convention pk(k*h, z)*h^3."""
    h = perts.h()
    return np.array([perts.pk(k * h, z) * h**3 for k in klist])

# ----------------------------
# GENERAL / reference model
# IMPORTANT: do NOT use the split keys here if total is 0
# ----------------------------
dict_general = {
    'k_output_values': '0.05, 0.002',
    'output': 'tCl,pCl,lCl,mPk,mTk',
    'lensing': 'yes',
    'start_large_k_at_tau_h_over_tau_k': 0.01,
    'perturbations_verbose': '0',
    'thermodynamics_verbose': '0',
    'z_max_pk': 4.,
    'H0': 67.5,
    'N_ur': 2.0328,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'tau_reio': 0.054,
    'ln10^{10}A_s': 3,
    'l_max_scalars': 2500,
    'n_s': 0.965,
    # NOTE: NO N_ncdm_standard / N_ncdm_interacting here if total is 0
}

# ----------------------------
# SPECIFIC model (DRMD + interacting ncdm)
# ----------------------------
dict_specific = {
    **dict_general,  # inherit baseline
    # Interacting NCDM parameters (split form)
    'N_ncdm_standard': 0,
    'N_ncdm_interacting': 1,
    'deg_ncdm_interacting': 3,
    'm_ncdm_interacting': 1e-2,
    'ncdm_fluid_approximation': 3,      # match Tobias (he uses 3)
    'G_eff_ncdm_interacting': 1e-3,

    # DRMD parameters
    "G_over_aH_drmd_ini": 1e-3,
    'delta_Neff_drmd': 0.1,
    'f_idm_drmd': 0.1,
    'z_stop': 5000,
}

# ----------------------------
# Run your models
# ----------------------------
model_spec = classy_pp.Class()
model_spec.set(dict_specific)
model_spec.compute()

model_gen = classy_pp.Class()
model_gen.set(dict_general)
model_gen.compute()

pk_spec = mPk(model_spec, z_plot)
pk_gen  = mPk(model_gen,  z_plot)
ratio_me = pk_spec / pk_gen

# Clean up CLASS objects (optional but good practice)
model_spec.struct_cleanup()
model_spec.empty()

model_gen.struct_cleanup()
model_gen.empty()

# ----------------------------
# Load Tobias file and compute his ratio
# ----------------------------
tob = np.loadtxt(TOBIAS_FILE)
k_tob = tob[:, 0]
pk_spec_tob = tob[:, 1]
pk_gen_tob = tob[:, 2]
ratio_tob = pk_spec_tob / pk_gen_tob

# ----------------------------
# Plot comparison
# ----------------------------
fig, ax = plt.subplots()
ax.plot(klist, ratio_me, label=f"Me (z={z_plot})")
ax.plot(k_tob, ratio_tob, label="Tobias (from file)")
ax.set(xscale='log', xlabel="k [1/Mpc]", ylabel="Pk_specific / Pk_general")
ax.set_title(f"Tobias-style ratio comparison (z={z_plot} used in pk())")
ax.legend()
plt.show()
