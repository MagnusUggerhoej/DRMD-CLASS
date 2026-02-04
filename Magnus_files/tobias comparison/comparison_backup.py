import os
import classy_NEDE as classy_pp

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Output folder
# Saves next to this script, always.
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "comparison_data")
os.makedirs(OUT_DIR, exist_ok=True)

print("Running from (cwd):", os.getcwd())
print("Script location:", BASE_DIR)
print("Saving outputs to:", OUT_DIR)


dict_general={
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk,mTk',
        'lensing':'yes',
        'start_large_k_at_tau_h_over_tau_k': 0.01,
        'perturbations_verbose':'0',
        'thermodynamics_verbose':'0',
        'z_max_pk': 4.,
        'H0': 67.5,
        'N_ur' : 2.0328,
        'N_ncdm_standard' : 0,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,    
    }

dict_specific = {
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk,mTk',
        'lensing':'yes',
        'start_large_k_at_tau_h_over_tau_k': 0.01,
        'perturbations_verbose':'0',
        'thermodynamics_verbose':'0',
        'z_max_pk': 4.,
        'H0': 67.5,
        'N_ur' : 2.0328,
        'N_ncdm_standard' : 0,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,

        # Interacting NCDM parameters start (Tobias names)
        'N_ncdm_interacting': 1,
        'deg_ncdm_interacting': 3,
        'm_ncdm_interacting': 1e-2,
        'ncdm_fluid_approximation': 3,
        'G_eff_ncdm_interacting': 1e-3,
        # Interacting NCDM parameters stop

        # DRMD parameters start
        "G_over_aH_drmd_ini":1e-3,
        'delta_Neff_drmd':0.1,
        'f_idm_drmd':0.1,
        'z_stop':5000,
        # DRMD parameters end
    }


# ==========================================================
# Translate Tobias parameter names -> names used in *my* CLASS
# ==========================================================
# Your code uses ppt->G_eff_ncdm (NOT G_eff_ncdm_interacting)
dict_specific['G_eff_ncdm'] = dict_specific.get('G_eff_ncdm_interacting', 0.0)

# Optional: warn if Tobias-only keys are present but not supported by your build
# (These keys are kept for readability, but may be ignored by your CLASS.)
UNUSED_KEYS = [
    'N_ncdm_interacting',
    'deg_ncdm_interacting',
    'm_ncdm_interacting',
]
present_unused = [k for k in UNUSED_KEYS if k in dict_specific]
if present_unused:
    print("NOTE: These keys are Tobias-style and may be ignored by your CLASS build:", present_unused)

print("Using G_eff_ncdm =", dict_specific['G_eff_ncdm'], "(copied from G_eff_ncdm_interacting)")


# =========================
# CLASS initialisation (FIX)
# =========================

model_pp = classy_pp.Class()
model_pp.set(dict_specific)
model_pp.compute()

model_reference = classy_pp.Class()
model_reference.set(dict_general)
model_reference.compute()


## Matter-Power Spectrum

kmax= 1
klist = np.logspace(-4, np.log10(kmax), 1000)

def mPk(perts):
    pk = np.array([perts.pk(k*perts.h(), 0.)*perts.h()**3 for k in klist])
    return pk

pk_model_pp = mPk(model_pp)
pk_model_reference = mPk(model_reference)


output_data = np.column_stack([
    klist,
    pk_model_pp,
    pk_model_reference
])

np.savetxt(
    os.path.join(OUT_DIR, "mPk.txt"),
    output_data,
    header="k [1/Mpc]    Pk_CLASS++    Pk_reference",
)

fig, ax = plt.subplots()

ax.plot(klist, pk_model_pp, label = "CLASS++")
ax.plot(klist, pk_model_reference, label = "CLASS++, No Hot NEDE nor Interacting Neutrino")
ax.set(yscale='log', xscale='log', xlabel="k", ylabel="mP(k)")
ax.set_title(rf"Comparison between CLASS++ and CLASS implementation of Hot NEDE and Interacting neutrino model")
ax.legend()

fig, ax = plt.subplots()

ax.plot(klist, (pk_model_pp/pk_model_reference))
ax.set()
ax.set(xscale='log', xlabel="k", ylabel="mP(k)_pp / mP(k)")
ax.set_title(rf"Comparison between CLASS++ and CLASS implementation of Hot NEDE and Interacting neutrino model")



plt.show()
