import classy_NEDE as classy_pp

#import classy_other as classy


import numpy as np
import matplotlib.pyplot as plt





dict_general={
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk, mTk',
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
        'output':'tCl,pCl,lCl,mPk, mTk',
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
        #Interacting NCDM parameters start
        'N_ncdm_interacting': 1,        
        'deg_ncdm_interacting': 3, #Three degenerate interacting neutrinos with
        'm_ncdm_interacting': 1e-2,
        'ncdm_fluid_approximation': 3,
        'G_eff_ncdm_interacting': 1e-3,
        #Interacting NCDM parameters stop
        #DRMD parameters start
        "G_over_aH_drmd_ini":1e-3,      
        'delta_Neff_drmd':0.1,
        'f_idm_drmd':0.1,
        'z_stop':5000,
        #DRMD parameters end       
    }

    
model_pp = classy_pp.Class(dict_specific)
model_pp.compute()


model_reference = classy_pp.Class(dict_general)
model_reference.compute()



##Matter-Power Spectrum

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


fig, ax = plt.subplots()

ax.plot(klist, pk_model_pp, label = "CLASS++")
ax.plot(klist, pk_model_reference, label = "CLASS++, No Hot NEDE nor Interacting Neutrino")
ax.set(yscale='log', xscale='log', xlabel="k", ylabel="mP(k)")
ax.set_title(rf"Comparison between CLASS++ and CLASS implementation of Hot NEDE and Interacting neutrino model")
#ax.set_xscale("log")
ax.legend()

fig, ax = plt.subplots()

ax.plot(klist, (pk_model_pp/pk_model_reference))
ax.set()
ax.set( xscale='log',xlabel="k", ylabel="mP(k)_pp / mP(k)")

ax.set_title(rf"Comparison between CLASS++ and CLASS implementation of Hot NEDE and Interacting neutrino model")



plt.show()

