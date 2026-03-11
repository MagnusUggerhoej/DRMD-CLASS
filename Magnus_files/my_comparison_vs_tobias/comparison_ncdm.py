#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from comparison_functions import *


dict_general_pp={
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk,mTk',
        'lensing':'yes',
        'start_large_k_at_tau_h_over_tau_k': 0.01,
        'perturbations_verbose':'0',
        'thermodynamics_verbose':'0',
        'z_max_pk': 4.,
        'h': 0.67,
        'N_ur' : 2.0328,
        #'N_ncdm': 1,        
        #'deg_ncdm': 3, #Three degenerate interacting neutrinos with
        #'m_ncdm': 1e-2,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,
        'back_integration_stepsize': 7e-4,
        #'thermo_integration_stepsize': 7e-4,
        'perturb_integration_stepsize': 7e-4,
        'gauge': 'synchronous',

    }


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
        #'N_ncdm': 1,        
        #'deg_ncdm': 3, #Three degenerate interacting neutrinos with
        #'m_ncdm': 1e-2,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,
        'background_integration_stepsize': 7e-4,
        'thermo_integration_stepsize': 7e-4,
        'perturbations_integration_stepsize': 7e-4,
        'gauge': 'synchronous',
    }

dict_ncdm={
        'quadrature_strategy_ncdm_interacting': 0,
        'N_momentum_bins_ncdm_interacting': 5,
        'maximum_q_ncdm_interacting': 15.0,
        'T_ncdm_interacting': 0.71611, #T_ncdm_default
        'ksi_ncdm_interacting': 0.0, #ksi_ncdm_default
        'N_ncdm_interacting': 1,        
        'deg_ncdm_interacting':1, #Three degenerate interacting neutrinos with
        'm_ncdm_interacting': 1e-2,
        'ncdm_fluid_approximation': 3,
        'G_eff_ncdm_interacting': 1e-3,
        'Omega_ncdm_interacting': 0.0,
    }


dict_drmd={ 
        #'G_over_aH_drmd_ini':1e-3,      
        #'delta_Neff_drmd':1.0,
        #'f_idm_drmd':0.5,
        #'z_stop':5000,
}


drmd_off = {
    "f_idm_drmd": 1e-5,
    "delta_Neff_drmd": 0.0,
    "z_stop": 0.0,
    "G_over_aH_drmd_ini": 0.0,
}

def get_dict_general():
    return dict_general

def get_dict_general_pp():
    return dict_general_pp

def get_dict_ncdm():
    dict_out = dict_general.copy()
    dict_out.update(drmd_off)   # <-- keep this
    dict_out.update(dict_ncdm)
    return dict_out

def get_dict_ncdm_pp():
    dict_out = dict_general_pp.copy()
    dict_out.update(drmd_off)   # <-- keep this
    dict_out.update(dict_ncdm)
    return dict_out

def get_dict_drmd():
    dict_out = dict_general.copy()
    dict_out.update(dict_drmd)
    return dict_out

def get_dict_drmd_pp():
    dict_out = dict_general_pp.copy()
    dict_out.update(dict_drmd)
    return dict_out





#import classy as classy_pp
import classy_NEDE as classy_Magnus
import classy_tobias as classy_Tobias
import classy_tobias, os
print(classy_tobias.__file__)
import classy as classy_reference

#from classy import Class as classy_reference
#from classy_NEDE import Class as classy_Magnus
#from classy_tobias import Class as classy_Tobias


model_Magnus_ncdm = run_cosmo(classy_Magnus, get_dict_ncdm())
#model_Tobias_ncdm = run_cosmo(classy_Tobias, get_dict_ncdm_pp())
model_reference_ncdm = run_cosmo(classy_reference, get_dict_ncdm_pp())


from scipy.interpolate import interp1d


# Load backgrounds
background_Magnus_ncdm = model_Magnus_ncdm.get_background()
#background_Tobias_ncdm = model_Tobias_ncdm.get_background()
background_reference_ncdm = model_reference_ncdm.get_background()

print(background_Magnus_ncdm['z'])
#print(background_Tobias_ncdm['z'])



z_magnus = background_Magnus_ncdm['z']
#z_tobias = background_Tobias_ncdm['z']
z_reference = background_reference_ncdm['z']


for key in background_Magnus_ncdm.keys():
    if key == 'z':
        continue
    

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    
    axs[0].plot(z_magnus, background_Magnus_ncdm[key], label='Magnus', ls='solid')
    #axs[0].plot(z_tobias, background_Tobias_ncdm[key], label='Tobias', ls='dashed', color = 'green')
    axs[0].plot(z_reference, background_reference_ncdm[key], label='Reference', ls='dotted', lw =2, color = 'magenta')
    axs[0].set(xlabel='z', ylabel=key, xscale='log')
    axs[0].invert_xaxis()  # Invert x-axis to have z decreasing from left to right
    axs[0].set_title(f'Comparison for {key}')
    axs[0].legend()


    #Create a logscale plot of the relative difference between Magnus and Tobias. First, we need to interpolate the Tobias data to the same z values as Magnus.
    #interp_Tobias = interp1d(z_tobias, background_Tobias_ncdm[key], kind='cubic', fill_value='extrapolate')
    #Tobias_interp = interp_Tobias(z_magnus)

    #relative_difference = (background_Magnus_ncdm[key] - Tobias_interp) / background_Magnus_ncdm[key]
    #axs[1].plot(z_magnus, relative_difference, label='Relative difference', ls='solid')
    axs[1].set(xlabel='z', ylabel=f'Relative difference for {key}', xscale='log')
    axs[1].invert_xaxis()  # Invert x-axis to have z decreasing from left to right
    axs[1].legend()






