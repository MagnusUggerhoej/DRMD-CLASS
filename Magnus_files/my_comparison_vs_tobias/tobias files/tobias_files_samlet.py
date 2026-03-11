dict_general={
        'k_output_values': '0.05, 0.002',
        'output':'tCl,pCl,lCl,mPk,mTk',
        'lensing':'yes',
        'start_large_k_at_tau_h_over_tau_k': 0.01,
        'perturbations_verbose':'0',
        'thermodynamics_verbose':'0',
        'z_max_pk': 4.,
        'h': 0.67,
        'N_ur' : 0,
        #'N_ncdm': 1,        
        #'deg_ncdm': 3, #Three degenerate interacting neutrinos with
        #'m_ncdm': 1e-2,
        'omega_b': 0.022,
        'omega_cdm': 0.12,
        'tau_reio': 0.054,
        'ln10^{10}A_s': 3,
        'l_max_scalars': 2500,
        'n_s': 0.965,
        
        'gauge': 'synchronous',
        #'background_verbose': '3',
    }

dict_precision_pp = {  'back_integration_stepsize': 7e-4,
                    #'thermo_integration_stepsize': 7e-4,
                    'perturb_integration_stepsize': 7e-4,}

dict_precision = {'background_integration_stepsize': 7e-4,
        'thermo_integration_stepsize': 7e-4,
        'perturbations_integration_stepsize': 7e-4,}

dict_ncdm={
        'quadrature_strategy_ncdm_interacting': 0,
        'N_momentum_bins_ncdm_interacting': 5,
        'maximum_q_ncdm_interacting': 15.0,
        'T_ncdm_interacting': 0.71611, #T_ncdm_default
        'ksi_ncdm_interacting': 0.0, #ksi_ncdm_default
        'N_ncdm_interacting': 1,        
        'deg_ncdm_interacting':1, #Three degenerate interacting neutrinos with
        'm_ncdm_interacting': 1e-5,
        'ncdm_fluid_approximation': 3,
        'G_eff_ncdm_interacting': '1e-3',
        'Omega_ncdm_interacting': 0.0,
        'background_verbose': '3',
    }


dict_ncdm_multiple_species={
        'N_ncdm_interacting': 2,   
        'quadrature_strategy_ncdm_interacting': '0, 0',
        'N_momentum_bins_ncdm_interacting': '5, 5',
        'maximum_q_ncdm_interacting': '15.0, 15.0',
        'T_ncdm_interacting': '0.71611, 0.71611' , #T_ncdm_default
        'ksi_ncdm_interacting': '0.0, 0.0', #ksi_ncdm_default
        'deg_ncdm_interacting':'1,1', #Three degenerate interacting neutrinos with
        'm_ncdm_interacting': '1e-2, 1e-3',
        'ncdm_fluid_approximation': '3, 3',
        'G_eff_ncdm_interacting': '1e-3,1e10',
        'Omega_ncdm_interacting': '0.0, 0.0',
    }


dict_drmd={ 
        'G_over_aH_drmd_ini':1e-3,      
        'delta_Neff_drmd':1.0,
        'f_idm_drmd':0.5,
        'z_stop':5000,
}

def get_dict_general():
    return dict_general | dict_precision

def get_dict_general_pp():
    return dict_general | dict_precision_pp

def get_dict_ncdm():
    dict_out = dict_general.copy()
    dict_out.update(dict_ncdm)
    return dict_out

def get_dict_ncdm_pp():
    dict_out = get_dict_general_pp()
    dict_out.update(dict_ncdm)
    return dict_out

def get_dict_ncdm_multiple_species():
    dict_out = get_dict_general()
    dict_out.update(dict_ncdm_multiple_species)
    return dict_out

def get_dict_ncdm_pp_multiple_species():
    dict_out = get_dict_general_pp.copy()
    dict_out.update(dict_ncdm_multiple_species)
    return dict_out

def get_dict_drmd():
    dict_out = get_dict_general.copy()
    dict_out.update(dict_drmd)
    return dict_out

def get_dict_drmd_pp():
    dict_out = get_dict_general_pp.copy()
    dict_out.update(dict_drmd)
    return dict_out


def run_cosmo(classy_version, dict_in):
    model = classy_version.Class()
    model.set(dict_in)
    model.compute()
    return model


## Methods for making various plots or




##Matter-Power Spectrum

def plot_mPk(model, ax = None, kmin = 1e-4, kmax = 1, label = 'Model',save_path = None, **kwargs):
    kmax = 1
    klist = np.logspace(-4, np.log10(kmax), 1000)


    mPk = np.array([model.pk(k*model.h(), 0.)*model.h()**3 for k in klist])
        

    if save_path is not None:
        output_data = np.column_stack([
            klist,
            mPk,
        ])

        np.savetxt(
            save_path,
            output_data,
            header="k [1/Mpc]    mPk",
        )

    if ax is not None:
            ax.plot(klist, mPk, label = label, **kwargs)

    return klist, mPk




def plot_Cl_TT(model, ax = None, label = 'Model', save_path = None,**kwargs):

    # Get the computed data dictionary
    cl = model.lensed_cl(2500)
    l = cl['ell']
    # Compute C_l = l(l+1)C_l / 2pi (in muK^2)
    factor = l * (l + 1) / (2 * np.pi) * 1e12
    cl_TT = factor * cl['tt']

  
    if save_path is not None:
        output_data = np.column_stack([
            l,
            cl_TT,
        ])

        np.savetxt(
            save_path,
            output_data,
            header="l   cl_TT",
        )


    if ax is not None:
        ax.plot(l, cl_TT, label=label, **kwargs)

    return l, cl_TT

def plot_Cl_TE(model, ax = None, label = 'Model', save_path = None, **kwargs):

    # Get the computed data dictionary
    cl = model.lensed_cl(2500)
    l = cl['ell']
    # Compute C_l = l(l+1)C_l / 2pi (in muK^2)
    factor = l * (l + 1) / (2 * np.pi) * 1e12
    cl_TE = factor * cl['te']

  
    if save_path is not None:
        output_data = np.column_stack([
            l,
            cl_TE,
        ])

        np.savetxt(
            save_path,
            output_data,
            header="l   cl_TE",
        )


    if ax is not None:
        ax.plot(l, cl_TE, label=label,**kwargs)

    return l, cl_TE


def plot_Cl_EE(model, ax = None, label = 'Model', save_path = None,**kwargs):

    # Get the computed data dictionary
    cl = model.lensed_cl(2500)
    l = cl['ell']
    # Compute C_l = l(l+1)C_l / 2pi (in muK^2)
    factor = l * (l + 1) / (2 * np.pi) * 1e12
    cl_EE = factor * cl['ee']

  
    if save_path is not None:
        output_data = np.column_stack([
            l,
            cl_EE,
        ])

        np.savetxt(
            save_path,
            output_data,
            header="l   cl_EE",
        )


    if ax is not None:
        ax.plot(l, cl_EE, label=label, **kwargs)

    return l, cl_EE


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  


def plot_mTk(model, z = 0, species_list=None, ax=None, label='Model', save_path=None, **kwargs):
    if species_list is None:
        species_list = ['d_b', 'd_cdm']

    mTk = model.get_transfer(z=z)
    klist = mTk['k (h/Mpc)']

    # Compute transfer functions
    mTks = {species: np.abs(mTk[species]) for species in species_list}

    # Save to file if requested
    if save_path is not None:
        output_data = np.column_stack(
            [klist] + [mTks[sp] for sp in species_list]
        )

        header = "k [h/Mpc]\t" + "\t".join(species_list)

        np.savetxt(
            save_path,
            output_data,
            header=header
        )

    # Plot if axis provided
    if ax is not None:
        for species in species_list:
            ax.plot(
                klist,
                mTks[species],
                label=f"{label}: {species}",
                **kwargs
            )

    return klist, mTks



perturbations_Magnus_ncdm = model_Magnus_ncdm.get_perturbations()['scalar'][0]
perturbations_Tobias_ncdm = model_Tobias_ncdm.get_perturbations()['scalar'][0]
perturbations_reference_ncdm = model_reference_ncdm.get_perturbations()['scalar'][0]

a_Magnus = perturbations_Magnus_ncdm['a']
a_reference = perturbations_reference_ncdm['a']
a_Tobias = perturbations_Tobias_ncdm['a']
for key in perturbations_Magnus_ncdm.keys():
    if key == 'a':
        continue
    

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    axs[0].plot(a_Magnus, perturbations_Magnus_ncdm[key], label='Magnus', ls='solid')
    #ax.plot(a_reference, perturbations_reference_ncdm[key], label='Reference', ls='dotted', lw =2, color = 'magenta')
    axs[0].plot(a_Tobias, perturbations_Tobias_ncdm[key], label='Tobias', ls='dashed', color = 'green')
    axs[0].set(xlabel='a', ylabel=key, xscale='log')
    #ax.invert_xaxis()  # Invert x-axis to have z decreasing from left to right
    axs[0].set_title(f'Comparison for {key}')
    axs[0].legend()

    #Create a logscale plot of the relative difference between Magnus and Tobias. First, we need to interpolate the Tobias data to the same a values as Magnus.
    interp_Tobias = interp1d(a_Tobias, perturbations_Tobias_ncdm[key], kind='cubic', fill_value='extrapolate')
    Tobias_interp = interp_Tobias(a_Magnus)

    relative_difference = (perturbations_Magnus_ncdm[key] - Tobias_interp) / perturbations_Magnus_ncdm[key]
    axs[1].plot(a_Magnus, relative_difference, label='Relative difference', ls='solid')
    axs[1].set(xlabel='a', ylabel=f'Relative difference for {key}', xscale='log')
    axs[1].legend()


for key in perturbations_Magnus_ncdm.keys():
    print(key)

for key in perturbations_Tobias_ncdm.keys():
    print(key)