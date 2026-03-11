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


