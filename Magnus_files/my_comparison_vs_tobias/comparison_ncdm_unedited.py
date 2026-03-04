#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from comparison_functions import *

#import classy as classy_pp
import classy_NEDE as classy_Magnus
import classy_tobias as classy_Tobias
import classy as classy_reference

#from classy import Class as classy_reference
#from classy_NEDE import Class as classy_Magnus
#from classy_tobias import Class as classy_Tobias


model_Magnus_ncdm = run_cosmo(classy_Magnus, get_dict_ncdm())
model_Tobias_ncdm = run_cosmo(classy_Tobias, get_dict_ncdm_pp())
model_reference_ncdm = run_cosmo(classy_reference, get_dict_ncdm_pp())


from scipy.interpolate import interp1d


# Load backgrounds
background_Magnus_ncdm = model_Magnus_ncdm.get_background()
background_Tobias_ncdm = model_Tobias_ncdm.get_background()
background_reference_ncdm = model_reference_ncdm.get_background()

print(background_Magnus_ncdm['z'])
print(background_Tobias_ncdm['z'])



z_magnus = background_Magnus_ncdm['z']
z_tobias = background_Tobias_ncdm['z']
z_reference = background_reference_ncdm['z']


for key in background_Magnus_ncdm.keys():
    if key == 'z':
        continue
    

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    
    axs[0].plot(z_magnus, background_Magnus_ncdm[key], label='Magnus', ls='solid')
    axs[0].plot(z_tobias, background_Tobias_ncdm[key], label='Tobias', ls='dashed', color = 'green')
    axs[0].plot(z_reference, background_reference_ncdm[key], label='Reference', ls='dotted', lw =2, color = 'magenta')
    axs[0].set(xlabel='z', ylabel=key, xscale='log')
    axs[0].invert_xaxis()  # Invert x-axis to have z decreasing from left to right
    axs[0].set_title(f'Comparison for {key}')
    axs[0].legend()


    #Create a logscale plot of the relative difference between Magnus and Tobias. First, we need to interpolate the Tobias data to the same z values as Magnus.
    interp_Tobias = interp1d(z_tobias, background_Tobias_ncdm[key], kind='cubic', fill_value='extrapolate')
    Tobias_interp = interp_Tobias(z_magnus)

    relative_difference = (background_Magnus_ncdm[key] - Tobias_interp) / background_Magnus_ncdm[key]
    axs[1].plot(z_magnus, relative_difference, label='Relative difference', ls='solid')
    axs[1].set(xlabel='z', ylabel=f'Relative difference for {key}', xscale='log')
    axs[1].invert_xaxis()  # Invert x-axis to have z decreasing from left to right
    axs[1].legend()






