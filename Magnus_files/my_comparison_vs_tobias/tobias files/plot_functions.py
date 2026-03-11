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



