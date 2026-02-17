from classy_NEDE import Class
import numpy as np
import matplotlib.pyplot as plt

def get_h(cosmo):
    """Support both cosmo.h and cosmo.h()."""
    h_attr = getattr(cosmo, "h")
    return h_attr() if callable(h_attr) else h_attr

k_hMpc = np.logspace(-3, 0.7, 200)
z = 0.0

common = {
    "output": "mPk",
    "H0": 67.5,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "ln10^{10}A_s": 3.0,
    "n_s": 0.965,
    "N_ur": 2.0328,

    # avoid pk out-of-bounds
    "z_max_pk": 0.0,
    "P_k_max_h/Mpc": 10.0,

    # write background so we can VERIFY the species split in the header
    "write background": "yes",
    "write parameters": "yes",
}

# Baseline: split requested, but interaction ~ 0
pars_free = dict(common)
pars_free.update({
    "N_ncdm_standard": 1,
    "N_ncdm_interacting": 1,
    "m_ncdm": "0.06,0.06",
    "ncdm_fluid_approximation": 3,
    "log10_G_eff_ncdm_interacting_species": "-30",
    "root": "out_free_",
})

# Interacting: same split, but interaction on
pars_int = dict(common)
pars_int.update({
    "N_ncdm_standard": 1,
    "N_ncdm_interacting": 1,
    "m_ncdm": "0.06,0.06",
    "ncdm_fluid_approximation": 3,
    "log10_G_eff_ncdm_interacting_species": "9",
    "root": "out_int_",
})

cosmo_free = Class(); cosmo_free.set(pars_free); cosmo_free.compute()
cosmo_int  = Class(); cosmo_int.set(pars_int);  cosmo_int.compute()

hF, hI = get_h(cosmo_free), get_h(cosmo_int)
print("FREE h =", hF, " -> wrote out_free_background.dat")
print("INT  h =", hI, " -> wrote out_int_background.dat")

PkF = np.array([cosmo_free.pk(kk*hF, z) * hF**3 for kk in k_hMpc])
PkI = np.array([cosmo_int.pk(kk*hI, z) * hI**3 for kk in k_hMpc])

plt.figure()
plt.loglog(k_hMpc, PkF, label="baseline (G_eff ~ 0)")
plt.loglog(k_hMpc, PkI, label="interacting")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.xlabel("k [h/Mpc]")
plt.ylabel("P(k) [(Mpc/h)^3]")
plt.tight_layout()

plt.figure()
plt.semilogx(k_hMpc, PkI/PkF)
plt.axhline(1.0, color="k", ls="--")
plt.grid(True, which="both", ls=":")
plt.xlabel("k [h/Mpc]")
plt.ylabel("ratio")
plt.tight_layout()
plt.show()

cosmo_free.struct_cleanup(); cosmo_free.empty()
cosmo_int.struct_cleanup();  cosmo_int.empty()

print("\nNow verify the split from the BACKGROUND file headers:")
print("  head -n 1 out_free_background.dat")
print("  head -n 1 out_int_background.dat")
print("You should see TWO ncdm species columns (rho_ncdm... twice).")
