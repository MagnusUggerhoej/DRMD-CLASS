"""
Consistency check: Thomas (CLASSpp) vs Mine (modified NEDE branch)
with ALL neutrinos self-interacting:

- UR (massless neutrino sector) interactions ON
- NCDM (massive neutrino sector) interactions ON for all ncdm species

We compare linear matter power spectrum P(k) at z=0 on the same physical k [1/Mpc] grid.
"""

from classy_NEDE import Class as ClassMine
from classy import Class as ClassThomas

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Interaction strength knobs (log10, so 6 means 10^6)
# Choose something "clearly interacting" but not insane.
# ------------------------------------------------------------
LOG10_G_EFF_UR_ON   = 6.0
LOG10_G_EFF_NCDM_ON = 6.0


def common_base(lmax=2500):
    return {
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "A_s": 2.1e-9,
        "n_s": 0.9649,
        "tau_reio": 0.0543,

        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": lmax,

        # ensure pk table exists and is wide enough
        "P_k_max_1/Mpc": 10.0,

        "T_cmb": 2.7255,
        "Omega_k": 0.0,

        # IMPORTANT: keep the same radiation budget in both
        # (CLASS uses N_ur for massless relics; your build likely does too)
        "N_ur": 3.046,
    }


def run(Cls, p, label):
    print(f"Running {label}...")
    c = Cls()
    c.set(p)
    c.compute()
    return c


def pk_safe(cosmo, k, z):
    if hasattr(cosmo, "pk_lin"):
        return cosmo.pk_lin(k, z)
    return cosmo.pk(k, z)


# ============================================================
# Thomas / CLASSpp parameters: "all neutrinos interacting"
# ============================================================
params_thomas = common_base(lmax=2500)

# Turn ON UR interactions (Thomas build expects this key)
params_thomas["log10_G_eff_ur"] = LOG10_G_EFF_UR_ON

# Massive neutrinos: use interacting-ncdm interface (Tobias-style)
params_thomas.update({
    "N_ncdm_interacting": 1,
    "m_ncdm_interacting": 0.06,
    "deg_ncdm_interacting": 4.0,   # If you want total = 3.046, set 3.046; see note below
    "log10G_eff_ncdm_interacting": LOG10_G_EFF_NCDM_ON,
})

# NOTE on degeneracy:
# - If you want “one massive neutrino carrying the full neutrino degeneracy budget”:
#     deg_ncdm_interacting = 3.046
# - If you want “one massive species representing just one neutrino family”:
#     deg_ncdm_interacting ~ 1.0
# Pick one and match it in Mine below. Right now it's set to 1.0.



# ============================================================
# Mine / modified branch parameters: "all neutrinos interacting"
# ============================================================
params_mine = common_base(lmax=2500)

# Turn ON UR interactions in your build (you used this before)
params_mine["log10_G_eff_ur"] = LOG10_G_EFF_UR_ON

# Use your new split keys: DO NOT set N_ncdm when using split
params_mine.update({
    "N_ncdm_standard": 0,
    "N_ncdm_interacting": 1,

    # your split-sector mass/deg keys
    "m_ncdm_interacting": 0.06,
    "deg_ncdm_interacting": 4.0,  # MUST match Thomas choice above

    # your split-sector coupling key (list or scalar)
    # you already verified this works:
    "log10G_eff_ncdm_interacting": LOG10_G_EFF_NCDM_ON,
})

# If your build still also reads legacy scalar log10_G_eff_ncdm,
# you can keep it consistent (harmless):
params_mine["log10_G_eff_ncdm"] = LOG10_G_EFF_NCDM_ON


# ============================================================
# Run both + compare P(k)
# ============================================================
cosmo_T = run(ClassThomas, params_thomas, "Thomas (CLASSpp)")
cosmo_M = run(ClassMine,   params_mine,   "Mine (split ncdm + interactions)")

# same physical k [1/Mpc]
k_1Mpc = np.logspace(-4, 1.0, 250)

Pk_T_1Mpc = np.array([pk_safe(cosmo_T, k, 0.0) for k in k_1Mpc])
Pk_M_1Mpc = np.array([pk_safe(cosmo_M, k, 0.0) for k in k_1Mpc])

# put both in common (Mpc/h)^3 using Thomas h as reference
h_ref = cosmo_T.h()
k_hMpc = k_1Mpc / h_ref
Pk_T = Pk_T_1Mpc * h_ref**3
Pk_M = Pk_M_1Mpc * h_ref**3

plt.figure(figsize=(7, 5))
plt.loglog(k_hMpc, Pk_T, label="Thomas (interacting)")
plt.loglog(k_hMpc, Pk_M, ls="--", label="Mine (interacting)")
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
plt.title("P(k) – interacting UR + interacting ncdm (linear)")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

ratio = Pk_M / Pk_T
plt.figure(figsize=(7, 3.8))
plt.semilogx(k_hMpc, ratio)
plt.axhline(1.0, ls=":")
plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$ (common $h$)")
plt.ylabel("Mine / Thomas")
plt.title("P(k) ratio (interacting)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

print("h(Thomas) =", cosmo_T.h(), " h(Mine) =", cosmo_M.h())
print("max |ratio-1| =", np.max(np.abs(ratio - 1)))

cosmo_T.struct_cleanup(); cosmo_T.empty()
cosmo_M.struct_cleanup(); cosmo_M.empty()
