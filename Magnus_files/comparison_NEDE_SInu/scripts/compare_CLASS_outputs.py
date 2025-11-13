import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Setup TT power spectrum
# ------------------------------------------------------------
base_dir = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/Magnus_files"

def load_cl(filename):
    """Load CLASS C_l file (l, TT, EE, BB, TE, ...) and return l and C_l^TT."""
    data = np.loadtxt(os.path.join(base_dir, filename))
    l = data[:, 0]
    cl_TT = data[:, 1]  # TT spectrum (µK^2)
    return l, cl_TT

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
l_base, clT_base = load_cl("output_baseline_00_cl_lensed.dat")

# NEDE models with different G_eff values
geff_runs = {
    r"$G_{\rm eff}=10^{-20}$": "output_Geff_1eminus20_00_cl_lensed.dat",
    r"$G_{\rm eff}=10^{5}$": "output_Geff_1e5_00_cl_lensed.dat",
    r"$G_{\rm eff}=10^{9}$": "output_Geff_1e9_00_cl_lensed.dat"
}

Dl_base = l_base * (l_base + 1) * clT_base / (2 * np.pi)

Dl_geff = {}
for label, fname in geff_runs.items():
    l, clT = load_cl(fname)
    Dl_geff[label] = l * (l + 1) * clT / (2 * np.pi)

# ------------------------------------------------------------
# Plot combined figure
# ------------------------------------------------------------
plt.figure(figsize=(8, 7))

# --- Upper panel: C_l spectra
ax1 = plt.subplot(211)
ax1.semilogx(l_base, Dl_base, label=r"ΛCDM (CLASS++)", lw=2.0, color='black')

colors = ['red', 'green', 'blue']
for (label, Dl), c in zip(Dl_geff.items(), colors):
    ax1.semilogx(l_base, Dl, '--', lw=1.8, label=label, color=c)

ax1.set_xlim(2, 3000)
ax1.set_ylabel(r"$\ell(\ell+1)C_\ell^{TT}/(2\pi)\, [\mu{\rm K}^2]$")
ax1.set_title(r"CMB TT Spectrum — Effect of Neutrino Self-Interactions ($G_{\rm eff}$)")
ax1.legend(frameon=False)
ax1.grid(True, which='both', ls=':', alpha=0.5)

# --- Lower panel: relative difference
ax2 = plt.subplot(212, sharex=ax1)
for (label, Dl), c in zip(Dl_geff.items(), colors):
    rel = (Dl - Dl_base) / Dl_base * 100
    ax2.semilogx(l_base, rel, lw=1.5, label=label, color=c)

ax2.axhline(0, color='k', lw=0.7)
ax2.set_xlim(2, 3000)
ax2.set_ylim(-10, 10)
ax2.set_xlabel(r"Multipole $\ell$")
ax2.set_ylabel(r"$\Delta C_\ell^{TT} / C_\ell^{TT}\,[\%]$")
ax2.grid(True, which='both', ls=':', alpha=0.5)

# Annotate regions of interest
ax2.text(200, 5, "Suppression\nof peaks", color='C3', fontsize=10, ha='center')
ax2.text(1500, -5, "Small-scale damping", color='C3', fontsize=10, ha='center')

plt.tight_layout(h_pad=0.3)
plt.show()






# ------------------------------------------------------------
# Setup for matter power spectrum
# ------------------------------------------------------------


def load_pk(filename):
    """Load CLASS P(k) file (k, P(k)) and return arrays."""
    data = np.loadtxt(os.path.join(base_dir, filename))
    k = data[:, 0]
    pk = data[:, 1]
    return k, pk

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
k_base, pk_base = load_pk("output_baseline_00_pk.dat")

# NEDE models with different G_eff values
geff_runs = {
    r"$G_{\rm eff}=10^{-20}$": "output_Geff_1eminus20_00_pk.dat",
    r"$G_{\rm eff}=10^{5}$": "output_Geff_1e5_00_pk.dat",
    r"$G_{\rm eff}=10^{9}$": "output_Geff_1e9_00_pk.dat"
}

pk_geff = {}
for label, fname in geff_runs.items():
    k, pk = load_pk(fname)
    pk_geff[label] = (k, pk)

# ------------------------------------------------------------
# Plot combined figure
# ------------------------------------------------------------
plt.figure(figsize=(8, 7))

# --- Upper panel: P(k)
ax1 = plt.subplot(211)
ax1.loglog(k_base, pk_base, label=r"ΛCDM (CLASS++)", lw=2.0, color='black')

colors = ['red', 'green', 'blue']
for (label, (k, pk)), c in zip(pk_geff.items(), colors):
    ax1.loglog(k, pk, '--', lw=1.8, label=label, color=c)

ax1.set_xlim(1e-4, 1)
ax1.set_ylabel(r"$P(k)\,[{\rm Mpc}^3]$")
ax1.set_title(r"Matter Power Spectrum — Effect of Neutrino Self-Interactions ($G_{\rm eff}$)")
ax1.legend(frameon=False)
ax1.grid(True, which='both', ls=':', alpha=0.5)

# --- Lower panel: relative difference
ax2 = plt.subplot(212, sharex=ax1)
for (label, (k, pk)), c in zip(pk_geff.items(), colors):
    rel = (pk - pk_base) / pk_base * 100
    ax2.semilogx(k, rel, lw=1.5, label=label, color=c)

ax2.axhline(0, color='k', lw=0.7)
ax2.set_xlim(1e-4, 1)
ax2.set_ylim(-15, 15)
ax2.set_xlabel(r"$k\,[{\rm Mpc}^{-1}]$")
ax2.set_ylabel(r"$\Delta P / P\,[\%]$")
ax2.grid(True, which='both', ls=':', alpha=0.5)

# Annotate regions of interest
ax2.text(2e-3, 10, "Large-scale (linear)\nregime", color='C3', fontsize=10, ha='center')
ax2.text(0.1, -10, "Small-scale\nsuppression", color='C3', fontsize=10, ha='center')

plt.tight_layout(h_pad=0.3)
plt.show()

