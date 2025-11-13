import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
base_dir = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/Magnus_files/comparison_NEDE_SInu"
nede_dir = os.path.join(base_dir, "outputs_NEDE")
classpp_dir = os.path.join(base_dir, "outputs_CLASSpp")

# ----------------------------------------------------------------------
# Helper: load CLASS spectra or matter power files
# ----------------------------------------------------------------------
def load_cl(filename):
    """Load CLASS C_l file (l, TT, EE, BB, TE, ...) and return l and C_l^TT."""
    path = os.path.join(nede_dir if "Geff" in filename else classpp_dir, filename)
    data = np.loadtxt(path)
    l = data[:, 0]
    cl_TT = data[:, 1]  # TT spectrum (μK²)
    return l, cl_TT

def load_pk(filename):
    """Load matter power spectrum P(k)."""
    path = os.path.join(nede_dir if "Geff" in filename else classpp_dir, filename)
    data = np.loadtxt(path)
    k, pk = data[:, 0], data[:, 1]
    return k, pk

# ----------------------------------------------------------------------
# Choose which plot to make (CMB or matter)
# ----------------------------------------------------------------------
make_cl_plot = True   # toggle this to False if you want P(k)

# Baseline CLASS++ ΛCDM
l_base, clT_base = load_cl("output_baseline_00_cl_lensed.dat")
Dl_base = l_base * (l_base + 1) * clT_base / (2 * np.pi)

# NEDE runs with different G_eff values
geff_runs = {
    r"$G_{\rm eff}=10^{-20}$": "output_Geff_1eminus20_00_cl_lensed.dat",
    r"$G_{\rm eff}=10^{5}$": "output_Geff_1e5_00_cl_lensed.dat",
    r"$G_{\rm eff}=10^{9}$": "output_Geff_1e9_00_cl_lensed.dat"
}

Dl_geff = {}
for label, fname in geff_runs.items():
    l, clT = load_cl(fname)
    Dl_geff[label] = l * (l + 1) * clT / (2 * np.pi)

# ----------------------------------------------------------------------
# --- Plot: CMB TT Comparison NEDE vs CLASS++
# ----------------------------------------------------------------------
if make_cl_plot:
    plt.figure(figsize=(8, 7))

    # Upper panel — TT power
    ax1 = plt.subplot(211)
    ax1.semilogx(l_base, Dl_base, label=r"ΛCDM (CLASS++)", lw=2.0, color='black')
    colors = ['red', 'green', 'blue']
    for (label, Dl), c in zip(Dl_geff.items(), colors):
        ax1.semilogx(l_base, Dl, '--', lw=1.8, label=f"NEDE {label}", color=c)
    ax1.set_xlim(2, 3000)
    ax1.set_ylabel(r"$\ell(\ell+1)C_\ell^{TT}/(2\pi)\, [\mu{\rm K}^2]$")
    ax1.set_title(r"Comparison: NEDE vs CLASS++ — CMB TT Spectrum")
    ax1.legend(frameon=False)
    ax1.grid(True, which='both', ls=':', alpha=0.5)

    # Lower panel — relative difference
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
    ax2.text(200, 5, "Suppression\nof peaks", color='C3', fontsize=10, ha='center')
    ax2.text(1500, -5, "Small-scale damping", color='C3', fontsize=10, ha='center')

    plt.tight_layout(h_pad=0.3)
    plt.show()

# ----------------------------------------------------------------------
# --- Optional: P(k) plot (if make_cl_plot=False)
# ----------------------------------------------------------------------
else:
    k_base, pk_base = load_pk("output_baseline_00_pk.dat")
    plt.figure(figsize=(8, 7))
    plt.loglog(k_base, pk_base, label="CLASS++ ΛCDM", lw=2.0, color='black')
    colors = ['red', 'green', 'blue']
    for (label, fname), c in zip(geff_runs.items(), colors):
        k, pk = load_pk(fname.replace("_cl_lensed", "_pk"))
        plt.loglog(k, pk, '--', lw=1.8, label=f"NEDE {label}", color=c)
    plt.xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)\,[\mathrm{Mpc}^3]$")
    plt.title("Matter Power Spectrum — NEDE vs CLASS++")
    plt.legend(frameon=False)
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
