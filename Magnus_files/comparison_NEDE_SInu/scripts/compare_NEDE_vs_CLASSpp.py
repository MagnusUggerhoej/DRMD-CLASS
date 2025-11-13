import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Setup paths
# ------------------------------------------------------------
base_dir = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/Magnus_files/comparison_NEDE_SInu"
path_classpp = os.path.join(base_dir, "outputs_CLASSpp")
path_nede = os.path.join(base_dir, "outputs_NEDE")

def load_cl(path, filename):
    data = np.loadtxt(os.path.join(path, filename))
    l = data[:, 0]
    cl_TT = data[:, 1]
    Dl = l * (l + 1) * cl_TT / (2 * np.pi)
    return l, Dl

# ------------------------------------------------------------
# Load CLASS++ and NEDE spectra (same G_eff)
# ------------------------------------------------------------
l_classpp, Dl_classpp = load_cl(path_classpp, "output_Geff_1e9_00_cl_lensed.dat")
l_nede, Dl_nede = load_cl(path_nede, "output_NEDE_00_cl_lensed.dat")  # rename if needed

# ------------------------------------------------------------
# Plot comparison
# ------------------------------------------------------------
plt.figure(figsize=(8, 7))

# --- Upper panel: C_l spectra
ax1 = plt.subplot(211)
ax1.semilogx(l_classpp, Dl_classpp, label=r"CLASS++ ($G_{\rm eff}=10^9$)", color="C0", lw=1.8)
ax1.semilogx(l_nede, Dl_nede, "--", label=r"CLASS$_{\rm NEDE}$ ($G_{\rm eff}=10^9$)", color="C3", lw=1.8)
ax1.set_xlim(2, 3000)
ax1.set_ylabel(r"$\ell(\ell+1)C_\ell^{TT}/(2\pi)$ [$\mu$K$^2$]")
ax1.set_title("CMB TT Spectrum — Comparison of CLASS++ vs CLASS_NEDE ($G_{\\rm eff}=10^9$)")
ax1.legend(frameon=False)
ax1.grid(True, which="both", ls=":", alpha=0.5)

# --- Lower panel: relative difference
ax2 = plt.subplot(212, sharex=ax1)
rel_diff = (Dl_nede - Dl_classpp) / Dl_classpp * 100
ax2.semilogx(l_classpp, rel_diff, color="C3", lw=1.5)
ax2.axhline(0, color="k", lw=0.7)
ax2.set_xlim(2, 3000)
ax2.set_ylim(-5, 5)
ax2.set_xlabel(r"Multipole $\ell$")
ax2.set_ylabel(r"$(C_\ell^{\rm NEDE}-C_\ell^{\rm CLASS++})/C_\ell^{\rm CLASS++}$ [%]")
ax2.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout(h_pad=0.3)
plt.show()
