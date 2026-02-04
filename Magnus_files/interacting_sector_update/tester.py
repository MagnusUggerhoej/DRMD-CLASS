#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from classy_NEDE import Class

def run_pk(params, z=3.0, kmin=1e-4, kmax_req=3.0, nk=250):
    """Compute P(k,z) on a safe k-grid (clipped to CLASS kmax)."""
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # Find maximum k allowed by this run (avoid CosmoSevereError)
    kmax = None
    try:
        fo = cosmo.get_fourier()
        for key in ["k [1/Mpc]", "k", "k (1/Mpc)", "k (h/Mpc)"]:
            if key in fo:
                arr = np.asarray(fo[key])
                if arr.size:
                    kmax = float(np.max(arr))
                    break
    except Exception:
        pass

    if kmax is None:
        # fallback: start from requested and shrink until pk works
        k_try = float(kmax_req)
        while True:
            try:
                _ = cosmo.pk(k_try, z)
                kmax = k_try
                break
            except Exception:
                k_try *= 0.9
                if k_try < kmin * 1.1:
                    raise RuntimeError("Could not find valid kmax for pk().")

    kmax_use = min(kmax_req, 0.999 * kmax)
    ks = np.logspace(np.log10(kmin), np.log10(kmax_use), nk)
    pk = np.array([cosmo.pk(k, z) for k in ks])

    # a couple of useful sanity numbers
    mid = nk // 2
    print(f"z={z}  kmax≈{kmax:.3g}  using kmax_use={kmax_use:.3g}  pk(k={ks[mid]:.3e})={pk[mid]:.3e}")

    cosmo.struct_cleanup()
    cosmo.empty()
    return ks, pk

def interp_loglog(x_src, y_src, x_new):
    """Log-log interpolation for smooth spectra."""
    return np.exp(np.interp(np.log(x_new), np.log(x_src), np.log(y_src)))

def main():
    # Pick a redshift where neutrino effects are typically more visible
    z = 3.0

    base = {
        # mPk needed for pk(); keep lensing off to speed up if you want
        "output": "mPk",
        "z_max_pk": z,

        "H0": 67.5,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "tau_reio": 0.054,
        "ln10^{10}A_s": 3.0,
        "n_s": 0.965,

        # two massive neutrinos (toy)
        "N_ncdm": 2,
        "m_ncdm": "0.05, 0.05",
        "deg_ncdm": "1.0, 1.0",
        "T_ncdm": "0.71611, 0.71611",

        # quiet
        "background_verbose": 0,
        "thermodynamics_verbose": 0,
        "perturbations_verbose": 0,
        "transfer_verbose": 0,
    }

    # --- Baseline (free-streaming) ---
    p0 = dict(base)
    p0["G_eff_ncdm_species"] = "0.0, 0.0"

    # --- Interacting: BOTH species interact (crank it) ---
    # Use your biggest value; if forced-damping is still in the C code,
    # any positive number will trigger it. Otherwise, this tests the real rate.
    p1 = dict(base)
    p1["G_eff_ncdm_species"] = "1e9, 1e9"
    # brute-force fallback: also set scalar coupling (won't hurt)
    p1["G_eff_ncdm"] = 1e9

    print("\nBaseline run:")
    k0, pk0 = run_pk(p0, z=z, kmin=1e-4, kmax_req=3.0, nk=300)

    print("\nInteracting run (both species):")
    k1, pk1 = run_pk(p1, z=z, kmin=1e-4, kmax_req=3.0, nk=300)

    # Common k-grid: intersection up to min(kmax)
    kmax_common = min(k0.max(), k1.max())
    k_common = k0[k0 <= kmax_common]

    pk0c = pk0[:len(k_common)]
    pk1c = interp_loglog(k1, pk1, k_common)

    ratio = pk1c / pk0c
    frac = ratio - 1.0

    # Print a compact “is it obvious?” summary
    print("\nDiagnostics (fractional change):")
    print(f"  max |ΔP/P| = {np.max(np.abs(frac)):.3e} at z={z}")
    print(f"  ratio range: [{ratio.min():.6f}, {ratio.max():.6f}]")

    # Plot only what matters for debugging: fractional difference
    plt.figure()
    plt.semilogx(k_common, frac)
    plt.axhline(0.0)
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("P_interacting / P_baseline  -  1")
    plt.title(f"ΔP/P at z={z}: both ncdm species interacting vs free-streaming")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
