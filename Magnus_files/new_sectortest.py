#!/usr/bin/env python3
"""
Two-sector ncdm input test for DRMD-CLASS / classy_NEDE.

This script:
  1) Forces import of classy_NEDE (NOT classy)
  2) Prints the loaded .so path (so we know which build you're running)
  3) Runs:
       - legacy: N_ncdm
       - two-sector: N_ncdm_standard + N_ncdm_interacting
  4) Compares outputs (TT/EE and optional P(k)).

Run:
  (CLASS) python Magnus_files/tests/test_two_sector_ncdm.py
"""

import numpy as np

import classy_NEDE as classy  # <-- force the correct wrapper


def relmax(a, b, eps=1e-30):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), eps))


def run_case(params: dict, tag: str):
    print(f"\n=== Running: {tag} ===")
    cosmo = classy.Class()
    cosmo.set(params)
    cosmo.compute()

    # Use lensed if lensing=yes, else raw
    lmax = int(params.get("l_max_scalars", 2500))
    if params.get("lensing", "no") == "yes":
        cls = cosmo.lensed_cl(lmax)
    else:
        cls = cosmo.raw_cl(lmax)

    ell = cls["ell"]
    tt = cls["tt"]
    ee = cls["ee"]

    pk_out = None
    if "mPk" in params.get("output", ""):
        ks = np.logspace(-4, 0, 50)
        z = 0.0
        pk = np.array([cosmo.pk(k, z) for k in ks])
        pk_out = (ks, pk)

    cosmo.struct_cleanup()
    cosmo.empty()

    print(f"OK: {tag}")
    return ell, tt, ee, pk_out


def main():
    # --- diagnostics: which wrapper are we using? ---
    print("Using classy_NEDE from:", classy.__file__)

    # --- baseline cosmology (keep simple/stable) ---
    base = {
        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": 2500,
        "z_max_pk": 2.0,
        "P_k_max_h/Mpc": 1.0,

        "H0": 67.5,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "tau_reio": 0.054,
        "ln10^{10}A_s": 3.0,
        "n_s": 0.965,

        # keep whatever you normally use
        "N_ur": 2.0328,
    }

    # --- ncdm setup: 2 species total ---
    # NOTE: do NOT include G_eff_ncdm or log10_G_eff_ncdm_species here yet.
    # We are testing the sector split input-path first.
    ncdm_common = {
        "m_ncdm": "0.05,0.05",
        "T_ncdm": "0.71611,0.71611",
        "deg_ncdm": "1,1",
    }

    # --- legacy run ---
    legacy = dict(base)
    legacy.update({"N_ncdm": 2})
    legacy.update(ncdm_common)

    # --- two-sector run ---
    two_sector = dict(base)
    two_sector.update({"N_ncdm_standard": 1, "N_ncdm_interacting": 1})
    two_sector.update(ncdm_common)

    # --- run ---
    ell1, tt1, ee1, pk1 = run_case(legacy, "legacy (N_ncdm=2)")
    ell2, tt2, ee2, pk2 = run_case(two_sector, "two-sector (1+1)")

    # --- compare ---
    if not np.array_equal(ell1, ell2):
        raise RuntimeError("ell arrays differ between runs (unexpected for same l_max).")

    m = ell1 >= 2
    dTT = relmax(tt2[m], tt1[m])
    dEE = relmax(ee2[m], ee1[m])

    print("\n=== Comparison (two-sector vs legacy) ===")
    print(f"max rel diff TT (ell>=2): {dTT:.3e}")
    print(f"max rel diff EE (ell>=2): {dEE:.3e}")

    if pk1 is not None and pk2 is not None:
        ks1, pkv1 = pk1
        ks2, pkv2 = pk2
        if not np.allclose(ks1, ks2):
            raise RuntimeError("k grids differ between runs.")
        dPk = relmax(pkv2, pkv1)
        print(f"max rel diff P(k,z=0):  {dPk:.3e}")
    else:
        dPk = None

    # If sector split is purely an input change right now, these should match extremely well.
    tol = 1e-10
    ok_pk = (dPk is None) or (dPk < tol)

    if dTT < tol and dEE < tol and ok_pk:
        print("\n✅ PASS: two-sector input works and matches legacy (expected if physics unchanged).")
    else:
        print("\n⚠️ NOTE: outputs differ.")
        print("   - If you expect identical results, the sector split is already affecting something,")
        print("     or you are not comparing identical setups.")
        print("   - Next step would be to print key derived params (e.g., Omega_ncdm_tot) from both runs.")

    print("\nDone.")


if __name__ == "__main__":
    main()
