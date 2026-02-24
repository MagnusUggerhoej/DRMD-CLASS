#!/usr/bin/env python3
"""
BACKGROUND INTERPOLATION COMPARISON
Thomas CLASS vs Magnus NEDE-CLASS (legacy neutrinos, interactions OFF)

What this script does:
- Runs Thomas and Mine with an explicit baseline parameter set (legacy N_ncdm).
- Automatically strips parameters that a build reports as "did not read".
- Automatically falls back HyRec -> recfast if HyRec is not compiled.
- Compares BACKGROUND quantities at the SAME requested redshifts using interpolation
  (so you are not comparing different z-grids).
- Prints only mismatches + closure diagnostics.

Run:
  conda activate CLASS
  python test_interp_bg.py
"""

import re
import numpy as np

from classy import Class as ClassThomas
from classy_NEDE import Class as ClassMine

# -----------------------------
# User knobs
# -----------------------------
Z_LIST = [0.0, 10.0, 300.0, 1100.0, 1e5]
RTOL_BG = 1e-6
MAX_PASSES = 12

# -----------------------------
# Error parsing
# -----------------------------
_unread_re = re.compile(r"did not read input parameter\(s\):\s*(.*)", re.IGNORECASE)
_hyrec_missing_re = re.compile(r"compiled without including the HyRec code", re.IGNORECASE)

def parse_unread_params(exc: Exception):
    msg = str(exc)
    m = _unread_re.search(msg)
    if not m:
        return []
    tail = m.group(1).strip()
    return [p.strip() for p in tail.split(",") if p.strip()]

# -----------------------------
# Parameters (no alias conflicts)
# -----------------------------
def base_common():
    """
    Explicit baseline. Some keys may be unread by Thomas; we auto-strip.
    We start with recombination='recfast' because Thomas lacks HyRec.
    Interactions are OFF with exact zeros (important for your G_eff>0 logic).
    """
    return {
        # geometry / units
        "h": 0.6736,
        "Omega_k": 0.0,
        "T_cmb": 2.7255,

        # physical densities
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,

        # radiation / neutrinos
        "N_ur": 3.046,

        # outputs (background comes from get_background() after compute)
        "output": "mTk",
        "lensing": "no",
        "gauge": "synchronous",

        # try-set extras (may be unread depending on build)
        "non_linear": "none",
        "bbn": "BBN",
        "recombination": "recfast",

        # interactions OFF (exactly zero)
        "G_eff_ur": 0.0,
        "G_eff_ncdm": 0.0,
    }

def legacy_ncdm(p):
    p = dict(p)
    p["N_ncdm"] = 1
    p["m_ncdm"] = 0.06
    return p

# -----------------------------
# Run with auto-fixes
# -----------------------------
def run_with_autofix(ClassObj, params_in, label, max_passes=MAX_PASSES):
    params = dict(params_in)
    stripped_total = []
    reco_fallback_used = False

    for attempt in range(1, max_passes + 1):
        c = ClassObj()
        c.set(params)

        try:
            c.compute()
            print(f"\n=== {label}: compute OK after {attempt} pass(es) ===")
            if stripped_total:
                print(f"{label}: stripped unread keys:", ", ".join(stripped_total))
            else:
                print(f"{label}: no keys stripped")
            if reco_fallback_used:
                print(f"{label}: recombination fallback used -> recfast")
            return c, params, stripped_total

        except Exception as e:
            msg = str(e)

            # (A) HyRec missing -> force recfast and retry
            if _hyrec_missing_re.search(msg):
                print(f"\n=== {label}: HyRec not available; switching recombination -> recfast ===")
                params["recombination"] = "recfast"
                reco_fallback_used = True
                continue

            # (B) Unread parameters -> strip and retry
            unread = parse_unread_params(e)
            if unread:
                print(f"\n=== {label}: pass {attempt} -> unread keys detected ===")
                print("Unread:", ", ".join(unread))

                removed_now = []
                for k in unread:
                    if k in params:
                        params.pop(k, None)
                        removed_now.append(k)

                if not removed_now:
                    raise RuntimeError(
                        f"{label}: CLASS reported unread keys {unread}, "
                        f"but none were present in params dict."
                    )

                stripped_total.extend(removed_now)
                continue

            # (C) Some other error -> raise
            print(f"\n=== {label}: compute FAILED (non-unread-params error) ===")
            raise

    raise RuntimeError(f"{label}: exceeded max_passes={max_passes}")

# -----------------------------
# Interpolated background access
# -----------------------------
def interp_1d(x_grid, y_grid, x_req):
    """
    Safe linear interpolation y(x_req) from tabulated (x_grid, y_grid).
    Handles decreasing grids by reversing.
    Drops non-finite points.
    Clamps outside range to endpoints.
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 2:
        return np.nan

    # make x increasing
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # clamp
    if x_req <= x[0]:
        return float(y[0])
    if x_req >= x[-1]:
        return float(y[-1])

    return float(np.interp(x_req, x, y))

def bg_row_at_z_interp(c, z_req):
    """
    Interpolate background columns to EXACT same requested z for each cosmology.
    """
    bg = c.get_background()
    z_grid = np.asarray(bg["z"], dtype=float)

    row = {
        "z_req": float(z_req),
        "z_min_table": float(np.min(z_grid)),
        "z_max_table": float(np.max(z_grid)),
    }

    wanted = [
        "H [1/Mpc]",
        "H [km/s/Mpc]",
        "(.)rho_tot",
        "(.)rho_crit",
        "(.)rho_g",
        "(.)rho_ur",
        "(.)rho_ncdm",
        "(.)rho_b",
        "(.)rho_cdm",
        "(.)rho_lambda",
        "(.)rho_fld",
        "(.)Omega_m",
        "(.)Omega_r",
        "(.)Omega_lambda",
        "(.)Omega_fld",
    ]

    for k in wanted:
        if k in bg:
            row[k] = interp_1d(z_grid, bg[k], z_req)

    return row

# -----------------------------
# Comparison + diagnostics
# -----------------------------
def closure(row):
    comp_keys = [
        "(.)rho_g",
        "(.)rho_ur",
        "(.)rho_ncdm",
        "(.)rho_b",
        "(.)rho_cdm",
        "(.)rho_lambda",
        "(.)rho_fld",
    ]
    rho_sum = 0.0
    used = 0
    for k in comp_keys:
        if k in row and np.isfinite(row[k]):
            rho_sum += row[k]
            used += 1

    out = {}
    if "(.)rho_tot" in row and used > 0 and row["(.)rho_tot"] != 0:
        out["rho_sum/rho_tot - 1"] = rho_sum / row["(.)rho_tot"] - 1.0
    else:
        out["rho_sum/rho_tot - 1"] = np.nan

    Omega_sum = 0.0
    usedO = 0
    for k in ["(.)Omega_m", "(.)Omega_r", "(.)Omega_lambda", "(.)Omega_fld"]:
        if k in row and np.isfinite(row[k]):
            Omega_sum += row[k]
            usedO += 1
    out["Omega_sum - 1"] = Omega_sum - 1.0 if usedO > 0 else np.nan

    if "(.)rho_tot" in row and "(.)rho_crit" in row and np.isfinite(row.get("(.)rho_tot", np.nan)) and np.isfinite(row.get("(.)rho_crit", np.nan)) and row["(.)rho_crit"] != 0:
        out["rho_tot/rho_crit - 1"] = row["(.)rho_tot"] / row["(.)rho_crit"] - 1.0
    else:
        out["rho_tot/rho_crit - 1"] = np.nan

    return out

def compare_rows(rowT, rowM, rtol=RTOL_BG):
    z = rowT["z_req"]
    print(f"\n=== Background compare at z={z:g} (INTERPOLATED) ===")
    print(f"Thomas table z-range=[{rowT['z_min_table']:.3g}, {rowT['z_max_table']:.3g}]")
    print(f"Mine   table z-range=[{rowM['z_min_table']:.3g}, {rowM['z_max_table']:.3g}]")

    keys = sorted(set(rowT.keys()) & set(rowM.keys()))
    bad = 0

    for k in keys:
        if k in ["z_req", "z_min_table", "z_max_table"]:
            continue
        a = rowT[k]
        b = rowM[k]
        if np.isfinite(a) and np.isfinite(b):
            if not np.isclose(a, b, rtol=rtol, atol=0.0):
                bad += 1
                rel = (a - b) / b if b != 0 else np.nan
                print(f"DIFF {k:16s}: Thomas={a:.6e}  Mine={b:.6e}  rel={rel:.3e}")

    if bad == 0:
        print("All overlapping background columns match within tolerance.")

    cT = closure(rowT)
    cM = closure(rowM)
    print("Closure:")
    for k in sorted(cT.keys()):
        print(f"  {k:22s}  Thomas={cT[k]:.3e}  Mine={cM[k]:.3e}")

def print_param_diff(pA, pB, nameA="A", nameB="B"):
    keys = sorted(set(pA.keys()) | set(pB.keys()))
    diffs = []
    for k in keys:
        vA = pA.get(k, "<missing>")
        vB = pB.get(k, "<missing>")
        if vA != vB:
            diffs.append((k, vA, vB))
    print(f"\n--- FINAL PARAM DIFF ({nameA} vs {nameB}) ---")
    if not diffs:
        print("none")
        return
    for k, vA, vB in diffs:
        print(f"{k:28s}  {nameA}: {vA}   |   {nameB}: {vB}")

# -----------------------------
# Main
# -----------------------------
def main():
    print("\n#############################")
    print("# Gate1 / Phase1: Background (INTERPOLATED) — Thomas vs Mine (legacy ncdm)")
    print("#############################")

    params = legacy_ncdm(base_common())

    cosmo_T, pT_final, _ = run_with_autofix(ClassThomas, params, "Thomas(legacy)")
    cosmo_M, pM_final, _ = run_with_autofix(ClassMine,   params, "Mine(legacy)")

    print_param_diff(pT_final, pM_final, "Thomas(final)", "Mine(final)")

    for z in Z_LIST:
        rowT = bg_row_at_z_interp(cosmo_T, z)
        rowM = bg_row_at_z_interp(cosmo_M, z)
        compare_rows(rowT, rowM, rtol=RTOL_BG)

    cosmo_T.struct_cleanup(); cosmo_T.empty()
    cosmo_M.struct_cleanup(); cosmo_M.empty()

if __name__ == "__main__":
    main()




    #!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from classy import Class as ClassThomas
from classy_NEDE import Class as ClassMine

def params_common_pk():
    """
    Minimal parameter set that BOTH Thomas and Mine accept,
    avoiding keys Thomas rejects (non_linear, bbn, G_eff_ncdm, split keys, HyRec).
    """
    return {
        # background
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "Omega_k": 0.0,
        "T_cmb": 2.7255,

        # primordial
        "A_s": 2.1e-9,
        "n_s": 0.9649,
        "tau_reio": 0.0543,

        # neutrinos (legacy)
        "N_ur": 3.046,
        "N_ncdm": 1,
        "m_ncdm": 0.06,

        # output for P(k)
        "output": "mPk",
        "lensing": "no",
        "gauge": "synchronous",

        # make sure CLASS builds a pk table over our k-range
        "P_k_max_1/Mpc": 10.0,
        "z_pk": 0.0,

        # recombination engine that Thomas definitely supports
        "recombination": "recfast",
    }

def run_cosmo(Cls, params, label):
    print(f"\n=== Running {label} ===")
    c = Cls()
    c.set(params)
    c.compute()
    return c

def pk_lin_safe(cosmo, k, z):
    # Some wrappers have pk_lin; others only pk
    if hasattr(cosmo, "pk_lin"):
        return cosmo.pk_lin(k, z)
    return cosmo.pk(k, z)

def main():
    p = params_common_pk()

    cosmo_T = run_cosmo(ClassThomas, p, "Thomas")
    cosmo_M = run_cosmo(ClassMine,   p, "Mine")

    # k-grid in 1/Mpc
    k_1Mpc = np.logspace(-4, 1.0, 300)

    # Compute P(k) in CLASS native units: P(k) in (Mpc)^3 when k in 1/Mpc
    z = 0.0
    P_T = np.array([pk_lin_safe(cosmo_T, k, z) for k in k_1Mpc])
    P_M = np.array([pk_lin_safe(cosmo_M, k, z) for k in k_1Mpc])

    # Convert both to h/Mpc convention for plotting (optional but nice)
    h_ref = cosmo_T.h()
    k_hMpc = k_1Mpc / h_ref
    P_T_h = P_T * h_ref**3
    P_M_h = P_M * h_ref**3

    # Plot P(k)
    plt.figure(figsize=(7, 5))
    plt.loglog(k_hMpc, P_T_h, label="Thomas")
    plt.loglog(k_hMpc, P_M_h, ls="--", label="Mine")
    plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    plt.ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
    plt.title("Linear matter power spectrum at z=0")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot ratio
    ratio = P_M_h / P_T_h
    plt.figure(figsize=(7, 3.8))
    plt.semilogx(k_hMpc, ratio)
    plt.axhline(1.0, ls=":")
    plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    plt.ylabel("Mine / Thomas")
    plt.title("P(k) ratio (linear, z=0)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    print("\n=== Summary ===")
    print("h(Thomas) =", cosmo_T.h(), "  h(Mine) =", cosmo_M.h())
    print("max |ratio-1| =", np.max(np.abs(ratio - 1)))
    print("median |ratio-1| =", np.median(np.abs(ratio - 1)))

    cosmo_T.struct_cleanup(); cosmo_T.empty()
    cosmo_M.struct_cleanup(); cosmo_M.empty()

if __name__ == "__main__":
    main()