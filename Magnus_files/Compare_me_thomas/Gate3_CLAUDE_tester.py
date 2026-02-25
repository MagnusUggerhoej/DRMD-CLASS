#!/usr/bin/env python3
"""
Gate 3 (direct): Absolute comparison of interacting transfer functions
======================================================================
Tests:
  mismatch_direct(k, z) = Mine_INT(k, z) / Thomas_INT(k, z) - 1

This is the direct complement to the effect-ratio script. Instead of
comparing how each build *responds* to interactions, this compares the
actual interacting transfer function values directly between the two builds.

A systematic error shared by both FS baselines would not show up in the
effect-ratio script but WILL show up here.

Pass condition: mismatch_direct < rtol at all (k, z) for all variables.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from classy      import Class as ClassThomas
from classy_NEDE import Class as ClassMine

# ─────────────────────────────────────────────
# Knobs
# ─────────────────────────────────────────────
G_EFF_ON  = 1e-1
M_NCDM_EV = 0.06
DEG_NCDM  = 1.0

K_PROBE   = [1e-3, 1e-2, 1e-1]   # 1/Mpc

Z_GRID = np.unique(np.sort(np.concatenate([
    np.logspace(np.log10(0.01), np.log10(10),   15),
    np.logspace(np.log10(10),   np.log10(100),  15),
    np.logspace(np.log10(100),  np.log10(1000), 20),
])))

TRANSFER_COLS = [
    "d_cdm",
    "d_b",
    "d_ncdm[0]",
    "phi",
    "psi",
]

RTOL_WARN = 1e-3

# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
_unread_re        = re.compile(r"did not read input parameter\(s\):\s*(.*)", re.IGNORECASE)
_hyrec_missing_re = re.compile(r"compiled without including the HyRec code",  re.IGNORECASE)

def parse_unread(exc):
    m = _unread_re.search(str(exc))
    if not m: return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]

def run_with_autofix(ClassObj, params_in, label, max_passes=10):
    params = dict(params_in)
    stripped = []
    for attempt in range(1, max_passes + 1):
        c = ClassObj(); c.set(params)
        try:
            c.compute()
            print(f"  [{label}] OK (pass {attempt})"
                  + (f" stripped={stripped}" if stripped else ""))
            return c, stripped
        except Exception as e:
            msg = str(e)
            if _hyrec_missing_re.search(msg):
                params["recombination"] = "recfast"; continue
            unread = parse_unread(e)
            if unread:
                removed = [k for k in unread if k in params]
                if not removed: raise RuntimeError(f"{label}: unread {unread} not in dict.")
                for k in removed: params.pop(k)
                stripped.extend(removed); continue
            print(f"  [{label}] FAILED"); raise
    raise RuntimeError(f"{label}: exceeded {max_passes} passes.")

# ─────────────────────────────────────────────
# Param builders
# ─────────────────────────────────────────────
def base():
    return {
        "output":        "mTk,vTk",
        "lensing":       "no",
        "gauge":         "synchronous",
        "recombination": "recfast",
        "h":             0.6736,
        "omega_b":       0.02237,
        "omega_cdm":     0.1200,
        "A_s":           2.1e-9,
        "n_s":           0.9649,
        "tau_reio":      0.0543,
        "T_cmb":         2.7255,
        "Omega_k":       0.0,
        "G_eff_ur":      0.0,
        "z_max_pk":      max(Z_GRID) * 1.05,
    }

def params_thomas_int():
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_interacting": 1, "m_ncdm_interacting": M_NCDM_EV,
              "deg_ncdm_interacting": DEG_NCDM, "G_eff_ncdm_interacting": G_EFF_ON})
    return p

def params_mine_int():
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_standard": 0, "N_ncdm_interacting": 1,
              "m_ncdm": M_NCDM_EV, "G_eff_ncdm_interacting": G_EFF_ON})
    return p

# ─────────────────────────────────────────────
# Transfer function helpers
# ─────────────────────────────────────────────
def get_transfer(cosmo, z):
    try:    return cosmo.get_transfer(z=z, output_format='class')
    except TypeError: return cosmo.get_transfer(z=z)

def get_k_grid(tr):
    for key in ["k (h/Mpc)", "k"]:
        if key in tr: return np.asarray(tr[key], dtype=float)
    raise KeyError("No k column in transfer dict.")

def interp_at_k(tr, col, k_target):
    if col not in tr: return np.nan
    k = get_k_grid(tr)
    y = np.asarray(tr[col], dtype=float)
    order = np.argsort(k)
    return float(np.interp(k_target, k[order], y[order]))

def extract_time_series(cosmo, k_probe, cols, z_grid):
    result = {c: np.full(len(z_grid), np.nan) for c in cols}
    for i, z in enumerate(z_grid):
        try:
            tr = get_transfer(cosmo, z)
            for c in cols:
                result[c][i] = interp_at_k(tr, c, k_probe)
        except Exception:
            pass
    return result

# ─────────────────────────────────────────────
# Direct mismatch
# ─────────────────────────────────────────────
def compute_direct_mismatch(ts_T, ts_M, cols):
    """
    mismatch_direct[col][z] = Mine_INT(k,z) / Thomas_INT(k,z) - 1
    Uses max(|T|, |M|) as denominator to handle near-zero values safely.
    """
    mismatch = {}
    for c in cols:
        T = ts_T[c]
        M = ts_M[c]
        denom = np.where(
            np.maximum(np.abs(T), np.abs(M)) > 1e-30,
            T,
            np.nan
        )
        mismatch[c] = M / denom - 1.0
    return mismatch

# ─────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────
def print_summary(k, cols, mismatch):
    print(f"\n  k = {k:.1e} 1/Mpc")
    print(f"  {'Column':<20s}  {'max|mismatch|':>14s}  {'rms|mismatch|':>14s}  status")
    print(f"  {'─'*20}  {'─'*14}  {'─'*14}  {'─'*10}")
    for c in cols:
        m = mismatch[c]
        valid = m[np.isfinite(m)]
        if len(valid) == 0:
            print(f"  {c:<20s}  {'(no data)':>14s}"); continue
        mx  = np.max(np.abs(valid))
        rms = np.sqrt(np.mean(valid**2))
        flag = "← WARN" if mx > RTOL_WARN else "✓"
        print(f"  {c:<20s}  {mx:>14.3e}  {rms:>14.3e}  {flag}")

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_direct(k, cols, z_grid, ts_T, ts_M, mismatch):
    active = [c for c in cols if np.any(np.isfinite(ts_T[c])) or np.any(np.isfinite(ts_M[c]))]
    if not active: return

    n = len(active)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.2 * n), squeeze=False)
    fig.suptitle(f"Gate 3 (direct): k = {k:.1e} 1/Mpc,  G_eff = {G_EFF_ON:.1e}", fontsize=11)

    for row, c in enumerate(active):
        ax_abs, ax_mis = axes[row]

        # Left: absolute values for Thomas INT and Mine INT
        for label, ts, ls, col in [
            ("Thomas INT", ts_T, "-",  "C0"),
            ("Mine   INT", ts_M, "--", "C1"),
        ]:
            y = ts[c]; mask = np.isfinite(y)
            if mask.any():
                ax_abs.plot(1 + z_grid[mask], y[mask], ls=ls, color=col,
                            label=label, lw=1.4)
        ax_abs.set_xscale("log")
        ax_abs.set_ylabel(c)
        ax_abs.legend(fontsize=7)
        ax_abs.grid(True, ls=":", which="both")
        if row == 0: ax_abs.set_title("Absolute value (INT)")

        # Right: direct mismatch
        y = mismatch[c]; mask = np.isfinite(y)
        if mask.any():
            ax_mis.plot(1 + z_grid[mask], y[mask], c="C2", lw=1.4)
        ax_mis.axhline( RTOL_WARN, ls="--", c="r", lw=0.8)
        ax_mis.axhline(-RTOL_WARN, ls="--", c="r", lw=0.8)
        ax_mis.axhline(0, ls=":", c="k", lw=0.8)
        ax_mis.set_xscale("log")
        ax_mis.set_ylabel(r"Mine$_{\rm INT}$/Thomas$_{\rm INT}$ − 1")
        ax_mis.grid(True, ls=":", which="both")
        if row == 0: ax_mis.set_title(f"Direct mismatch  (rtol={RTOL_WARN:.0e})")

        for ax in (ax_abs, ax_mis):
            ax.set_xlabel("1 + z")

    plt.tight_layout()
    plt.savefig(f"gate3_direct_k{k:.0e}.png", dpi=120, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print(f"  Gate 3 (direct): Mine_INT vs Thomas_INT")
    print(f"  G_eff = {G_EFF_ON:.1e},  m_ncdm = {M_NCDM_EV} eV")
    print(f"  k values: {K_PROBE} 1/Mpc")
    print("█"*60 + "\n")

    print("Running two interacting models...")
    cT_int, _ = run_with_autofix(ClassThomas, params_thomas_int(), "Thomas INT")
    cM_int, _ = run_with_autofix(ClassMine,   params_mine_int(),   "Mine   INT")

    # Print available columns once
    tr_T = get_transfer(cT_int, Z_GRID[len(Z_GRID)//2])
    tr_M = get_transfer(cM_int, Z_GRID[len(Z_GRID)//2])
    cols_T = [c for c in sorted(tr_T) if c not in ("k (h/Mpc)", "k") and "\x00" not in c]
    cols_M = [c for c in sorted(tr_M) if c not in ("k (h/Mpc)", "k")]
    print(f"\n  Transfer cols Thomas: {cols_T}")
    print(f"  Transfer cols Mine:   {cols_M}")

    # Only compare columns present in both
    common_cols = [c for c in TRANSFER_COLS if c in cols_T and c in cols_M]
    missing = [c for c in TRANSFER_COLS if c not in common_cols]
    if missing:
        print(f"\n  Note: {missing} absent in one or both builds — skipping.")

    print("\n" + "═"*60)
    print("  GATE 3 (DIRECT) SUMMARY")
    print("  mismatch = Mine_INT / Thomas_INT - 1")
    print("═"*60)

    for k in K_PROBE:
        print(f"\n  Extracting time series at k = {k:.1e} 1/Mpc ...")
        ts_T = extract_time_series(cT_int, k, common_cols, Z_GRID)
        ts_M = extract_time_series(cM_int, k, common_cols, Z_GRID)
        mismatch = compute_direct_mismatch(ts_T, ts_M, common_cols)
        print_summary(k, common_cols, mismatch)
        plot_direct(k, common_cols, Z_GRID, ts_T, ts_M, mismatch)

    for c in (cT_int, cM_int):
        try: c.struct_cleanup(); c.empty()
        except: pass


if __name__ == "__main__":
    main()