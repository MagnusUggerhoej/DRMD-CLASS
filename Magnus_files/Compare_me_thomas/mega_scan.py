#!/usr/bin/env python3
"""
Gate 3: Fixed-k perturbation time evolution
============================================
Compare perturbation variables as functions of redshift at fixed k.

Strategy (effect-style, matching supervisor's approach):
  For each k:
    effect_T(k, z) = X_T_INT(k, z) / X_T_FS(k, z) - 1
    effect_M(k, z) = X_M_INT(k, z) / X_M_FS(k, z) - 1
    mismatch(k, z) = effect_M / effect_T - 1

  If mismatch ~ 0 at all (k, z): collision operator and hierarchy are consistent.
  If mismatch grows at specific z: pinpoints the epoch where implementations diverge.
  If mismatch is k-dependent: points to a scale-dependent bug (fluid/exact switchover,
    truncation, or taudot formula).

Variables compared (synchronous gauge, CLASS naming):
  d_cdm, d_b       : density contrasts
  t_cdm, t_b       : velocity divergences
  d_ncdm[0]        : ncdm density contrast
  t_ncdm[0]        : ncdm velocity divergence
  shear_ncdm[0]    : ncdm anisotropic stress (if exposed)
  phi, psi         : metric potentials
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from classy      import Class as ClassThomas
from classy_NEDE import Class as ClassMine

# ─────────────────────────────────────────────
# Knobs
# ─────────────────────────────────────────────
G_EFF_ON  = 1e-1          # coupling to test in Gate 3
M_NCDM_EV = 0.06
DEG_NCDM  = 1.0

# k values to probe (1/Mpc) — one per decade, all linear regime
K_PROBE   = [1e-3, 1e-2, 1e-1]

# Redshift grid for time evolution
# Dense near recombination and neutrino NR transition (~z=360 for 0.06 eV)
Z_GRID = np.concatenate([
    np.logspace(np.log10(0.01), np.log10(10),    15),
    np.logspace(np.log10(10),   np.log10(100),   15),
    np.logspace(np.log10(100),  np.log10(1000),  20),
])
Z_GRID = np.unique(np.sort(Z_GRID))

# Transfer function columns to compare
# CLASS synchronous gauge naming — we try each and skip absent ones
TRANSFER_COLS = [
    "d_cdm",
    "d_b",
    "t_b",
    "d_ncdm[0]",
    "t_ncdm[0]",
    "shear_ncdm[0]",
    "phi",
    "psi",
]

RTOL_WARN = 1e-3   # flag mismatches above this in summary table

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
        "output":        "mTk",          # transfer functions only, avoids delta_m gauge issue
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

def params_thomas_fs():
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_interacting": 1, "m_ncdm_interacting": M_NCDM_EV,
              "deg_ncdm_interacting": DEG_NCDM, "G_eff_ncdm_interacting": 0.0})
    return p

def params_thomas_int():
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_interacting": 1, "m_ncdm_interacting": M_NCDM_EV,
              "deg_ncdm_interacting": DEG_NCDM, "G_eff_ncdm_interacting": G_EFF_ON})
    return p

def params_mine_fs():
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_standard": 1, "N_ncdm_interacting": 0, "m_ncdm": M_NCDM_EV})
    return p

def params_mine_int():
    p = base()
    p["N_ur"] = max(0.0, 3.046 - DEG_NCDM)
    p.update({"N_ncdm_standard": 0, "N_ncdm_interacting": 1,
              "m_ncdm": M_NCDM_EV, "G_eff_ncdm_interacting": G_EFF_ON})
    return p

# ─────────────────────────────────────────────
# Transfer function extraction
# ─────────────────────────────────────────────
def get_transfer(cosmo, z):
    """Get transfer dict at redshift z, trying both output_format options."""
    try:
        return cosmo.get_transfer(z=z, output_format='class')
    except TypeError:
        return cosmo.get_transfer(z=z)

def get_k_grid(tr):
    """Return k array (1/Mpc) from transfer dict."""
    for key in ["k (h/Mpc)", "k"]:
        if key in tr:
            return np.asarray(tr[key], dtype=float)
    raise KeyError("No k column found in transfer dict.")

def interp_at_k(tr, col, k_target):
    """Interpolate transfer column at scalar k_target (1/Mpc). NaN if absent."""
    if col not in tr: return np.nan
    k  = get_k_grid(tr)
    y  = np.asarray(tr[col], dtype=float)
    order = np.argsort(k)
    return float(np.interp(k_target, k[order], y[order]))

def extract_time_series(cosmo, k_probe, cols, z_grid):
    """
    For each z in z_grid, get transfer function and interpolate each col at k_probe.
    Returns dict: col -> array over z_grid.
    """
    result = {c: np.full(len(z_grid), np.nan) for c in cols}
    for i, z in enumerate(z_grid):
        try:
            tr = get_transfer(cosmo, z)
            for c in cols:
                result[c][i] = interp_at_k(tr, c, k_probe)
        except Exception as e:
            pass   # leave as NaN for this z
    return result

# ─────────────────────────────────────────────
# Effect + mismatch computation
# ─────────────────────────────────────────────
def compute_effect(ts_fs, ts_int, cols):
    """effect[col][z] = ts_int[col][z] / ts_fs[col][z] - 1, safe near zero."""
    effect = {}
    for c in cols:
        fs  = ts_fs[c]
        itr = ts_int[c]
        denom = np.where(np.abs(fs) > 1e-30, fs, np.nan)
        effect[c] = itr / denom - 1.0
    return effect

def compute_mismatch(eff_T, eff_M, cols):
    """mismatch[col][z] = eff_M[col][z] / eff_T[col][z] - 1, safe near zero."""
    mismatch = {}
    for c in cols:
        denom = np.where(np.abs(eff_T[c]) > 1e-6, eff_T[c], np.nan)
        mismatch[c] = eff_M[c] / denom - 1.0
    return mismatch

# ─────────────────────────────────────────────
# Printing + plotting
# ─────────────────────────────────────────────
def print_summary(k, cols, mismatch, z_grid):
    print(f"\n  k = {k:.1e} 1/Mpc")
    print(f"  {'Column':<20s}  {'max|mismatch|':>14s}  {'rms|mismatch|':>14s}  {'status'}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*14}  {'─'*10}")
    for c in cols:
        m = mismatch[c]
        valid = m[np.isfinite(m)]
        if len(valid) == 0:
            print(f"  {c:<20s}  {'(no data)':>14s}")
            continue
        mx  = np.max(np.abs(valid))
        rms = np.sqrt(np.mean(valid**2))
        flag = "← WARN" if mx > RTOL_WARN else "✓"
        print(f"  {c:<20s}  {mx:>14.3e}  {rms:>14.3e}  {flag}")

def plot_gate3(k, cols, z_grid,
               eff_T, eff_M, mismatch,
               ts_T_fs, ts_T_int, ts_M_fs, ts_M_int):
    """One figure per k value with 3 panels per column."""

    # Filter to columns that have actual data
    active_cols = [c for c in cols
                   if np.any(np.isfinite(ts_T_fs[c])) or np.any(np.isfinite(ts_M_fs[c]))]
    if not active_cols:
        print(f"  No data to plot for k={k:.1e}")
        return

    n = len(active_cols)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.2 * n), squeeze=False)
    fig.suptitle(f"Gate 3: k = {k:.1e} 1/Mpc,  G_eff = {G_EFF_ON:.1e}",
                 fontsize=12, y=1.01)

    for row, c in enumerate(active_cols):
        ax_abs, ax_eff, ax_mis = axes[row]

        # Panel 1: absolute values (FS and INT for both builds)
        for label, ts, ls, col in [
                ("Thomas FS",  ts_T_fs,  "-",  "C0"),
                ("Thomas INT", ts_T_int, "--", "C0"),
                ("Mine FS",    ts_M_fs,  "-",  "C1"),
                ("Mine INT",   ts_M_int, "--", "C1"),
        ]:
            y = ts[c]
            mask = np.isfinite(y)
            if mask.any():
                ax_abs.plot(1 + z_grid[mask], y[mask], ls=ls, color=col,
                            label=label, lw=1.2)
        ax_abs.set_xscale("log"); ax_abs.set_ylabel(c)
        ax_abs.legend(fontsize=6); ax_abs.grid(True, ls=":", which="both")
        if row == 0: ax_abs.set_title("Absolute value")

        # Panel 2: effect (INT/FS - 1) for Thomas and Mine
        for label, eff, ls, col in [
                ("Thomas", eff_T, "-",  "C0"),
                ("Mine",   eff_M, "--", "C1"),
        ]:
            y = eff[c]
            mask = np.isfinite(y)
            if mask.any():
                ax_eff.plot(1 + z_grid[mask], y[mask], ls=ls, color=col,
                            label=label, lw=1.2)
        ax_eff.axhline(0, ls=":", c="k", lw=0.8)
        ax_eff.set_xscale("log"); ax_eff.set_ylabel("INT/FS − 1")
        ax_eff.legend(fontsize=6); ax_eff.grid(True, ls=":", which="both")
        if row == 0: ax_eff.set_title("Interaction effect")

        # Panel 3: mismatch
        y = mismatch[c]
        mask = np.isfinite(y)
        if mask.any():
            ax_mis.plot(1 + z_grid[mask], y[mask], c="C2", lw=1.2)
        ax_mis.axhline( RTOL_WARN, ls="--", c="r", lw=0.8)
        ax_mis.axhline(-RTOL_WARN, ls="--", c="r", lw=0.8)
        ax_mis.axhline(0, ls=":", c="k", lw=0.8)
        ax_mis.set_xscale("log"); ax_mis.set_ylabel(r"$r_M/r_T - 1$")
        ax_mis.grid(True, ls=":", which="both")
        if row == 0: ax_mis.set_title(f"Mismatch  (rtol={RTOL_WARN:.0e})")

        for ax in (ax_abs, ax_eff, ax_mis):
            ax.set_xlabel("1 + z")

    plt.tight_layout()
    plt.savefig(f"gate3_k{k:.0e}.png", dpi=120, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print(f"  Gate 3: perturbation time evolution")
    print(f"  G_eff = {G_EFF_ON:.1e},  m_ncdm = {M_NCDM_EV} eV")
    print(f"  k values: {K_PROBE} 1/Mpc")
    print(f"  z range: [{Z_GRID.min():.2f}, {Z_GRID.max():.0f}]  ({len(Z_GRID)} points)")
    print("█"*60 + "\n")

    print("Running four models...")
    cT_fs,  _ = run_with_autofix(ClassThomas, params_thomas_fs(),  "Thomas FS")
    cT_int, _ = run_with_autofix(ClassThomas, params_thomas_int(), "Thomas INT")
    cM_fs,  _ = run_with_autofix(ClassMine,   params_mine_fs(),    "Mine   FS")
    cM_int, _ = run_with_autofix(ClassMine,   params_mine_int(),   "Mine   INT")

    # Print available transfer columns once
    tr_sample = get_transfer(cT_fs, Z_GRID[len(Z_GRID)//2])
    print(f"\n  Available transfer cols (Thomas): {[c for c in sorted(tr_sample) if c != 'k (h/Mpc)' and c != 'k']}")
    tr_sample_M = get_transfer(cM_fs, Z_GRID[len(Z_GRID)//2])
    print(f"  Available transfer cols (Mine):   {[c for c in sorted(tr_sample_M) if c != 'k (h/Mpc)' and c != 'k']}")

    print("\n" + "═"*60)
    print("  GATE 3 SUMMARY")
    print("═"*60)

    for k in K_PROBE:
        print(f"\n  Extracting time series at k = {k:.1e} 1/Mpc ...")

        ts_T_fs  = extract_time_series(cT_fs,  k, TRANSFER_COLS, Z_GRID)
        ts_T_int = extract_time_series(cT_int, k, TRANSFER_COLS, Z_GRID)
        ts_M_fs  = extract_time_series(cM_fs,  k, TRANSFER_COLS, Z_GRID)
        ts_M_int = extract_time_series(cM_int, k, TRANSFER_COLS, Z_GRID)

        eff_T    = compute_effect(ts_T_fs, ts_T_int, TRANSFER_COLS)
        eff_M    = compute_effect(ts_M_fs, ts_M_int, TRANSFER_COLS)
        mismatch = compute_mismatch(eff_T, eff_M, TRANSFER_COLS)

        print_summary(k, TRANSFER_COLS, mismatch, Z_GRID)
        plot_gate3(k, TRANSFER_COLS, Z_GRID,
                   eff_T, eff_M, mismatch,
                   ts_T_fs, ts_T_int, ts_M_fs, ts_M_int)

    for c in (cT_fs, cT_int, cM_fs, cM_int):
        try: c.struct_cleanup(); c.empty()
        except: pass


if __name__ == "__main__":
    main()