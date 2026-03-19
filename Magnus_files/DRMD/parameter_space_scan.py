#!/usr/bin/env python3

import csv
import os
import numpy as np

from classy_NEDE import Class as ClassMine
from classy_tobias import Class as ClassTobias


# =============================================================================
# Helpers
# =============================================================================

CSV_OUTPUT_DIR = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/Magnus_files/DRMD/csv files"

def pk_safe(cosmo, k_1Mpc_arr, z):
    fn = cosmo.pk_lin if hasattr(cosmo, "pk_lin") else cosmo.pk
    return np.array([fn(float(k), float(z)) for k in k_1Mpc_arr])

def get_derived_safe(cosmo):
    out = {}
    for key in ["h", "Omega_m", "Omega_r", "z_eq"]:
        try:
            out[key] = cosmo.get_current_derived_parameters([key])[key]
        except Exception:
            pass

    try:
        out["h_direct"] = cosmo.h()
    except Exception:
        pass

    try:
        out["Neff"] = cosmo.get_current_derived_parameters(["Neff"])["Neff"]
    except Exception:
        pass

    return out

def run_model(ClassObj, params, label):
    c = ClassObj()
    c.set(params)
    c.compute()
    print(f"[{label}] computed.")
    d = get_derived_safe(c)
    if d:
        print(f"[{label}] derived: {d}")
    return c

def cleanup(*cosmos):
    for c in cosmos:
        try:
            c.struct_cleanup()
            c.empty()
        except Exception:
            pass

def pct_diff(a, b):
    return 100.0 * (a / b - 1.0)

def fmt_float(x, ndp=6):
    return f"{x:.{ndp}f}"

def fmt_sci(x, ndp=6):
    return f"{x:.{ndp}e}"

def print_separator(width=120):
    print("-" * width)


# =============================================================================
# Shared cosmology
# =============================================================================
H0        = 67.32
omega_b   = 0.02238
omega_cdm = 0.1201
A_s       = 2.101e-9
n_s       = 0.9660
tau_reio  = 0.0543
k_pivot   = 0.05
T_cmb     = 2.7255
Omega_k   = 0.0

Z_PK   = 0.0
L_MAX  = 2500

# Baseline interacting-ncdm / DRMD point
M_NCDM_EV        = 0.06
DEG_NCDM_BASE    = 1.0
G_EFF_NCDM_ON    = 1e-1

DELTA_NEFF_ON    = 0.8
F_IDM_BASE       = 0.03
Z_STOP_ON        = 5.0e4
G_AH_ON          = 1.0e9

# Selected physically motivated k values [1/Mpc]
K_PRINT_1MPC = np.array([
    1e-2,
    5e-2,
    2e-1,
    1.0,
], dtype=float)


# =============================================================================
# Scan definitions
# =============================================================================
DEG_SCAN_VALUES = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0]
FIDM_SCAN_VALUES = [0.00, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.24, 0.30]


# =============================================================================
# Base params
# =============================================================================
def base_params():
    return {
        "H0": float(H0),
        "omega_b": float(omega_b),
        "omega_cdm": float(omega_cdm),
        "A_s": float(A_s),
        "n_s": float(n_s),
        "tau_reio": float(tau_reio),
        "k_pivot": float(k_pivot),
        "T_cmb": float(T_cmb),
        "Omega_k": float(Omega_k),

        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "l_max_scalars": int(L_MAX),
        "P_k_max_1/Mpc": float(np.max(K_PRINT_1MPC) * 1.20),

        "recombination": "recfast",
        "reio_parametrization": "reio_camb",
    }


# =============================================================================
# Parameter builders
# =============================================================================
def build_tobias_params(deg_ncdm_interacting, f_idm_drmd):
    n_ur_run = max(0.0, 3.046 - float(deg_ncdm_interacting))

    p = base_params()
    p.update({
        "N_ur"                   : float(n_ur_run),

        "N_ncdm_interacting"     : 1,
        "m_ncdm_interacting"     : float(M_NCDM_EV),
        "deg_ncdm_interacting"   : float(deg_ncdm_interacting),
        "G_eff_ncdm_interacting" : float(G_EFF_NCDM_ON),

        "f_idm_drmd"             : float(f_idm_drmd),
        "delta_Neff_drmd"        : float(DELTA_NEFF_ON),
        "z_stop"                 : float(Z_STOP_ON),
        "G_over_aH_drmd_ini"     : float(G_AH_ON),
    })

    for key in ["N_ncdm", "m_ncdm", "deg_ncdm", "T_ncdm"]:
        p.pop(key, None)

    return p


def build_mine_params(deg_ncdm_interacting, f_idm_drmd):
    n_ur_run = max(0.0, 3.046 - float(deg_ncdm_interacting))

    p = base_params()
    p.update({
        "N_ur"                   : float(n_ur_run),

        "N_ncdm_standard"        : 0,
        "N_ncdm_interacting"     : 1,
        "m_ncdm"                 : float(M_NCDM_EV),
        "deg_ncdm"               : float(deg_ncdm_interacting),
        "G_eff_ncdm_interacting" : float(G_EFF_NCDM_ON),

        "f_idm_drmd"             : float(f_idm_drmd),
        "delta_Neff_drmd"        : float(DELTA_NEFF_ON),
        "z_stop"                 : float(Z_STOP_ON),
        "G_over_aH_drmd_ini"     : float(G_AH_ON),
    })

    p.pop("N_ncdm", None)

    return p


# =============================================================================
# Printing helpers
# =============================================================================
def print_scan_header(title, fixed_text):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)
    print(fixed_text)
    print("Selected k values [1/Mpc]: " + ", ".join(fmt_sci(k, 2) for k in K_PRINT_1MPC))

def print_point_summary(scan_name, scan_value, d_mine, d_tob):
    print(f"\n[{scan_name} = {scan_value}] derived summary")
    print_separator(84)
    print(f"{'Code':<10} {'h':>12} {'Omega_m':>14} {'z_eq':>14} {'Neff':>14}")
    print_separator(84)

    def f(d, key):
        x = d.get(key, np.nan)
        return "n/a" if np.isnan(x) else fmt_float(x, 6)

    print(f"{'Mine':<10} {f(d_mine,'h'):>12} {f(d_mine,'Omega_m'):>14} {f(d_mine,'z_eq'):>14} {f(d_mine,'Neff'):>14}")
    print(f"{'Tobias':<10} {f(d_tob,'h'):>12} {f(d_tob,'Omega_m'):>14} {f(d_tob,'z_eq'):>14} {f(d_tob,'Neff'):>14}")

def print_pk_table(scan_name, scan_value, h_ref, pk_mine, pk_tob):
    print(f"\n[{scan_name} = {scan_value}] P(k) comparison table")
    print_separator(110)
    print(
        f"{'k [1/Mpc]':>11} "
        f"{'k [h/Mpc]':>11} "
        f"{'P_Mine':>16} "
        f"{'P_Tobias':>16} "
        f"{'Mine/Tob-1 [%]':>18}"
    )
    print_separator(110)

    worst_idx = 0
    worst_abs = -1.0
    best_idx = 0
    best_abs = np.inf

    for i, k in enumerate(K_PRINT_1MPC):
        k_h = k / h_ref
        diff = pct_diff(pk_mine[i], pk_tob[i])

        if abs(diff) > worst_abs:
            worst_abs = abs(diff)
            worst_idx = i
        if abs(diff) < best_abs:
            best_abs = abs(diff)
            best_idx = i

        print(
            f"{fmt_sci(k, 2):>11} "
            f"{fmt_float(k_h, 6):>11} "
            f"{fmt_sci(pk_mine[i], 6):>16} "
            f"{fmt_sci(pk_tob[i], 6):>16} "
            f"{fmt_float(diff, 6):>18}"
        )

    print_separator(110)
    print(
        f"Worst selected-point mismatch at k = {K_PRINT_1MPC[worst_idx]:.3e} 1/Mpc "
        f"({K_PRINT_1MPC[worst_idx]/h_ref:.6f} h/Mpc): "
        f"{pct_diff(pk_mine[worst_idx], pk_tob[worst_idx]):.6f}%"
    )
    print(
        f"Best selected-point mismatch  at k = {K_PRINT_1MPC[best_idx]:.3e} 1/Mpc "
        f"({K_PRINT_1MPC[best_idx]/h_ref:.6f} h/Mpc): "
        f"{pct_diff(pk_mine[best_idx], pk_tob[best_idx]):.6f}%"
    )


# =============================================================================
# CSV containers
# =============================================================================
point_rows = []
summary_rows = []


# =============================================================================
# Shared summary-row builder
# =============================================================================
def append_summary_row(scan_name, scan_value, h_ref, diffs, d_mine, d_tob):
    worst_idx = int(np.argmax(np.abs(diffs)))
    best_idx  = int(np.argmin(np.abs(diffs)))

    summary_rows.append({
        "scan_name": scan_name,
        "scan_value": scan_value,

        "worst_selected_k_1Mpc": float(K_PRINT_1MPC[worst_idx]),
        "worst_selected_k_hMpc": float(K_PRINT_1MPC[worst_idx] / h_ref),
        "worst_mine_over_tobias_minus1_pct": float(diffs[worst_idx]),
        "worst_abs_mine_over_tobias_minus1_pct": float(np.abs(diffs[worst_idx])),

        "best_selected_k_1Mpc": float(K_PRINT_1MPC[best_idx]),
        "best_selected_k_hMpc": float(K_PRINT_1MPC[best_idx] / h_ref),
        "best_mine_over_tobias_minus1_pct": float(diffs[best_idx]),
        "best_abs_mine_over_tobias_minus1_pct": float(np.abs(diffs[best_idx])),

        "mine_h": d_mine.get("h", np.nan),
        "tobias_h": d_tob.get("h", np.nan),
        "mine_Omega_m": d_mine.get("Omega_m", np.nan),
        "tobias_Omega_m": d_tob.get("Omega_m", np.nan),
        "mine_z_eq": d_mine.get("z_eq", np.nan),
        "tobias_z_eq": d_tob.get("z_eq", np.nan),
        "mine_Neff": d_mine.get("Neff", np.nan),
        "tobias_Neff": d_tob.get("Neff", np.nan),
    })


# =============================================================================
# Scan runners
# =============================================================================
def run_deg_scan():
    print_scan_header(
        "Scan 1: deg_ncdm_interacting",
        (
            f"Fixed parameters: f_idm_drmd={F_IDM_BASE}, "
            f"delta_Neff_drmd={DELTA_NEFF_ON}, z_stop={Z_STOP_ON:.1e}, "
            f"G_over_aH_drmd_ini={G_AH_ON:.1e}, "
            f"m_ncdm_interacting={M_NCDM_EV}, G_eff_ncdm_interacting={G_EFF_NCDM_ON}"
        )
    )

    compact_rows = []

    for deg_val in DEG_SCAN_VALUES:
        print("\n" + "#" * 84)
        print(f"Running deg_ncdm_interacting = {deg_val}")
        print("#" * 84)

        p_mine = build_mine_params(deg_val, F_IDM_BASE)
        p_tob  = build_tobias_params(deg_val, F_IDM_BASE)

        c_mine = run_model(ClassMine,   p_mine, f"Mine   deg={deg_val}")
        c_tob  = run_model(ClassTobias, p_tob,  f"Tobias deg={deg_val}")

        d_mine = get_derived_safe(c_mine)
        d_tob  = get_derived_safe(c_tob)
        print_point_summary("deg_ncdm_interacting", deg_val, d_mine, d_tob)

        h_ref = float(c_tob.h())
        pk_mine = pk_safe(c_mine, K_PRINT_1MPC, Z_PK) * h_ref**3
        pk_tob  = pk_safe(c_tob,  K_PRINT_1MPC, Z_PK) * h_ref**3

        print_pk_table("deg_ncdm_interacting", deg_val, h_ref, pk_mine, pk_tob)

        diffs = 100.0 * (pk_mine / pk_tob - 1.0)
        worst_pct = float(np.max(np.abs(diffs)))
        compact_rows.append((deg_val, worst_pct))

        for i, k in enumerate(K_PRINT_1MPC):
            point_rows.append({
                "scan_name": "deg_ncdm_interacting",
                "scan_value": deg_val,
                "k_1Mpc": float(k),
                "k_hMpc": float(k / h_ref),
                "P_mine": float(pk_mine[i]),
                "P_tobias": float(pk_tob[i]),
                "mine_over_tobias_minus1_pct": float(diffs[i]),
                "abs_mine_over_tobias_minus1_pct": float(np.abs(diffs[i])),
                "mine_h": d_mine.get("h", np.nan),
                "tobias_h": d_tob.get("h", np.nan),
                "mine_Omega_m": d_mine.get("Omega_m", np.nan),
                "tobias_Omega_m": d_tob.get("Omega_m", np.nan),
                "mine_z_eq": d_mine.get("z_eq", np.nan),
                "tobias_z_eq": d_tob.get("z_eq", np.nan),
                "mine_Neff": d_mine.get("Neff", np.nan),
                "tobias_Neff": d_tob.get("Neff", np.nan),
            })

        append_summary_row("deg_ncdm_interacting", deg_val, h_ref, diffs, d_mine, d_tob)

        cleanup(c_mine, c_tob)

    print("\n" + "=" * 84)
    print("Compact summary: deg_ncdm_interacting scan")
    print("=" * 84)
    print(f"{'deg_ncdm_interacting':>24} {'worst |Mine/Tobias-1| [%]':>32}")
    print_separator(84)
    for deg_val, worst_pct in compact_rows:
        print(f"{fmt_float(deg_val, 3):>24} {fmt_float(worst_pct, 6):>32}")


def run_fidm_scan():
    print_scan_header(
        "Scan 2: f_idm_drmd",
        (
            f"Fixed parameters: deg_ncdm_interacting={DEG_NCDM_BASE}, "
            f"delta_Neff_drmd={DELTA_NEFF_ON}, z_stop={Z_STOP_ON:.1e}, "
            f"G_over_aH_drmd_ini={G_AH_ON:.1e}, "
            f"m_ncdm_interacting={M_NCDM_EV}, G_eff_ncdm_interacting={G_EFF_NCDM_ON}"
        )
    )

    compact_rows = []

    for fidm_val in FIDM_SCAN_VALUES:
        print("\n" + "#" * 84)
        print(f"Running f_idm_drmd = {fidm_val}")
        print("#" * 84)

        p_mine = build_mine_params(DEG_NCDM_BASE, fidm_val)
        p_tob  = build_tobias_params(DEG_NCDM_BASE, fidm_val)

        c_mine = run_model(ClassMine,   p_mine, f"Mine   f_idm={fidm_val}")
        c_tob  = run_model(ClassTobias, p_tob,  f"Tobias f_idm={fidm_val}")

        d_mine = get_derived_safe(c_mine)
        d_tob  = get_derived_safe(c_tob)
        print_point_summary("f_idm_drmd", fidm_val, d_mine, d_tob)

        h_ref = float(c_tob.h())
        pk_mine = pk_safe(c_mine, K_PRINT_1MPC, Z_PK) * h_ref**3
        pk_tob  = pk_safe(c_tob,  K_PRINT_1MPC, Z_PK) * h_ref**3

        print_pk_table("f_idm_drmd", fidm_val, h_ref, pk_mine, pk_tob)

        diffs = 100.0 * (pk_mine / pk_tob - 1.0)
        worst_pct = float(np.max(np.abs(diffs)))
        compact_rows.append((fidm_val, worst_pct))

        for i, k in enumerate(K_PRINT_1MPC):
            point_rows.append({
                "scan_name": "f_idm_drmd",
                "scan_value": fidm_val,
                "k_1Mpc": float(k),
                "k_hMpc": float(k / h_ref),
                "P_mine": float(pk_mine[i]),
                "P_tobias": float(pk_tob[i]),
                "mine_over_tobias_minus1_pct": float(diffs[i]),
                "abs_mine_over_tobias_minus1_pct": float(np.abs(diffs[i])),
                "mine_h": d_mine.get("h", np.nan),
                "tobias_h": d_tob.get("h", np.nan),
                "mine_Omega_m": d_mine.get("Omega_m", np.nan),
                "tobias_Omega_m": d_tob.get("Omega_m", np.nan),
                "mine_z_eq": d_mine.get("z_eq", np.nan),
                "tobias_z_eq": d_tob.get("z_eq", np.nan),
                "mine_Neff": d_mine.get("Neff", np.nan),
                "tobias_Neff": d_tob.get("Neff", np.nan),
            })

        append_summary_row("f_idm_drmd", fidm_val, h_ref, diffs, d_mine, d_tob)

        cleanup(c_mine, c_tob)

    print("\n" + "=" * 84)
    print("Compact summary: f_idm_drmd scan")
    print("=" * 84)
    print(f"{'f_idm_drmd':>24} {'worst |Mine/Tobias-1| [%]':>32}")
    print_separator(84)
    for fidm_val, worst_pct in compact_rows:
        print(f"{fmt_float(fidm_val, 3):>24} {fmt_float(worst_pct, 6):>32}")


# =============================================================================
# CSV export
# =============================================================================
def write_csvs():
    point_file = os.path.join(CSV_OUTPUT_DIR, "parameter_space_scan_points.csv")
    summary_file = os.path.join(CSV_OUTPUT_DIR, "parameter_space_scan_summary.csv")

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    if point_rows:
        with open(point_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
            writer.writeheader()
            writer.writerows(point_rows)

    if summary_rows:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\nCSV files written:")
    print(f"  {point_file}")
    print(f"  {summary_file}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\nStarting parameter-space validation against Tobias.")
    run_deg_scan()
    run_fidm_scan()
    write_csvs()
    print("\nDone.")