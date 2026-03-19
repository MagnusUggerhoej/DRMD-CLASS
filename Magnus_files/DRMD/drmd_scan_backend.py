#!/usr/bin/env python3

# drmd_scan_backend.py
# Reusable backend for DRMD + interacting-ncdm parameter-space scans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classy_NEDE import Class as ClassMine
from classy_tobias import Class as ClassTobias


# =============================================================================
# Shared cosmology defaults
# =============================================================================
H0_DEFAULT        = 67.32
OMEGA_B_DEFAULT   = 0.02238
OMEGA_CDM_DEFAULT = 0.1201
A_S_DEFAULT       = 2.101e-9
N_S_DEFAULT       = 0.9660
TAU_REIO_DEFAULT  = 0.0543
K_PIVOT_DEFAULT   = 0.05
T_CMB_DEFAULT     = 2.7255
OMEGA_K_DEFAULT   = 0.0

L_MAX_DEFAULT = 2500
Z_PK_DEFAULT  = 0.0

K_GRID_1MPC_DEFAULT = np.logspace(-4, 1.0, 250)


# =============================================================================
# Helpers
# =============================================================================
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


# =============================================================================
# Base parameter builder
# =============================================================================
def base_params(
    H0=H0_DEFAULT,
    omega_b=OMEGA_B_DEFAULT,
    omega_cdm=OMEGA_CDM_DEFAULT,
    A_s=A_S_DEFAULT,
    n_s=N_S_DEFAULT,
    tau_reio=TAU_REIO_DEFAULT,
    k_pivot=K_PIVOT_DEFAULT,
    T_cmb=T_CMB_DEFAULT,
    Omega_k=OMEGA_K_DEFAULT,
    l_max=L_MAX_DEFAULT,
    k_grid_1Mpc=K_GRID_1MPC_DEFAULT,
):
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
        "l_max_scalars": int(l_max),
        "P_k_max_1/Mpc": float(np.max(k_grid_1Mpc) * 1.05),

        "recombination": "recfast",
        "reio_parametrization": "reio_camb",

        # keep quiet by default for notebook use
        "input_verbose": 0,
        "background_verbose": 0,
        "thermodynamics_verbose": 0,
        "perturbations_verbose": 0,
    }


# =============================================================================
# Benchmark parameter builder
# =============================================================================
def get_benchmark_params():
    """
    Returns the benchmark model as a plain dict.
    This dict represents the physics model, not code-specific formatting.
    """
    return {
        # shared cosmology
        "H0": H0_DEFAULT,
        "omega_b": OMEGA_B_DEFAULT,
        "omega_cdm": OMEGA_CDM_DEFAULT,
        "A_s": A_S_DEFAULT,
        "n_s": N_S_DEFAULT,
        "tau_reio": TAU_REIO_DEFAULT,
        "k_pivot": K_PIVOT_DEFAULT,
        "T_cmb": T_CMB_DEFAULT,
        "Omega_k": OMEGA_K_DEFAULT,

        # interacting ncdm benchmark
        "deg_ncdm_interacting": 1.0,
        "m_ncdm_interacting": 0.06,
        "T_ncdm_interacting": 0.71611,
        "G_eff_ncdm_interacting": 1.0e-1,

        # DRMD benchmark
        "delta_Neff_drmd": 0.8,
        "f_idm_drmd": 0.03,
        "z_stop": 5.0e4,
        "G_over_aH_drmd_ini": 1.0e9,
    }


def with_parameter(benchmark, parameter_name, parameter_value):
    """
    Return a copy of the benchmark dict with one parameter changed.
    """
    out = benchmark.copy()
    out[parameter_name] = parameter_value
    return out


# =============================================================================
# Code-specific parameter builders
# =============================================================================
def build_mine_params(
    model_params,
    l_max=L_MAX_DEFAULT,
    k_grid_1Mpc=K_GRID_1MPC_DEFAULT,
):
    """
    Build Mine-format input parameters from the generic benchmark/scan dict.
    """
    p = base_params(
        H0=model_params["H0"],
        omega_b=model_params["omega_b"],
        omega_cdm=model_params["omega_cdm"],
        A_s=model_params["A_s"],
        n_s=model_params["n_s"],
        tau_reio=model_params["tau_reio"],
        k_pivot=model_params["k_pivot"],
        T_cmb=model_params["T_cmb"],
        Omega_k=model_params["Omega_k"],
        l_max=l_max,
        k_grid_1Mpc=k_grid_1Mpc,
    )

    deg = float(model_params["deg_ncdm_interacting"])
    n_ur_run = max(0.0, 3.046 - deg)

    p.update({
        "N_ur": float(n_ur_run),

        "N_ncdm_standard": 0,
        "N_ncdm_interacting": 1,

        # In your fork, interacting ncdm mass/deg are passed through legacy arrays
        "m_ncdm": float(model_params["m_ncdm_interacting"]),
        "deg_ncdm": float(model_params["deg_ncdm_interacting"]),
        "T_ncdm": float(model_params["T_ncdm_interacting"]),
        "G_eff_ncdm_interacting": float(model_params["G_eff_ncdm_interacting"]),

        # DRMD
        "delta_Neff_drmd": float(model_params["delta_Neff_drmd"]),
        "f_idm_drmd": float(model_params["f_idm_drmd"]),
        "z_stop": float(model_params["z_stop"]),
        "G_over_aH_drmd_ini": float(model_params["G_over_aH_drmd_ini"]),
    })

    p.pop("N_ncdm", None)
    return p


def build_tobias_params(
    model_params,
    l_max=L_MAX_DEFAULT,
    k_grid_1Mpc=K_GRID_1MPC_DEFAULT,
):
    """
    Build Tobias-format input parameters from the generic benchmark/scan dict.
    """
    p = base_params(
        H0=model_params["H0"],
        omega_b=model_params["omega_b"],
        omega_cdm=model_params["omega_cdm"],
        A_s=model_params["A_s"],
        n_s=model_params["n_s"],
        tau_reio=model_params["tau_reio"],
        k_pivot=model_params["k_pivot"],
        T_cmb=model_params["T_cmb"],
        Omega_k=model_params["Omega_k"],
        l_max=l_max,
        k_grid_1Mpc=k_grid_1Mpc,
    )

    deg = float(model_params["deg_ncdm_interacting"])
    n_ur_run = max(0.0, 3.046 - deg)

    p.update({
        "N_ur": float(n_ur_run),

        "N_ncdm_interacting": 1,
        "m_ncdm_interacting": float(model_params["m_ncdm_interacting"]),
        "deg_ncdm_interacting": float(model_params["deg_ncdm_interacting"]),
        "T_ncdm_interacting": float(model_params["T_ncdm_interacting"]),
        "G_eff_ncdm_interacting": float(model_params["G_eff_ncdm_interacting"]),

        # DRMD
        "delta_Neff_drmd": float(model_params["delta_Neff_drmd"]),
        "f_idm_drmd": float(model_params["f_idm_drmd"]),
        "z_stop": float(model_params["z_stop"]),
        "G_over_aH_drmd_ini": float(model_params["G_over_aH_drmd_ini"]),
    })

    for key in ["N_ncdm", "m_ncdm", "deg_ncdm", "T_ncdm"]:
        p.pop(key, None)

    return p


# =============================================================================
# Single run comparison
# =============================================================================
def run_single_model_point(
    model_params,
    k_values_1Mpc,
    k_grid_1Mpc=K_GRID_1MPC_DEFAULT,
    z_pk=Z_PK_DEFAULT,
    l_max=L_MAX_DEFAULT,
    verbose=True,
):
    """
    Run Mine and Tobias at one model point and return:
      - point-by-point selected-k DataFrame
      - one-row summary dict
    """
    p_mine = build_mine_params(model_params, l_max=l_max, k_grid_1Mpc=k_grid_1Mpc)
    p_tob  = build_tobias_params(model_params, l_max=l_max, k_grid_1Mpc=k_grid_1Mpc)

    c_mine = run_model(ClassMine, p_mine, "Mine") if verbose else _run_silent(ClassMine, p_mine)
    c_tob  = run_model(ClassTobias, p_tob, "Tobias") if verbose else _run_silent(ClassTobias, p_tob)

    try:
        d_mine = get_derived_safe(c_mine)
        d_tob  = get_derived_safe(c_tob)

        h_ref = float(c_tob.h())

        k_values_1Mpc = np.array(k_values_1Mpc, dtype=float)
        pk_mine = pk_safe(c_mine, k_values_1Mpc, z_pk) * h_ref**3
        pk_tob  = pk_safe(c_tob,  k_values_1Mpc, z_pk) * h_ref**3

        diffs_pct = pct_diff(pk_mine, pk_tob)

        worst_idx = int(np.argmax(np.abs(diffs_pct)))
        best_idx  = int(np.argmin(np.abs(diffs_pct)))

        point_rows = []
        for i, k in enumerate(k_values_1Mpc):
            point_rows.append({
                "k_1Mpc": float(k),
                "k_hMpc": float(k / h_ref),
                "P_mine": float(pk_mine[i]),
                "P_tobias": float(pk_tob[i]),
                "mine_over_tobias_minus1_pct": float(diffs_pct[i]),
                "abs_mine_over_tobias_minus1_pct": float(np.abs(diffs_pct[i])),
            })

        point_df = pd.DataFrame(point_rows)

        summary = {
            "worst_selected_k_1Mpc": float(k_values_1Mpc[worst_idx]),
            "worst_selected_k_hMpc": float(k_values_1Mpc[worst_idx] / h_ref),
            "worst_mine_over_tobias_minus1_pct": float(diffs_pct[worst_idx]),
            "worst_abs_mine_over_tobias_minus1_pct": float(np.abs(diffs_pct[worst_idx])),

            "best_selected_k_1Mpc": float(k_values_1Mpc[best_idx]),
            "best_selected_k_hMpc": float(k_values_1Mpc[best_idx] / h_ref),
            "best_mine_over_tobias_minus1_pct": float(diffs_pct[best_idx]),
            "best_abs_mine_over_tobias_minus1_pct": float(np.abs(diffs_pct[best_idx])),

            "mine_h": d_mine.get("h", np.nan),
            "tobias_h": d_tob.get("h", np.nan),
            "mine_Omega_m": d_mine.get("Omega_m", np.nan),
            "tobias_Omega_m": d_tob.get("Omega_m", np.nan),
            "mine_Omega_r": d_mine.get("Omega_r", np.nan),
            "tobias_Omega_r": d_tob.get("Omega_r", np.nan),
            "mine_z_eq": d_mine.get("z_eq", np.nan),
            "tobias_z_eq": d_tob.get("z_eq", np.nan),
            "mine_Neff": d_mine.get("Neff", np.nan),
            "tobias_Neff": d_tob.get("Neff", np.nan),
        }

        return point_df, summary

    finally:
        cleanup(c_mine, c_tob)


def _run_silent(ClassObj, params):
    c = ClassObj()
    c.set(params)
    c.compute()
    return c


# =============================================================================
# Parameter scan
# =============================================================================
def run_parameter_scan(
    parameter_name,
    scan_values,
    benchmark=None,
    k_values_1Mpc=None,
    k_grid_1Mpc=K_GRID_1MPC_DEFAULT,
    z_pk=Z_PK_DEFAULT,
    l_max=L_MAX_DEFAULT,
    verbose=False,
):
    """
    Run a one-parameter scan and return a summary DataFrame.

    Each row corresponds to one scan point.
    """
    if benchmark is None:
        benchmark = get_benchmark_params()

    if k_values_1Mpc is None:
        k_values_1Mpc = [1e-2, 5e-2, 2e-1, 1.0]

    rows = []

    for value in scan_values:
        model_params = with_parameter(benchmark, parameter_name, value)

        if verbose:
            print("\n" + "=" * 80)
            print(f"Running scan: {parameter_name} = {value}")
            print("=" * 80)

        point_df, summary = run_single_model_point(
            model_params=model_params,
            k_values_1Mpc=k_values_1Mpc,
            k_grid_1Mpc=k_grid_1Mpc,
            z_pk=z_pk,
            l_max=l_max,
            verbose=verbose,
        )

        row = {
            "scan_family": _infer_scan_family(parameter_name),
            "parameter_name": parameter_name,
            "parameter_value": value,
        }

        # store benchmark values too
        for key, val in benchmark.items():
            row[f"benchmark_{key}"] = val

        # store actual model-point values
        for key, val in model_params.items():
            row[f"model_{key}"] = val

        # store summary metrics
        row.update(summary)

        rows.append(row)

    return pd.DataFrame(rows)


def _infer_scan_family(parameter_name):
    drmd_params = {
        "delta_Neff_drmd",
        "f_idm_drmd",
        "z_stop",
        "G_over_aH_drmd_ini",
    }
    ncdm_params = {
        "deg_ncdm_interacting",
        "G_eff_ncdm_interacting",
        "m_ncdm_interacting",
    }

    if parameter_name in drmd_params:
        return "DRMD"
    if parameter_name in ncdm_params:
        return "interacting_ncdm"
    return "other"


# =============================================================================
# Detailed scan output (optional)
# =============================================================================
def run_parameter_scan_detailed(
    parameter_name,
    scan_values,
    benchmark=None,
    k_values_1Mpc=None,
    k_grid_1Mpc=K_GRID_1MPC_DEFAULT,
    z_pk=Z_PK_DEFAULT,
    l_max=L_MAX_DEFAULT,
    verbose=False,
):
    """
    Returns:
      summary_df, points_df

    - summary_df: one row per scan point
    - points_df: one row per scan point per selected k value
    """
    if benchmark is None:
        benchmark = get_benchmark_params()

    if k_values_1Mpc is None:
        k_values_1Mpc = [1e-2, 5e-2, 2e-1, 1.0]

    summary_rows = []
    point_rows = []

    for value in scan_values:
        model_params = with_parameter(benchmark, parameter_name, value)

        if verbose:
            print("\n" + "=" * 80)
            print(f"Running scan: {parameter_name} = {value}")
            print("=" * 80)

        point_df, summary = run_single_model_point(
            model_params=model_params,
            k_values_1Mpc=k_values_1Mpc,
            k_grid_1Mpc=k_grid_1Mpc,
            z_pk=z_pk,
            l_max=l_max,
            verbose=verbose,
        )

        summary_row = {
            "scan_family": _infer_scan_family(parameter_name),
            "parameter_name": parameter_name,
            "parameter_value": value,
        }
        for key, val in benchmark.items():
            summary_row[f"benchmark_{key}"] = val
        for key, val in model_params.items():
            summary_row[f"model_{key}"] = val
        summary_row.update(summary)
        summary_rows.append(summary_row)

        point_df = point_df.copy()
        point_df["scan_family"] = _infer_scan_family(parameter_name)
        point_df["parameter_name"] = parameter_name
        point_df["parameter_value"] = value
        point_rows.append(point_df)

    summary_df = pd.DataFrame(summary_rows)
    points_df = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()

    return summary_df, points_df


# =============================================================================
# Plotting
# =============================================================================
def plot_scan_summary(df, parameter_name=None):
    """
    Plot worst and best absolute mismatch vs scanned parameter value.
    """
    if parameter_name is not None:
        df = df[df["parameter_name"] == parameter_name].copy()

    if df.empty:
        print("No rows to plot.")
        return

    df = df.sort_values("parameter_value")

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["parameter_value"],
        df["worst_abs_mine_over_tobias_minus1_pct"],
        marker="o",
        label="Worst mismatch"
    )
    plt.plot(
        df["parameter_value"],
        df["best_abs_mine_over_tobias_minus1_pct"],
        marker="o",
        linestyle="--",
        label="Best mismatch"
    )

    title_param = parameter_name if parameter_name is not None else "scan"
    plt.title(f"Summary mismatch scan: {title_param}")
    plt.xlabel(title_param)
    plt.ylabel(r"$|P_{\rm Mine}/P_{\rm Tobias}-1|$ [%]")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


def display_scan_table(df, parameter_name=None, round_digits=4):
    """
    Return a compact display table for notebook viewing.
    """
    if parameter_name is not None:
        df = df[df["parameter_name"] == parameter_name].copy()

    cols = [
        "scan_family",
        "parameter_name",
        "parameter_value",
        "worst_selected_k_1Mpc",
        "worst_abs_mine_over_tobias_minus1_pct",
        "worst_mine_over_tobias_minus1_pct",
        "best_selected_k_1Mpc",
        "best_abs_mine_over_tobias_minus1_pct",
        "best_mine_over_tobias_minus1_pct",
        "mine_Omega_m",
        "mine_z_eq",
        "mine_Neff",
        "tobias_Neff",
    ]

    out = df[cols].copy()
    return out.round(round_digits)