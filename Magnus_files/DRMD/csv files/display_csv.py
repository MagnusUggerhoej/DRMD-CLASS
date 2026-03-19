#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CSV = "/Users/magnusuggerhoj/Desktop/Speciale/CLASS_NEDE/DRMD-CLASS/Magnus_files/DRMD/csv files/parameter_space_scan_summary.csv"


def main():
    # Get CSV path from command line, else use default
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1]).expanduser()
    else:
        csv_path = Path(DEFAULT_CSV).expanduser()

    if not csv_path.exists():
        print(f"Error: file not found:\n{csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV:\n{e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Pretty printed table
    # -------------------------------------------------------------------------
    display_df = df[
        [
            "scan_name",
            "scan_value",

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
    ].copy()

    display_df = display_df.rename(
        columns={
            "scan_name": "scan",
            "scan_value": "value",

            "worst_selected_k_1Mpc": "worst_k [1/Mpc]",
            "worst_abs_mine_over_tobias_minus1_pct": "worst |Mine/Tobias-1| [%]",
            "worst_mine_over_tobias_minus1_pct": "worst signed [%]",

            "best_selected_k_1Mpc": "best_k [1/Mpc]",
            "best_abs_mine_over_tobias_minus1_pct": "best |Mine/Tobias-1| [%]",
            "best_mine_over_tobias_minus1_pct": "best signed [%]",

            "mine_Omega_m": "Omega_m",
            "mine_z_eq": "z_eq (Mine)",
            "mine_Neff": "Neff (Mine)",
            "tobias_Neff": "Neff (Tobias)",
        }
    )

    display_df = display_df.sort_values(by=["scan", "value"]).reset_index(drop=True)

    round_map = {
        "value": 3,

        "worst_k [1/Mpc]": 3,
        "worst |Mine/Tobias-1| [%]": 3,
        "worst signed [%]": 3,

        "best_k [1/Mpc]": 3,
        "best |Mine/Tobias-1| [%]": 3,
        "best signed [%]": 3,

        "Omega_m": 6,
        "z_eq (Mine)": 3,
        "Neff (Mine)": 6,
        "Neff (Tobias)": 6,
    }
    display_df = display_df.round(round_map)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print(f"\nLoaded: {csv_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print(display_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Plot summary
    # -------------------------------------------------------------------------
    deg_df = df[df["scan_name"] == "deg_ncdm_interacting"].copy().sort_values("scan_value")
    fidm_df = df[df["scan_name"] == "f_idm_drmd"].copy().sort_values("scan_value")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharey=False)

    # Top panel: deg_ncdm_interacting
    ax1.plot(
        deg_df["scan_value"],
        deg_df["worst_abs_mine_over_tobias_minus1_pct"],
        marker="o",
        label="Worst mismatch"
    )
    ax1.plot(
        deg_df["scan_value"],
        deg_df["best_abs_mine_over_tobias_minus1_pct"],
        marker="o",
        linestyle="--",
        label="Best mismatch"
    )
    ax1.set_title("Summary scan: deg_ncdm_interacting")
    ax1.set_xlabel("deg_ncdm_interacting")
    ax1.set_ylabel(r"$|P_{\rm Mine}/P_{\rm Tobias} - 1|$ [%]")
    ax1.grid(True, ls=":")
    ax1.legend()

    # Bottom panel: f_idm_drmd
    ax2.plot(
        fidm_df["scan_value"],
        fidm_df["worst_abs_mine_over_tobias_minus1_pct"],
        marker="o",
        label="Worst mismatch"
    )
    ax2.plot(
        fidm_df["scan_value"],
        fidm_df["best_abs_mine_over_tobias_minus1_pct"],
        marker="o",
        linestyle="--",
        label="Best mismatch"
    )
    ax2.set_title("Summary scan: f_idm_drmd")
    ax2.set_xlabel("f_idm_drmd")
    ax2.set_ylabel(r"$|P_{\rm Mine}/P_{\rm Tobias} - 1|$ [%]")
    ax2.grid(True, ls=":")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()