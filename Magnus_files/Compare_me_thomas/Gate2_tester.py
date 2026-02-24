#!/usr/bin/env python3
"""
Gate2 / Phase2: P(k) EFFECT test — Thomas vs Mine (ncdm interactions)

We run FOUR models:

  Thomas FS  : 1 massive species present, free-streaming (G_eff ~ 0)
  Thomas INT : 1 massive species present, interacting (G_eff = 1e-1)

  Mine FS    : 1 massive species present in STANDARD bucket (free-streaming)
  Mine INT   : 1 massive species present in INTERACTING bucket (G_eff = 1e-1)

Then compare effects:
  ratio_T = P_T(INT)/P_T(FS)
  ratio_M = P_M(INT)/P_M(FS)
  mismatch = ratio_M/ratio_T - 1
"""

import re
import numpy as np

from classy import Class as ClassThomas
from classy_NEDE import Class as ClassMine


# --------------------------
# Knobs
# --------------------------
LOG10_G_EFF_ZERO = -50.0   # effectively off
G_EFF_ON = 1e-1            # your suggested first "on"
M_NCDM_EV = 0.06
DEG_NCDM = 1.0             # choose 1.0 or 3.046 and keep consistent everywhere

Z_PK = 0.0
K_LIST = np.logspace(-4, 0, 140)   # 1/Mpc
K_REPORT = [1e-3, 1e-2, 1e-1, 1.0]

USE_LOG10_FOR_NCDM = False  # if True uses log10G_eff_ncdm_interacting instead of G_eff_ncdm_interacting


# --------------------------
# Helpers
# --------------------------
def rms(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x*x)))


def parse_unread_keys_from_error(errmsg: str):
    m = re.search(r"did not read input parameter\(s\):\s*(.*)", errmsg)
    if not m:
        return []
    tail = m.group(1).strip()
    return [k.strip() for k in tail.split(",") if k.strip()]


def run_with_autostrip(Cls, params_master, label, max_passes=8):
    """
    Strip unread keys until compute works.
    """
    params = dict(params_master)
    stripped = []

    for attempt in range(1, max_passes + 1):
        c = Cls()
        c.set(params)
        try:
            c.compute()
            print(f"\n=== {label}: compute OK after {attempt} pass(es) ===")
            if stripped:
                print(f"{label}: stripped unread keys: {', '.join(stripped)}")
            else:
                print(f"{label}: no keys stripped")
            return c, params, set(stripped)

        except Exception as e:
            msg = str(e)
            unread = parse_unread_keys_from_error(msg)
            if unread:
                print(f"\n=== {label}: pass {attempt} -> unread keys detected ===")
                print("Unread:", ", ".join(unread))
                for k in unread:
                    params.pop(k, None)
                stripped.extend(unread)
                continue

            print(f"\n=== {label}: compute FAILED (non-unread error) ===")
            raise

    raise RuntimeError(f"{label}: exceeded max_passes={max_passes} while stripping.")


def run_pk(cosmo, label):
    pk = np.array([cosmo.pk(k, Z_PK) for k in K_LIST], dtype=float)
    print(f"  {label}: pk(z={Z_PK}) min={pk.min():.6e} max={pk.max():.6e}")
    return pk


def report_at_k(name, arr):
    arr = np.asarray(arr)
    print(f"\n--- {name} at representative k ---")
    for k in K_REPORT:
        i = int(np.argmin(np.abs(K_LIST - k)))
        print(f"  k={K_LIST[i]:.3e} : {arr[i]:+.6e}")


# --------------------------
# Explicit base params
# --------------------------
def base_params():
    return {
        # outputs
        "output": "mPk",
        "z_pk": str(Z_PK),

        # LCDM
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "A_s": 2.100e-9,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "T_cmb": 2.7255,
        "Omega_k": 0.0,

        # keep explicit (some will be stripped by Thomas)
        "non_linear": "none",
        "lensing": "no",
        "modes": "s",
        "l_max_scalars": 2500,
        "bbn": "BBN",
        "recombination": "Recfast",
        "Omega_Lambda": 0.0,

        # pk coverage
        "P_k_max_1/Mpc": 5.0,

        # UR sector (keep fixed, interactions off here)
        "N_ur": max(0.0, 3.046 - float(DEG_NCDM)),
        "log10_G_eff_ur": LOG10_G_EFF_ZERO,
    }


# --------------------------
# Model dictionaries
# --------------------------
def thomas_fs():
    p = base_params()
    # Thomas uses interacting-ncdm interface; set G_eff ~ 0 for FS baseline
    p.update({
        "N_ncdm_interacting": 1,
        "m_ncdm_interacting": float(M_NCDM_EV),
        "deg_ncdm_interacting": float(DEG_NCDM),
    })
    if USE_LOG10_FOR_NCDM:
        p["log10G_eff_ncdm_interacting"] = LOG10_G_EFF_ZERO
    else:
        p["G_eff_ncdm_interacting"] = 0.0
    return p


def thomas_int():
    p = base_params()
    p.update({
        "N_ncdm_interacting": 1,
        "m_ncdm_interacting": float(M_NCDM_EV),
        "deg_ncdm_interacting": float(DEG_NCDM),
    })
    if USE_LOG10_FOR_NCDM:
        p["log10G_eff_ncdm_interacting"] = float(np.log10(G_EFF_ON))
    else:
        p["G_eff_ncdm_interacting"] = float(G_EFF_ON)
    return p


def mine_fs():
    p = base_params()
    # Mine split interface: FS means put the species in STANDARD bucket
    p.update({
        "N_ncdm_standard": 1,
        "N_ncdm_interacting": 0,
        "m_ncdm_standard": float(M_NCDM_EV),
        "deg_ncdm_standard": float(DEG_NCDM),
    })
    # IMPORTANT: do NOT set legacy N_ncdm anywhere
    p.pop("N_ncdm", None)
    return p


def mine_int():
    p = base_params()
    # Mine split interface: INT means put the species in INTERACTING bucket
    p.update({
        "N_ncdm_standard": 0,
        "N_ncdm_interacting": 1,
        "m_ncdm_interacting": float(M_NCDM_EV),
        "deg_ncdm_interacting": float(DEG_NCDM),
    })
    if USE_LOG10_FOR_NCDM:
        p["log10G_eff_ncdm_interacting"] = float(np.log10(G_EFF_ON))
    else:
        p["G_eff_ncdm_interacting"] = float(G_EFF_ON)

    p.pop("N_ncdm", None)
    return p


# --------------------------
# Main
# --------------------------
def main():
    print("\n#############################")
    print("# Gate2 / Phase2: P(k) EFFECT test — Thomas vs Mine (ncdm interacting)")
    print("#############################\n")

    pT_fs = thomas_fs()
    pT_int = thomas_int()
    pM_fs = mine_fs()
    pM_int = mine_int()

    cT_fs, pT_fs_final, strip_T_fs = run_with_autostrip(ClassThomas, pT_fs, "Thomas FS")
    pkT_fs = run_pk(cT_fs, "Thomas FS")

    cM_fs, pM_fs_final, strip_M_fs = run_with_autostrip(ClassMine, pM_fs, "Mine FS")
    pkM_fs = run_pk(cM_fs, "Mine FS")

    cT_int, pT_int_final, strip_T_int = run_with_autostrip(ClassThomas, pT_int, "Thomas INT")
    pkT_int = run_pk(cT_int, "Thomas INT")

    cM_int, pM_int_final, strip_M_int = run_with_autostrip(ClassMine, pM_int, "Mine INT")
    pkM_int = run_pk(cM_int, "Mine INT")

    ratio_T = pkT_int / pkT_fs
    ratio_M = pkM_int / pkM_fs
    mismatch = ratio_M / ratio_T - 1.0

    print("\n==============================")
    print("EFFECT COMPARISON SUMMARY")
    print("==============================")
    print(f"ratio_T = P_T(INT)/P_T(FS):   max|ratio-1|={np.max(np.abs(ratio_T-1)):.3e}   rms={rms(ratio_T-1):.3e}")
    print(f"ratio_M = P_M(INT)/P_M(FS):   max|ratio-1|={np.max(np.abs(ratio_M-1)):.3e}   rms={rms(ratio_M-1):.3e}")
    print(f"mismatch = ratio_M/ratio_T-1: max|.|={np.max(np.abs(mismatch)):.3e}        rms={rms(mismatch):.3e}")

    report_at_k("ratio_T - 1", ratio_T - 1)
    report_at_k("ratio_M - 1", ratio_M - 1)
    report_at_k("mismatch", mismatch)

    print("\n==============================")
    print("WHAT GOT STRIPPED")
    print("==============================")
    def fmt(s): return ", ".join(sorted(s)) if s else "(nothing)"
    print("Thomas FS :", fmt(strip_T_fs))
    print("Mine   FS :", fmt(strip_M_fs))
    print("Thomas INT:", fmt(strip_T_int))
    print("Mine   INT:", fmt(strip_M_int))

    for c in (cT_fs, cM_fs, cT_int, cM_int):
        try:
            c.struct_cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()