import os
os.chdir("/Users/magnusuggerhoj/Desktop/Speciale/CLASSpp_Tobias")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import classy_SIN as c

m = c.Class()
m.set({
    "output": "mPk",
    "H0": 67.5,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau_reio": 0.054,
    "ln10^{10}A_s": 3.0,
    "n_s": 0.965,
    "N_ur": 2.0328,
    "z_max_pk": 4.0,
    "P_k_max_1/Mpc": 10.0,
    "f_idm_drmd": 0.0,
})
m.compute()
print("OK", m.h())
