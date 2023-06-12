# %%
import numpy as np
from rydberg.run_simulation import run_simulation


# %%
Es_Eerrs, Ns_Nerrs, Ms_Merrs = run_simulation(12, 12, betas=np.array([10,]))
# %%