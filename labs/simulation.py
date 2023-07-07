# %%
import numpy as np
import time
from rydberg.run_simulation import run_simulation

from numba import config
from rydberg.assets import disable_jit
config.DISABLE_JIT = disable_jit


start = time.time()
beta = 10
Es_Eerrs, Ns_Nerrs, Ms_Merrs = run_simulation(4, 4, betas=np.array([beta,]))
total_time = time.time() - start

print(f"Execution time = {round(total_time, 2)} seconds.")
print("Energy per site ={E:.8f} with error = {Eerr:.8f} at T={T:.3f}".format(E=Es_Eerrs[0,0], Eerr= Es_Eerrs[0,1], T=1./beta))
print("Particle density ={N:.8f} with error = {Nerr:.8f} at T={T:.3f}".format(N=Ns_Nerrs[0,0], Nerr= Ns_Nerrs[0,1], T=1./beta))
print("Magnetization per site ={M:.8f} with error = {Merr:.8f} at T={T:.3f}".format(M=Ms_Merrs[0,0], Merr= Ms_Merrs[0,1], T=1./beta))
# %%