# %%
import numpy as np
from rydberg.run_simulation import run_simulation


# %%
Es_Eerrs, Ns_Nerrs, Ms_Merrs = run_simulation(14, 14, betas=np.array([10,]))
# %%
print(Es_Eerrs, Ns_Nerrs, Ms_Merrs)

# %%
beta = 10.0
print("Energy per site ={E:.8f} with error = {Eerr:.8f} at T={T:.3f}".format(E=Es_Eerrs[0,0], Eerr= Es_Eerrs[0,1], T=1./beta))
print("Particle density ={N:.8f} with error = {Nerr:.8f} at T={T:.3f}".format(N=Ns_Nerrs[0,0], Nerr= Ns_Nerrs[0,1], T=1./beta))
print("Magnetization per site ={M:.8f} with error = {Merr:.8f} at T={T:.3f}".format(M=Ms_Merrs[0,0], Merr= Ms_Merrs[0,1], T=1./beta))
# %%