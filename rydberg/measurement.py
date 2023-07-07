from numba import njit
import numpy as np
from rydberg.assets import njit_kwargs
from rydberg.diagonal_update import diagonal_update
from rydberg.nondiagonal_update import cluster_update
from rydberg.configuration import site
from numba import config
from rydberg.assets import disable_jit
config.DISABLE_JIT = disable_jit

@njit(**njit_kwargs)
def resize(a, new_size):
    new = np.zeros(new_size, a.dtype)
    new[:a.size] = a
    return new

@njit(**njit_kwargs)
def thermalize(spins, op_string, V_i, C_i, Pij, Pc, beta, n_updates_warmup):
    for _ in range(n_updates_warmup):
        n = diagonal_update(spins, op_string, V_i, C_i, Pij, Pc, beta)
        cluster_update(spins, op_string, V_i, C_i)
        print("n = ", n)
        M_old = len(op_string)
        M_new = n + n // 3
        if M_new > M_old:
            op_string = resize(op_string, M_new)
            op_string[M_old: ] = -1
        print("resized to ", op_string.shape[0])
    return op_string

@njit(**njit_kwargs)
def get_staggering(Lx, Ly):
    stag = np.zeros(Lx*Ly, np.intp) 
    for x in range(Lx):
        for y in range(Ly):
            s = site(x, y, Lx, Ly) 
            stag[s] = (-1)**(x+y)
    return stag

@njit(**njit_kwargs)
def staggered_magnetization(spins, stag): 
    return np.sum(spins * stag * 0.5)

@njit(**njit_kwargs)
def measure(spins, op_string, V_i, C_i, Pij, Pc, stag, beta, n_updates_measure):
    ns = np.zeros(n_updates_measure)
    nums = np.zeros(n_updates_measure)
    ms = np.zeros(n_updates_measure)
    for idx in range(n_updates_measure):
        ns[idx] = diagonal_update(spins, op_string, V_i, C_i, Pij, Pc, beta) 
        print("n = ", ns[idx])
        #ms[idx] = np.abs(staggered_magnetization(spins, stag))
        # # print("absolute magnetization = ", ms[idx])
        cluster_update(spins, op_string, V_i, C_i)
        ms[idx] = np.abs(staggered_magnetization(spins, stag))
        nums[idx] = np.sum(spins == 1)
        #num = spins.count(1)
        print("density = ", nums[idx])
        print("absolute magnetization = ", ms[idx])
    return ns, nums, ms
