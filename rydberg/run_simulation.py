import numpy as np
from numba import njit
from rydberg.measurement import thermalize, measure, get_staggering
from rydberg.configuration import V_i, C_i, cumulative, init_prob_2d, init_SSE_square

@njit
def run_simulation(Lx, Ly, betas, n_updates_measure=10000, n_bins=10):
    spins, op_string = init_SSE_square(Lx, Ly)
    stag = get_staggering(Lx, Ly)
    n_sites = len(spins)
    Vi = V_i(n_sites)
    Ci = C_i(n_sites)
    Pij = init_prob_2d(n_sites)
    Pc = cumulative(n_sites, Pij)
    n_betas = betas.shape[0]
    Es_Eerrs = np.zeros((n_betas, 2))
    Ns_Nerrs = np.zeros((n_betas, 2))
    Ms_Merrs = np.zeros((n_betas, 2))
    i_beta = 0
    for beta in betas:
        # print("beta = {beta:.3f}".format(beta=beta), flush=True)
        print("beta = ", beta)
        op_string = thermalize(spins, op_string, Vi, Ci, Pij, Pc, beta, n_updates_measure//10)#n_updates_measure//10
        Es = np.zeros(n_bins)
        Ns = np.zeros(n_bins)
        Ms = np.zeros(n_bins)
        for n_bin in range(n_bins):
            ns, nums, ms = measure(spins, op_string, Vi, Ci, Pij, Pc, stag, beta, n_updates_measure)
            n_mean = np.mean(ns)
            E = (-n_mean/beta) / n_sites
            num_mean = np.mean(nums)
            N = num_mean / n_sites
            #ms_mean = np.mean(np.abs(ms))
            ms_mean = np.mean(ms)
            M = ms_mean / n_sites
            Es[n_bin] = E
            Ns[n_bin] = N
            Ms[n_bin] = M
        Es_Eerrs[i_beta][0] = np.mean(Es)
        Es_Eerrs[i_beta][1] = np.std(Es)/np.sqrt(n_bins)
        Ns_Nerrs[i_beta][0] = np.mean(Ns)
        Ns_Nerrs[i_beta][1] = np.std(Ns)/np.sqrt(n_bins)
        Ms_Merrs[i_beta][0] = np.mean(Ms)
        Ms_Merrs[i_beta][1] = np.std(Ms)/np.sqrt(n_bins)
        i_beta = i_beta + 1
    return Es_Eerrs, Ns_Nerrs, Ms_Merrs