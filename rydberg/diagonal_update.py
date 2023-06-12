import numpy as np
from numba import njit
from rydberg.configuration import sample_from_distribution, vec_min, bond_to_operator, operator_to_bond


@njit
def diagonal_update(spins, op_string, Vi, Ci, Pij, Pc, beta):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    M = op_string.shape[0]
    n = np.sum(op_string != -1)
    norm = np.sum(Pij)
    #print("norm = ", norm)
    prob_ratio = norm * beta
    
    for p in range(M):
        op = op_string[p]
        if op == -1:
            index1 = sample_from_distribution(Pc)
            index2 = sample_from_distribution(Pij[index1])
            s0 = min(index1, index2)
            s1 = max(index1, index2)
            #print("sampled s0 = ", s0, " s1 = ", s1)
            prob = prob_ratio / (M - n)
            #print("inserting prob = ", prob)
            
            if np.random.rand() < prob:
                if s0 == s1:
                    op_string[p] = 2 * s0
                    n += 1
                else:
                    vec = vec_min(s0, s1, Lx, Ly)
                    V = Vi[vec]
                    C = Ci[vec]
                    db = V / 2
                    
                    if spins[s0] == -1 and spins[s1] == -1:
                        W_actual = C
                    elif spins[s0] == -1 and spins[s1] == 1:
                        W_actual = db + C
                    elif spins[s0] == 1 and spins[s1] == -1:
                        W_actual = db + C
                    elif spins[s0] == 1 and spins[s1] == 1:
                        W_actual = - V + 2 * db + C
                    W_sampled = Pij[s0][s1]
                    #print("sampled W = ", W_sampled, " actual W = ", W_actual, "for spins = ", spins[s0], " ", spins[s1])
                    ratio = W_actual / W_sampled
                    if np.random.rand() < ratio:
                        op_string[p] = bond_to_operator(s0, s1, Lx, Ly)
                        n += 1
        elif np.mod(op, 2) == 0 and op < 2 * n_sites or op >= 2 * n_sites:
            prob = (M - n + 1) / prob_ratio
            #print("removing prob = ", prob)
            if np.random.rand() < prob:
                op_string[p] = -1
                n -= 1
        elif np.mod(op, 2) != 0 and op < 2 * n_sites:
            bond = operator_to_bond(op, Lx, Ly)
            if bond[0] != bond[1]:
                raise ValueError("off diag operator not on site!")
            site = int(bond[0])
            spins[site] = - spins[site]
    return n