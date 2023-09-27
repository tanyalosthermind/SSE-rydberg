import numpy as np
from numba import jit, njit
import math
import time
# import random
# from collections import Countera = 1.0

a = 1.0
Rb = 1.0
d = 1.1 #2.32572039876754
Omega = 1.0
cutoff = 12 #1.9
eps = 0.5
#d = 2*(Rb)**6

@jit(nopython=True)
def site(x, y, Lx, Ly):
    # TODO: generalize to the rectangular case
    return y * Lx + x

@jit(nopython=True)
def site_to_xy(site, Lx, Ly):
    # TODO: generalize to the rectangular case
    xy = np.zeros(2)
    xy[0] = site%Lx
    xy[1] = site//Lx
    return xy

@jit(nopython=True)
def bond_to_operator(s0, s1, Lx, Ly):
    n_sites = Lx * Ly
    k = 2 * n_sites - 1
    if s0 != s1:
        # TODO: needs to be optimized, too.
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    continue
                k += 1
                if i == s0 and j == s1:
                    op = k
    return op

@jit(nopython=True)
def operator_to_bond(op, Lx, Ly):
    n_sites = Lx * Ly
    bond = np.zeros(2, np.intp)
    k = 2 * n_sites - 1
    if op < 2 * n_sites:
        bond[0] = op // 2
        bond[1] = op // 2
    elif op >= 2 * n_sites:
        # TODO: this cycle could be optimised
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    continue
                k += 1
                if op == k:
                    bond[0] = i
                    bond[1] = j
    return bond


@jit(nopython=True)
def distance_pbc(s0, s1, Lx, Ly):
    xy_0 = site_to_xy(s0, Lx, Ly)
    xy_1 = site_to_xy(s1, Lx, Ly)
    dx = abs(xy_1[0] - xy_0[0])
    dy = abs(xy_1[1] - xy_0[1])
    if dx > Lx / 2:
        dx = Lx - dx
    elif dx < - Lx / 2 + 1:
        dx = Lx + dx
    if dy > Ly / 2:
        dy = Ly - dy
    elif dy < - Ly / 2 + 1:
        dy = Ly + dy
    dist = math.hypot(dx, dy)
    return dist



@jit(nopython=True)
def vec_min(s0, s1, Lx: int, Ly: int):
    xy_0 = site_to_xy(s0, Lx, Ly)
    xy_1 = site_to_xy(s1, Lx, Ly)
    dx = abs(xy_1[0] - xy_0[0])
    dy = abs(xy_1[1] - xy_0[1])
    if dx > Lx // 2:
        dx = Lx - dx
    elif dx < - Lx // 2 + 1:
        dx = Lx + dx
    if dy > Ly // 2:
        dy = Ly - dy
    elif dy < - Ly // 2 + 1:
        dy = Ly + dy
    size = (Lx // 2 + 1, Ly // 2 + 1)
    #size = (8, 8)
    vec_list = np.zeros(size)
    nc = 0
    
    # TODO: would be much harder to optimize, but presumably still possible.
    for dx_ in range(Lx // 2 + 1):
        for dy_ in range(Ly // 2 + 1):
            if dx_ <= dy_:
                vec_list[int(dx_)][int(dy_)] = vec_list[int(dy_)][int(dx_)] = nc
                nc += 1

    return int(vec_list[int(dx)][int(dy)])
@jit(nopython=True)
def potential(s0, s1, Lx, Ly):
    dist = distance_pbc(s0, s1, Lx, Ly)
    if dist == 0.0:
        raise ValueError("PBC distance zero in potential!")
    elif dist > cutoff:
        return 0.0
    else:
        rij = dist / a
        #print("rij = ", rij)
        return (Rb / rij) ** 6


@jit(nopython=True)
def init_SSE_square(Lx, Ly):
    n_sites = Lx * Ly
    spins = 2 * np.mod(np.random.permutation(n_sites), 2) - 1
    op_string = -1 * np.ones(20, np.intp)
    return spins, op_string

@jit(nopython=True)
def V_i(n_sites):
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    # TODO: check that operations are performermed in the right order.
    size_v = (Lx // 2 + 1) * (Lx // 2 + 2) // 2
    Vi = np.zeros(size_v)
    # TODO: use MAP or smt, no for loops
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            vec = vec_min(i, j, Lx, Ly)
            #print(vec)
            Vi[vec] = potential(i, j, Lx, Ly) / 2  # factor 0.5 from double counting
    return Vi

@jit(nopython=True)
def d_b(n_sites):
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    # TODO: check that operations are performermed in the right order.
    size_v = (Lx // 2 + 1) * (Lx // 2 + 2) // 2
    db = np.zeros(size_v)
    d_norm = 0.0
    # TODO: no for loops!
    '''
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:
                dist = distance_pbc(i, j, Lx, Ly)
                if dist <= cutoff:
                    d_norm += 1
    '''
    i1 = site(Lx // 2, Ly // 2, Lx, Ly)
    print(i1)
    for j1 in range(n_sites):
        if j1 != i1:
            dist = distance_pbc(i1, j1, Lx, Ly)
            if dist <= cutoff:
                print("r = ", dist)
                d_norm += 1
                
    print(d_norm) # 4N
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            vec = vec_min(i, j, Lx, Ly)
            dist = distance_pbc(i, j, Lx, Ly)
            if dist > cutoff:
                db[vec] = 0.0
            else:
                db[vec] = d / d_norm / 2
            #db = d / (n_sites - 1) / 2
            #db = Vi / 2 
            #print("db = ", db, " Vi = ", Vi, " 2 * db - Vi = ", 2 * db - Vi)
            #db[vec] = d / (n_sites - 1) / 2
    return db

@jit(nopython=True)
def C_i(dbi, n_sites):
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    # TODO: check that operations are performermed in the right order.
    size_v = (Lx // 2 + 1) * (Lx // 2 + 2) // 2
    Ci = np.zeros(size_v)
    # TODO: no for loops!
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            vec = vec_min(i, j, Lx, Ly)
            Vi = potential(i, j, Lx, Ly) / 2
            db = dbi[vec]
            #db = d / (n_sites - 1) / 2
            #db = Vi / 2 
            #print("db = ", db, " Vi = ", Vi, " 2 * db - Vi = ", 2 * db - Vi)
            Ci[vec] = abs(min(0.0, min(db, 2 * db - Vi))) + eps * abs(min(db, 2 * db - Vi))
            if Ci[vec] < 0.0000000000000001:
                Ci[vec] = 0.0
    return Ci

@jit(nopython=True)
def init_prob_2d(dbi, n_sites):
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    prob_dist = np.zeros((n_sites, n_sites))
    # TODO: AVOID LOOPS, write in matrices!
    for i in range(n_sites):
        for j in range(i, n_sites):
            vec = vec_min(i, j, Lx, Ly)
            if i == j:
                prob_dist[i][j] = Omega * 0.5
            else:
                Vi = potential(i, j, Lx, Ly) / 2
                #db = d / (n_sites - 1)
                #db = Vi / 2 
                db = dbi[vec]
                Ci = abs(min(0.0, min(db, 2 * db - Vi))) + eps * abs(min(db, 2 * db - Vi))
                W1 = Ci
                W2 = db + Ci
                W3 = db + Ci
                W4 = - Vi + 2 * db + Ci
                prob_dist[i, j] = max(W1, max(W2, max(W3, W4)))
                prob_dist[j, i] = max(W1, max(W2, max(W3, W4)))
                #prob_dist[j, i] = 0.0
    return prob_dist

@jit(nopython=True)
def cumulative(n_sites, P_ij):
    # P_cumulfirst = np.zeros(n_sites)
    # sum_by_i = [sum(row) for row in P_ij]
    # qi = sum_by_i[0]
    # for i in range(n_sites):
    #     P_cumulfirst[i] = qi

    #P_cumulfirst = sum(P_ij[0]) * np.ones(n_sites)
    P_cumulfirst = (1 / n_sites) * np.ones(n_sites) # PBC <-> translation symmetry <-> uniform distribution
    return P_cumulfirst

def norm_triangular(n_sites, P_ij):
    return np.triu(P_ij).sum()

@jit(nopython=True)
def binary_search_sample(cumulative_probs, target):
    left = 0
    right = len(cumulative_probs) - 1

    while left < right:
        mid = (left + right) // 2

        if cumulative_probs[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left

@jit(nopython=True)
def sample_from_distribution(probs):
    cumulative_probs = np.cumsum(probs)

    total_prob = cumulative_probs[-1]
    cumulative_probs /= total_prob

    target = np.random.rand()
    #print("target = ", target)

    sampled_index = binary_search_sample(cumulative_probs, target)

    return sampled_index


@jit(nopython=True)
def diagonal_update(spins, op_string, dbi, Vi, Ci, Pij, Pc, beta):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    M = op_string.shape[0]
    n = np.sum(op_string != -1)
    #norm = np.triu(Pij).sum() #np.sum(Pij)
    norm = np.sum(Pij) #whole matrix
    print("norm = ", norm) #47.77574400000002
    prob_ratio = norm * beta
    
    for p in range(M):
        op = op_string[p]
        if op == -1:
            index1 = sample_from_distribution(Pc)
            index2 = sample_from_distribution(Pij[index1])
            #s0 = min(index1, index2)
            #s1 = max(index1, index2)
            s0 = index1
            s1 = index2
            #print("sampled index1 = ", index1, " index2 = ", index2)
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
                    #db = V / 2 
                    #db = d / (n_sites - 1)
                    db = dbi[vec]
                    if spins[s0] == -1 and spins[s1] == -1:
                        W_actual = C
                    elif spins[s0] == -1 and spins[s1] == 1:
                        W_actual = db + C
                    elif spins[s0] == 1 and spins[s1] == -1:
                        W_actual = db + C
                    elif spins[s0] == 1 and spins[s1] == 1:
                        W_actual = - V + 2 * db + C
                    W_sampled = Pij[s0, s1]
                    #print("sampled s0 = ", s0, " s1 = ", s1)
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

@jit(nopython=True)
def create_linked_vertex_list(spins, op_string):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    M = op_string.shape[0]
    vertex_list = np.zeros(4 * M, np.intp)
    first_vertex_at_site = -1 * np.ones(n_sites, np.intp) 
    last_vertex_at_site = -1 * np.ones(n_sites, np.intp)
    for p in range(M):
        v0 = p * 4
        v1 = v0 + 1
        op = op_string[p]
        if op == -1:
            vertex_list[v0:v0+4] = -2
        elif op >= 2 * n_sites:
            bond = operator_to_bond(op, Lx, Ly)
            s0 = bond[0]
            s1 = bond[1]
            v2 = last_vertex_at_site[s0]
            v3 = last_vertex_at_site[s1]
            if v2 == -1:
                first_vertex_at_site[s0] = v0
            else:
                vertex_list[v2] = v0
                vertex_list[v0] = v2
            if v3 == -1:
                first_vertex_at_site[s1] = v1
            else:
                vertex_list[v3] = v1
                vertex_list[v1] = v3
            last_vertex_at_site[s0] = v0 + 2
            last_vertex_at_site[s1] = v0 + 3
        elif op < 2 * n_sites:
            bond = operator_to_bond(op, Lx, Ly)
            s0 = bond[0]
            s1 = bond[1] 
            if s0 != s1:
                raise ValueError("site operator not on site!")
            v2 = last_vertex_at_site[s0]
            if v2 == -1:
                first_vertex_at_site[s0] = v0
            else:
                vertex_list[v2] = v0
                vertex_list[v0] = v2
            last_vertex_at_site[s0] = v0 + 2
            vertex_list[v0+1] = -2
            vertex_list[v0+3] = -2
    for s0 in range(n_sites):
        v0 = first_vertex_at_site[s0]
        if v0 != -1: # there is an operator acting on that site -> create link
            v1 = last_vertex_at_site[s0] 
            vertex_list[v1] = v0 
            vertex_list[v0] = v1
            
    v0 = 0
    leg_counter = 0

    n_id = 0
    n_offd = 0
    n_1a = 0
    n_1b = 0

    for i in range(M):
        if op_string[i] == -1:
            n_id += 1
        elif op_string[i] < 2 * n_sites and op_string[i] % 2 != 0:
            n_offd += 1
        elif op_string[i] < 2 * n_sites and op_string[i] % 2 == 0:
            n_1a += 1
        elif op_string[i] >= 2 * n_sites:
            n_1b += 1

    n_legs = 4 * n_1b + 2 * n_1a + 2 * n_offd

    for v in range(4*M):
        if vertex_list[v] != -2:
            leg_counter += 1
            if v != vertex_list[vertex_list[v]]:
                print(f"leg = {v} vertex_list[vertex_list[v]] = {vertex_list[vertex_list[v]]}")
                raise ValueError("inconsistent list")

    if leg_counter != n_legs:
        raise ValueError("linked list has missing links, leg_counter != n_legs")
    
    #vertex_list = np.array(vertex_list, dtype = np.intp)
    #first_vertex_at_site = np.array(first_vertex_at_site, dtype = np.intp)
    return vertex_list, first_vertex_at_site

@jit(nopython=True)
def change_type(v0, op_string, n_sites):
    p = v0 // 4
    if op_string[p] < 2 * n_sites and op_string[p] % 2 != 0:
        op_string[p] -= 1
    elif op_string[p] < 2 * n_sites and op_string[p] % 2 == 0:
        op_string[p] += 1
        
@jit(nopython=True)
def clusterupdate(spins, op_string, vertex_list, first_vertex_at_site, dbi, V_i, C_i):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    M = op_string.shape[0]
    max_ghostlegs = 4
    n_ghostlegs = max_ghostlegs * M
    color = np.zeros(n_ghostlegs, np.intp)
    stack_legs = -1 * np.ones(n_ghostlegs, np.intp)
    
    nc = 0
    for v0 in range(0, n_ghostlegs, 4):
        if vertex_list[v0] < 0:
            continue
        if color[v0] > 0:
            continue
        nc += 1
        vs = v0
        v1 = v0
        nd = 0
        while True:
            
            while True:
                color[v1] = nc
                if op_string[v1//4] >= 2 * n_sites:
                    if color[v1 ^ 1] == 0:
                        nd += 1
                        stack_legs[nd] = v1 ^ 1
                v2 = vertex_list[v1]
                color[v2] = nc
                if op_string[v2//4] < 2 * n_sites:
                    break
                elif op_string[v2//4] >= 2 * n_sites:
                    v1 = v2 ^ 2 # flips 0<->2, 1<->3 ... this is adjacent leg
                    if color[v2 ^ 1] == 0:
                        nd += 1
                        stack_legs[nd] = v2 ^ 1
                if v1 == vs:
                    break
                    
            if not op_string[vs//4] < 2 * n_sites:
                v1 = vs ^ 2
                
                while True:
                    color[v1] = nc
                    if op_string[v1//4] >= 2 * n_sites:
                        if color[v1 ^ 1] == 0:
                            nd += 1
                            stack_legs[nd] = v1 ^ 1
                    v2 = vertex_list[v1]
                    if v2 == vs:
                        break
                    color[v2] = nc
                    if op_string[v2//4] < 2 * n_sites:
                        break
                    else:
                        v1 = v2 ^ 2
                        if color[v2 ^ 1] == 0:
                            nd += 1
                            stack_legs[nd] = v2 ^ 1
                            
            while True:
                finished = False
                if nd == 0:
                    finished = True
                    break
                vs = stack_legs[nd]
                v1 = stack_legs[nd]
                nd -= 1
                if not color[v1] > 0:
                    break
            if finished == True:
                break
    
    
    nc1 = nc
    flip = np.zeros(nc1, np.intp)
    f_p = np.ones((nc1, 2), np.float64)
    
    for p in range(M):
        op = op_string[p]
        if op == -1:
            continue
        elif op >= 2 * n_sites:
            bond = operator_to_bond(op, Lx, Ly)
            s1 = bond[0]
            s2 = bond[1]
            vec = vec_min(s1, s2, Lx, Ly)
            db = dbi[vec]
            nc = color[4 * p]
            #print("spins = ", spins[s1], " ", spins[s2])
            if spins[s1] == -1 and spins[s2] == -1:
                f_p[nc - 1][0] *=  - V_i[vec] + 2 * db + C_i[vec] #flipped
                f_p[nc - 1][1] *=  C_i[vec] #not flipped
            elif spins[s1] == 1 and spins[s2] == 1:
                f_p[nc - 1][0] *=  C_i[vec]
                f_p[nc - 1][1] *=  - V_i[vec] + 2 * db + C_i[vec]
        elif op < 2 * n_sites:
            bond = operator_to_bond(op, Lx, Ly)
            s1 = bond[0]
            s2 = bond[1]
            nc = color[4 * p]
            if op % 2 != 0:
                spins[s1] = - spins[s1]
    
    for i_nc in range(nc1):
        #print("nc = ", i_nc, " w' = ", f_p[i_nc][0], " w = ", f_p[i_nc][1])
        w = f_p[i_nc][0] / f_p[i_nc][1]
        if abs(w - 1.0) < 0.00000001:
            w = 0.5
        if np.random.rand() < w:
            flip[i_nc] = 1
    
    for v0 in range(0, n_ghostlegs, 2):
        if vertex_list[v0] < 0:
            continue
        if op_string[v0//4] < 2 * n_sites:
            if flip[color[v0] - 1] == 1:
                change_type(v0, op_string, n_sites)
                vertex_list[v0] = -1
            else:
                vertex_list[v0] = -2
        elif op_string[v0//4] >= 2 * n_sites:
            if flip[color[v0] - 1] == 1:
                vertex_list[v0] = -1
                vertex_list[v0 + 1] = -1
            else:
                vertex_list[v0] = -2
                vertex_list[v0 + 1] = -2
    
    for i in range(n_sites):
        if first_vertex_at_site[i] != -1:
            if vertex_list[first_vertex_at_site[i]] == -1:
                spins[i] = - spins[i]
        else:
            if np.random.rand() < 0.5:
                spins[i] = - spins[i]

@jit(nopython=True)
def cluster_update(spins, op_string, dbi, V_i, C_i):
    vertex_list, first_vertex_at_site = create_linked_vertex_list(spins, op_string)
    #vertex_list = np.array(vertex_list, dtype = np.intp)
    #first_vertex_at_site = np.array(first_vertex_at_site, dtype = np.intp)
    clusterupdate(spins, op_string, vertex_list, first_vertex_at_site, dbi, V_i, C_i)
    
@njit
def resize(a, new_size):
    new = np.zeros(new_size, a.dtype)
    new[:a.size] = a
    return new

@jit(nopython=True)
def thermalize(spins, op_string, dbi, V_i, C_i, Pij, Pc, beta, n_updates_warmup):
    for _ in range(n_updates_warmup):
        n = diagonal_update(spins, op_string, dbi, V_i, C_i, Pij, Pc, beta)
        cluster_update(spins, op_string, dbi, V_i, C_i)
        print("n = ", n)
        M_old = len(op_string)
        M_new = n + n // 3
        if M_new > M_old:
            op_string = resize(op_string, M_new)
            op_string[M_old: ] = -1
        print("resized to ", op_string.shape[0])
    return op_string

@jit(nopython=True)
def get_staggering(Lx, Ly):
    stag = np.zeros(Lx*Ly, np.intp) 
    for x in range(Lx):
        for y in range(Ly):
            s = site(x, y, Lx, Ly) 
            stag[s] = (-1)**(x+y)
    return stag

@jit(nopython=True)
def staggered_magnetization(spins, stag): 
    return np.sum(spins * stag * 0.5)

@jit(nopython=True)
def measure(spins, op_string, dbi, V_i, C_i, Pij, Pc, stag, beta, n_updates_measure):
    ns = np.zeros(n_updates_measure)
    nums = np.zeros(n_updates_measure)
    ms = np.zeros(n_updates_measure)
    for idx in range(n_updates_measure):
        ns[idx] = diagonal_update(spins, op_string, dbi, V_i, C_i, Pij, Pc, beta) 
        print("n = ", ns[idx])
        #ms[idx] = np.abs(staggered_magnetization(spins, stag))
        #print("absolute magnetization = ", ms[idx])
        cluster_update(spins, op_string, dbi, V_i, C_i)
        ms[idx] = np.abs(staggered_magnetization(spins, stag))
        nums[idx] = np.sum(spins == 1)
        #num = spins.count(1)
        print("density = ", nums[idx])
        print("absolute magnetization = ", ms[idx])
    return ns, nums, ms

@jit(nopython=True)
def init_c(spins, C_i, Omega):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    
    c = Omega * 0.5

    for i in range(n_sites):
        for j in range(n_sites):
            vec = vec_min(i, j, Lx, Ly)
            if i != j:
                c += C_i[vec] / n_sites
    return c

#@jit(nopython=True)
def run_simulation(Lx, Ly, betas, n_updates_measure=10000, n_bins=10):
    spins, op_string = init_SSE_square(Lx, Ly)
    stag = get_staggering(Lx, Ly)
    n_sites = len(spins)
    Vi = V_i(n_sites)
    dbi = d_b(n_sites)
    Ci = C_i(dbi, n_sites)
    Pij = init_prob_2d(dbi, n_sites)
    Pc = cumulative(n_sites, Pij)
    n_betas = betas.shape[0]
    Es_Eerrs = np.zeros((n_betas, 2))
    Ns_Nerrs = np.zeros((n_betas, 2))
    Ms_Merrs = np.zeros((n_betas, 2))
    i_beta = 0
    c_sum = init_c(spins, Ci, Omega)
    print("added constant", c_sum)
    #file_n = open('density.txt', 'w+')
    #file_m = open('magnetization.txt', 'w+')
    open('energy.txt', 'w').close()
    open('density.txt', 'w').close()
    open('magnetization.txt', 'w').close()
    for beta in betas:
        # print("beta = {beta:.3f}".format(beta=beta), flush=True)
        print("beta = ", beta)
        op_string = thermalize(spins, op_string, dbi, Vi, Ci, Pij, Pc, beta, 10000)#n_updates_measure//10 100000
        Es = np.zeros(n_bins)
        Ns = np.zeros(n_bins)
        Ms = np.zeros(n_bins)
        for n_bin in range(n_bins):
            ns, nums, ms = measure(spins, op_string, dbi, Vi, Ci, Pij, Pc, stag, beta, n_updates_measure)
            n_mean = np.mean(ns)
            #E = (c_sum - n_mean/beta) / n_sites
            E = (- n_mean/beta) / n_sites + c_sum
            num_mean = np.mean(nums)
            N = num_mean / n_sites
            #ms_mean = np.mean(np.abs(ms))
            ms_mean = np.mean(ms)
            M = ms_mean / n_sites
            Es[n_bin] = E
            Ns[n_bin] = N
            Ms[n_bin] = M
            #file_e.write(E)
            #file_n.write(N)
            #file_m.write(M)
            #print(E, "\n", file=file_e)
            #print(N, "\n", file=file_n)
            #print(M, "\n", file=file_m)
            with open('energy.txt', 'a') as file_e:
                print(E, file=file_e)
            with open('density.txt', 'a') as file_n:
                print(N, file=file_n)
            with open('magnetization.txt', 'a') as file_m:
                print(M, file=file_m)
            #print(N, file=file_n)
            #print(M, file=file_m)
        Es_Eerrs[i_beta][0] = np.mean(Es)
        Es_Eerrs[i_beta][1] = np.std(Es)/np.sqrt(n_bins)
        Ns_Nerrs[i_beta][0] = np.mean(Ns)
        Ns_Nerrs[i_beta][1] = np.std(Ns)/np.sqrt(n_bins)
        Ms_Merrs[i_beta][0] = np.mean(Ms)
        Ms_Merrs[i_beta][1] = np.std(Ms)/np.sqrt(n_bins)
        i_beta = i_beta + 1
    #file_n.close()
    #file_m.close()
    return Es_Eerrs, Ns_Nerrs, Ms_Merrs

if __name__ == "__main__":
    start = time.time()
    Es_Eerrs, Ns_Nerrs, Ms_Merrs = run_simulation(12, 12, betas=np.array([40.0,]), n_updates_measure=1000, n_bins=400)
    total_time = time.time() - start

    beta = 40
    print(f"Execution time = {round(total_time, 2)} seconds.")
    print("Energy per site ={E:.8f} with error = {Eerr:.8f} at T={T:.3f}".format(E=Es_Eerrs[0,0], Eerr= Es_Eerrs[0,1], T=1./beta))
    print("Particle density ={N:.8f} with error = {Nerr:.8f} at T={T:.3f}".format(N=Ns_Nerrs[0,0], Nerr= Ns_Nerrs[0,1], T=1./beta))
    print("Magnetization per site ={M:.8f} with error = {Merr:.8f} at T={T:.3f}".format(M=Ms_Merrs[0,0], Merr= Ms_Merrs[0,1], T=1./beta))