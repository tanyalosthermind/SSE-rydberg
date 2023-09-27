import numpy as np
from numba import njit
import math
# import random
# from collections import Countera = 1.0
from numba import config
from rydberg.assets import njit_kwargs, disable_jit
config.DISABLE_JIT = disable_jit


@njit(**njit_kwargs)
def site(x, y, Lx, Ly):
    # TODO: generalize to the rectangular case
    return y * Lx + x

@njit(**njit_kwargs)
def site_to_xy(site, Lx, Ly):
    # TODO: generalize to the rectangular case
    xy = np.zeros(2)
    xy[0] = site % Lx
    xy[1] = site // Lx
    return xy

@njit(**njit_kwargs)
def bond_to_operator(s0, s1, Lx, Ly):
    n_sites = Lx * Ly
    k = 2 * n_sites - 1
    if s0 != s1:
        # TODO: needs to be optimized, too.
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                k += 1
                if i == min(s0, s1) and j == max(s0, s1):
                    op = k
    return op

@njit(**njit_kwargs)
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
            for j in range(i + 1, n_sites):
                k += 1
                if op == k:
                    bond[0] = i
                    bond[1] = j
    return bond


@njit(**njit_kwargs)
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

@njit(**njit_kwargs)
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


@njit(**njit_kwargs)
def potential(s0, s1, Lx, Ly, a, Rb):
    dist = distance_pbc(s0, s1, Lx, Ly)
    if dist == 0.0:
        raise ValueError("PBC distance zero in potential!")
    elif dist > 1.0:
        return 0.0
    else:
        rij = dist / a
        # # print("rij = ", rij)
        return (Rb / rij) ** 6


@njit(**njit_kwargs)
def init_SSE_square(Lx, Ly):
    n_sites = Lx * Ly
    spins = 2 * np.mod(np.random.permutation(n_sites), 2) - 1
    op_string = -1 * np.ones(10, np.intp)
    return spins, op_string


@njit(**njit_kwargs)
def V_i(n_sites, a, Rb):
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    # TODO: check that operations are performermed in the right order.
    size_v = (Lx // 2 + 1) * (Lx // 2 + 2) // 2
    Vi = np.zeros(size_v)
    # TODO: use MAP or smt, no for loops
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            vec = vec_min(i, j, Lx, Ly)
            Vi[vec] = potential(i, j, Lx, Ly, a, Rb)
    return Vi


@njit(**njit_kwargs)
def C_i(n_sites, a, Rb):
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    # TODO: check that operations are performermed in the right order.
    size_v = (Lx // 2 + 1) * (Lx // 2 + 2) // 2
    Ci = np.zeros(size_v)
    # TODO: no for loops!
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            vec = vec_min(i, j, Lx, Ly)
            Vi = potential(i, j, Lx, Ly, a, Rb)
            #db = d / (n_sites - 1);
            db = Vi / 2
            # # print("db = ", db, " Vi = ", Vi, " 2 * db - Vi = ", 2 * db - Vi)
            Ci[vec] = abs(min(0.0, min(db, 2 * db - Vi)))
    return Ci


@njit(**njit_kwargs)
def init_prob_2d(n_sites, a, Rb, Omega):
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
                Vi = potential(i, j, Lx, Ly, a, Rb)
                db = Vi / 2
                Ci = abs(min(0.0, min(db, 2 * db - Vi)))
                W1 = Ci
                W2 = db + Ci
                W3 = db + Ci
                W4 = - Vi + 2 * db + Ci
                prob_dist[i][j] = max(W1, max(W2, max(W3, W4)))
                #prob_dist[j][i] = max(W1, max(W2, max(W3, W4)))
                prob_dist[j][i] = 0.0
    return prob_dist


@njit(**njit_kwargs)
def cumulative(n_sites, P_ij):
    # P_cumulfirst = np.zeros(n_sites)
    # sum_by_i = [sum(row) for row in P_ij]
    # qi = sum_by_i[0]
    # for i in range(n_sites):
    #     P_cumulfirst[i] = qi

    #P_cumulfirst = sum(P_ij[0]) * np.ones(n_sites)
    P_cumulfirst = (1 / n_sites) * np.ones(n_sites)
    return P_cumulfirst

@njit(**njit_kwargs)
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

@njit(**njit_kwargs)
def sample_from_distribution(probs):
    cumulative_probs = np.cumsum(probs)

    total_prob = cumulative_probs[-1]
    cumulative_probs /= total_prob

    target = np.random.rand()
    # # print("target = ", target)

    sampled_index = binary_search_sample(cumulative_probs, target)

    return sampled_index