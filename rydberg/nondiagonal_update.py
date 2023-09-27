import numpy as np
from numba import njit
from rydberg.assets import njit_kwargs
from rydberg.configuration import operator_to_bond, vec_min
from numba import config
from rydberg.assets import disable_jit
config.DISABLE_JIT = disable_jit


@njit(**njit_kwargs)
def create_linked_vertex_list(spins, op_string):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    M = op_string.shape[0]
    vertex_list = np.zeros(4 * M, np.intp)
    first_vertex_at_site = -1 * np.ones(n_sites, np.intp) 
    last_vertex_at_site = -1 * np.ones(n_sites, np.intp)
    spin_m = np.zeros(4 * M, np.intp)
    
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
            spin_m[v0]   = spins[s0] 
            spin_m[v0+1] = spins[s1]
            spin_m[v0+2] = spins[s0]
            spin_m[v0+3] = spins[s1]
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
            if np.mod(op, 2) == 1: #off diag
                spins[s0] = - spins[s0]
            spin_m[v0]   = spins[s0] 
            spin_m[v0+1] = spins[s1]
            spin_m[v0+2] = spins[s0]
            spin_m[v0+3] = spins[s1]
            
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
    return vertex_list, first_vertex_at_site, spin_m

@njit(**njit_kwargs)
def change_type(v0, op_string, n_sites):
    p = v0 // 4
    if op_string[p] < 2 * n_sites and op_string[p] % 2 != 0:
        op_string[p] -= 1
    elif op_string[p] < 2 * n_sites and op_string[p] % 2 == 0:
        op_string[p] += 1
        
@njit(**njit_kwargs)
def clusterupdate(spins, op_string, vertex_list, first_vertex_at_site, V_i, C_i):
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
            db = V_i[vec] / 2
            nc = color[4 * p]
            # # print("spins = ", spins[s1], " ", spins[s2])
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
        # # print("w' = ", f_p[i_nc][0], " w = ", f_p[i_nc][1])
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

@njit(**njit_kwargs)
def prob(spins0, spins1, vec, f_p, V_i, C_i, prob_in):
    prob_in[0] = 1
    #print("spin0 = ", spins0, " spin1 = ", spins1, " vec = ", vec)
    db = V_i[vec] / 2
    if spins0 == -1 and spins1 == -1:
        f_p[0] *=  db + C_i[vec] #flipped
        f_p[1] *=  C_i[vec] #not flipped
    elif spins0 == 1 and spins1 == 1:
        f_p[0] *=  db + C_i[vec]
        f_p[1] *=  - V_i[vec] + 2 * db + C_i[vec]
    elif spins0 == -1 and spins1 == 1: #W2
        f_p[0] *=  - V_i[vec] + 2 * db + C_i[vec] #flipped: W2 -> W4
        f_p[1] *=  db + C_i[vec] #not flipped
    elif spins0 == 1 and spins1 == -1: #W3
        f_p[0] *=  C_i[vec] #flipped: W3 -> W1
        f_p[1] *=  db + C_i[vec] #not flipped

@njit(**njit_kwargs)
def lineupdate(spins, op_string, vertex_list, first_vertex_at_site, spin_m, V_i, C_i):
    n_sites = spins.shape[0]
    Lx = np.int32(n_sites**0.5)
    Ly = np.int32(n_sites**0.5)
    M = op_string.shape[0]
    max_ghostlegs = 4
    n_ghostlegs = max_ghostlegs * M
    f_p = np.ones(2, np.float64)
    
    loop = False
    prob_in = np.zeros(2, np.intp)
    
    for i in range(1001):
        factor = int(np.random.rand() * M)
        #print("random vertex = ", factor)
        v0 = factor * 4
        if op_string[v0 // 4] < 2 * n_sites and op_string[v0 // 4] >= 0:
            break
    ### ? return 
    #print("v0 = ", v0)
    if i > 1000:
        print("cringe")
        return
    
    bond = operator_to_bond(op_string[v0 // 4], Lx, Ly)
    s0 = bond[0]
    s1 = bond[1]
    #if s0 != s1:
        #-> no site operator was found in operator string! 
        #-> we need to flip each line as it is a cluster itself
    #    print("s0 = ", s0, " s1 = ", s1)
    #    raise ValueError("random vertex not from site operator ")

    vs = v0 #we move v1 as open end of the loop around until we come back to v0
    v1 = vs
    
    while True:
        op = op_string[v1 // 4]
        bond = operator_to_bond(op, Lx, Ly)
        s1 = bond[0]
        s2 = bond[1]
        if s1 != s2:
            vec = vec_min(s1, s2, Lx, Ly)
            spins0 = spin_m[v1]
            spins1 = spin_m[v1 ^ 1]
            prob(spins0, spins1, vec, f_p, V_i, C_i, prob_in)
            #print("prob_in = ", prob_in[0])
            #prop(spin_m[v1], spin_m[v1 ^ 1], vec, f_p, V_i, C_i)
        v2 = vertex_list[v1]
        if op_string[v2 // 4] < 2 * n_sites:
            break
        else:
            v1 = v2 ^ 2 
        if v1 == vs:
            loop = True
            break
            
    if not op_string[vs // 4] < 2 * n_sites or loop:
        v1 = vs ^ 2
        while True:
            v2 = vertex_list[v1]
            if v2 == vs:
                break
            if op_string[v2 // 4] < 2 * n_sites: 
                break
            else:
                v1 = v2 ^ 2
        
            op = op_string[v1 // 4]
            bond = operator_to_bond(op, Lx, Ly)
            s1 = bond[0]
            s2 = bond[1]

            if s1 != s2:
                vec = vec_min(s1, s2, Lx, Ly)
                spins0 = spin_m[v1]
                spins1 = spin_m[v1 ^ 1]
                prob(spins0, spins1, vec, f_p, V_i, C_i, prob_in)
                #print("prob_in = ", prob_in[0])
                #prop(spin_m[v1], spin_m[v1 ^ 1], vec, f_p, V_i, C_i)
                
    w = f_p[0] / f_p[1]
    #print("w = ", w)
    vs = v0
    v1 = vs
    
    if np.random.rand() < w:
        if op_string[vs // 4] < 2 * n_sites:
            change_type(vs, op_string, n_sites)
        while True:
            v2 = vertex_list[v1]
            vertex_list[v1] = -1
            vertex_list[v2] = -1
            
            if op_string[v2 // 4] < 2 * n_sites:
                change_type(v2, op_string, n_sites)
                break
            else:
                v1 = v2 ^ 2
            if v1 == vs:
                loop = True
                break
        
        if not op_string[vs // 4] < 2 * n_sites or loop:
            v1 = vs ^ 2
            while True:
                v2 = vertex_list[v1]
                vertex_list[v1] = -1
                vertex_list[v2] = -1
                if v2 == vs:
                    break
                if op_string[v2 // 4] < 2 * n_sites: 
                    change_type(v2, op_string, n_sites)
                    break
                else:
                    v1 = v2 ^ 2
    
    for i in range(n_sites):
        if first_vertex_at_site[i] != -1:
            if vertex_list[first_vertex_at_site[i]] == -1:
                spins[i] = - spins[i]
        else:
            if np.random.rand() < 0.5:
                spins[i] = - spins[i]


@njit(**njit_kwargs)
def cluster_update(spins, op_string, V_i, C_i, line, line_step):
    if line:
        for _ in range(line_step):
            vertex_list, first_vertex_at_site, spin_m = create_linked_vertex_list(spins, op_string)
            lineupdate(spins, op_string, vertex_list, first_vertex_at_site, spin_m, V_i, C_i)
    else:
        vertex_list, first_vertex_at_site,  spin_m = create_linked_vertex_list(spins, op_string)
        #vertex_list = np.array(vertex_list, dtype = np.intp)
        #first_vertex_at_site = np.array(first_vertex_at_site, dtype = np.intp)
        clusterupdate(spins, op_string, vertex_list, first_vertex_at_site, V_i, C_i)