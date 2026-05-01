import numpy as np
import networkx as nx

"""
Creates the modularity matrix.
"""
def modularity_matrix(G):
    A = nx.to_numpy_array(G, weight="weight", dtype=float)
    g = A.sum(axis=1)
    m = g.sum() / 2.0
    B = A - np.outer(g, g) / (2.0 * m)
    return B, A, g, m


"""
"""
def build_community_detection_qubo(B, k, gamma, beta):
    n = B.shape[0]
    N = n * k
    Q = np.zeros((N, N), dtype=float)

    for c in range(k):
        sl = slice(c * n, (c + 1) * n)
        Q[sl, sl] += beta * B

    for i in range(n):
        idxs = [c * n + i for c in range(k)]

        for idx in idxs:
            Q[idx, idx] += -gamma

        for a in range(k):
            for b in range(a + 1, k):
                ia, ib = idxs[a], idxs[b]
                Q[ia, ib] += gamma
                Q[ib, ia] += gamma

    return Q

"""
"""
def decode_one_hot(x, n, k):
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        vals = np.array([x[c * n + i] for c in range(k)])
        labels[i] = int(np.argmax(vals))
    return labels

"""
"""

def binary_to_float(bits, split):
    chunk_size = len(bits) // split
    decimals = []

    for i in range(split):
        chunk = bits[i * chunk_size : (i + 1) * chunk_size]
        value = sum(bit * 2**-j for j, bit in enumerate(chunk, 1))
        decimals.append(value)

    return decimals

"""
"""
def build_aa_qubo(n, mu, C, p=0.1, lambda_3=10):
    size = 6 * n
    Q = np.zeros((size, size))

    for u in range(size):
        for v in range(size):
            i = u // 6
            a = (u % 6) + 1

            j = v // 6
            b = (v % 6) + 1

            if u == v:
                term1 = (2 ** (-2 * a)) * (mu[i]**2 + p**(-2) + lambda_3 * C[i, i])
                term2 = 2 * (2 ** (-a)) * (p * mu[i] + p**(-2))
                Q[u, u] = term1 - term2
            else:
                Q[u, v] = (2 ** (-a - b)) * (mu[i] * mu[j] + p**(-2) + lambda_3 * C[i, j])

    return Q
