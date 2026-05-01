import numpy as np
from .qubo import *

class CommunityDetection(Qubo):
    def __init__(self, adjacency_matrix, k, gamma=8.0, beta=1.0):
        super().__init__()
        self.adjacency_matrix = np.asarray(adjacency_matrix)
        self.k = k
        self.gamma = gamma
        self.beta = beta

    def modularity_matrix(self):
        A = self.adjacency_matrix
        g = A.sum(axis=1)
        m = g.sum() / 2.0

        B = A - np.outer(g, g) / (2.0 * m)

        return B

    def build_qubo(self):
        B = self.modularity_matrix()

        n = B.shape[0]
        N = n * self.k

        Q = np.zeros((N, N), dtype=float)

        for c in range(self.k):
            sl = slice(c * n, (c + 1) * n)
            Q[sl, sl] += self.beta * B

        for i in range(n):
            idxs = [c * n + i for c in range(self.k)]

            for idx in idxs:
                Q[idx, idx] += -self.gamma

            for a in range(self.k):
                for b in range(a + 1, self.k):
                    ia, ib = idxs[a], idxs[b]
                    Q[ia, ib] += self.gamma
                    Q[ib, ia] += self.gamma

        return Q

    def decode_solution(self, x):
        n = self.adjacency_matrix.shape[0]
        labels = np.zeros(n, dtype=int)

        for i in range(n):
            vals = np.array([
                x[c * n + i]
                for c in range(self.k)
            ])

            labels[i] = int(np.argmax(vals))

        return labels