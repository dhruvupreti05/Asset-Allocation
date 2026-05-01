import numpy as np
from .qubo import *

class AssetAllocation(Qubo):
    def __init__(self, returns, covariance, p=0.1, lambda_3=10, bits_per_asset=6):
        super().__init__()
        self.returns = np.array(returns)
        self.covariance = np.array(covariance)
        self.p = p
        self.lambda_3 = lambda_3
        self.bits_per_asset = bits_per_asset

    def build_qubo(self):
        n = len(self.returns)
        size = self.bits_per_asset * n
        Q = np.zeros((size, size))

        for u in range(size):
            for v in range(size):
                i = u // self.bits_per_asset
                a = (u % self.bits_per_asset) + 1

                j = v // self.bits_per_asset
                b = (v % self.bits_per_asset) + 1

                if u == v:
                    term1 = (2 ** (-2 * a)) * (
                        self.returns[i] ** 2
                        + self.p ** (-2)
                        + self.lambda_3 * self.covariance[i, i]
                    )

                    term2 = 2 * (2 ** (-a)) * (
                        self.p * self.returns[i]
                        + self.p ** (-2)
                    )

                    Q[u, u] = term1 - term2

                else:
                    Q[u, v] = (2 ** (-a - b)) * (
                        self.returns[i] * self.returns[j]
                        + self.p ** (-2)
                        + self.lambda_3 * self.covariance[i, j]
                    )

        return Q
    
    def decode_solution(self, x):
        n = len(self.returns)
        allocations = []

        for i in range(n):
            bits = x[
                i * self.bits_per_asset : (i + 1) * self.bits_per_asset
            ]

            value = sum(
                bit * 2 ** (-j)
                for j, bit in enumerate(bits, start=1)
            )

            allocations.append(value)

        allocations = np.array(allocations)

        if allocations.sum() != 0:
            allocations = allocations / allocations.sum()

        return allocations