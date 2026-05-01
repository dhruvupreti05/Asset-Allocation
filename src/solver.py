import numpy as np
from tqdm import tqdm

import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

solver = "Advantage_system4.1"
endpoint = "https://cloud.dwavesys.com/sapi"

def solve_qubo(Q, token, num_reads=3000, num_repeats=1):
    global client 

    n = Q.shape[0]

    Q_dict = {}
    for i in range(n):
        for j in range(i, n):
            if Q[i, j] != 0:
                Q_dict[(i, j)] = float(Q[i, j])

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token, solver=solver))

    all_sets = []

    for i in tqdm(range(num_repeats)):
        ss = sampler.sample(bqm, num_reads=num_reads)
        all_sets.append(ss)

    sampleset = dimod.concatenate(all_sets)
    lowest = sampleset.lowest(rtol=0, atol=0).aggregate()

    best = max(lowest.data(["sample", "energy", "num_occurrences"]),
              key=lambda row: row.num_occurrences)

    best_sample = best.sample
    best_energy = best.energy

    print(f"Number of minima: {len(lowest)}")

    x = np.array([best_sample[i] for i in range(n)], dtype=int)

    return x, best_energy, sampleset
