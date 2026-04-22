# Purdue Experimental Math Lab: Kaufmann Group

import numpy as np
import pandas as pd
import networkx as nx
import yfinance as yf
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import colormaps

import dimod
from dwave.cloud import Client
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

endpoint = "https://cloud.dwavesys.com/sapi"
token = "ttYi-e51121f5f637eda79b9c88daffa50c66b7f08d10"

client = Client.from_config(token="ttYi-e51121f5f637eda79b9c88daffa50c66b7f08d10")
solver = "Advantage_system4.1"

num_partitions = 3
gamma=8.0
beta=1.0

def modularity_matrix(G):
    A = nx.to_numpy_array(G, weight="weight", dtype=float)
    g = A.sum(axis=1)
    m = g.sum() / 2.0
    B = A - np.outer(g, g) / (2.0 * m)
    return B, A, g, m

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

def draw_graph(Graph, name, labels=None, showEdgeWeights=False):
    pos = nx.spring_layout(Graph, seed=7)
    cmap = colormaps["tab10"]

    node_colors = "lightblue" if labels is None else [cmap(int(c) % 10) for c in labels]

    plt.figure(figsize=(8, 6))

    if showEdgeWeights:
        edges = list(Graph.edges(data=True))
        weights = np.array([d.get("weight", 0.0) for _, _, d in edges], dtype=float)
        if len(weights) > 0 and weights.max() != weights.min():
            norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
        else:
            norm_weights = np.ones_like(weights) * 0.5

        nx.draw_networkx(
            Graph,
            pos=pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=9,
            edge_color=norm_weights,
            edge_cmap=plt.cm.Greys,
            width=2,
        )
    else:
        nx.draw_networkx(
            Graph,
            pos=pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=9,
            edge_color="gray",
        )

    plt.title("Graph")
    plt.axis("off")
    plt.savefig(name, dpi=300)
    plt.show()

def solve_qubo(Q, num_reads=3000, num_repeats=1):
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

def decode_one_hot(x, n, k):
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        vals = np.array([x[c * n + i] for c in range(k)])
        labels[i] = int(np.argmax(vals))
    return labels

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

def binary_to_float(bits, split):
    chunk_size = len(bits) // split
    decimals = []

    for i in range(split):
        chunk = bits[i * chunk_size : (i + 1) * chunk_size]
        value = sum(bit * 2**-j for j, bit in enumerate(chunk, 1))
        decimals.append(value)

    return decimals

if __name__ == "__main__":
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "JPM", "V", "JNJ",
            "WMT", "PG", "MA", "DIS", "HD", "BAC", "VZ", "ADBE", "CMCSA", "NFLX"]

    closing_prices = yf.download(assets, start="2020-01-01")["Close"]
    daily_returns = closing_prices.pct_change().dropna()

    covariance_matrix = daily_returns.cov().copy()
    #covariance_matrix.iloc[np.diag_indices_from(covariance_matrix)] = 0

    returns = daily_returns.mean() * 252

    Graph = nx.from_numpy_array(covariance_matrix.to_numpy())
    draw_graph(Graph=Graph, name=".graph.png", showEdgeWeights=True)

    B, _, _, _ = modularity_matrix(G=Graph)
    Q_communities = build_community_detection_qubo(B, k=num_partitions, gamma=gamma, beta=beta)
    best_sample_communities, _, _ = solve_qubo(Q=Q_communities, num_repeats=1)

    communities = decode_one_hot(n=len(assets), k=num_partitions, x=best_sample_communities)

    comms = {}
    nodes = list(Graph.nodes())
    for node, c in zip(nodes, communities):
        comms.setdefault(int(c), set()).add(node)

    partitions = [group for group in comms.values() if group]

    draw_graph(Graph=Graph, name=".graph_with_communities.png", labels=communities, showEdgeWeights=True)

    group_average_returns = {}
    group_daily_returns = np.zeros((len(daily_returns), num_partitions))

    for group_index, asset_group in enumerate(partitions):
        asset_group = list(asset_group)

        average_asset_returns = [returns.iloc[asset] for asset in asset_group]
        group_average_returns[group_index] = np.mean(average_asset_returns)

        mean_daily_return_series = daily_returns.iloc[:, asset_group].mean(axis=1)
        group_daily_returns[:, group_index] = mean_daily_return_series.to_numpy()

    partition_covariance_matrix = np.cov(group_daily_returns, rowvar=False)

    Q_upper = build_aa_qubo(n=num_partitions, mu=group_average_returns, C=partition_covariance_matrix)
    best_sample_upper, _, _ = solve_qubo(Q=Q_upper)

    upper_allocations = binary_to_float(bits=best_sample_upper, split=num_partitions)

    lower_allocations = []

    for i, cluster in enumerate(partitions):
        cluster = list(cluster)

        cluster_returns = [returns.iloc[asset] for asset in cluster]
        cluster_covariance = covariance_matrix.iloc[cluster, cluster].to_numpy()

        cluster_qubo = build_aa_qubo(
            n=len(cluster),
            mu=cluster_returns,
            C=cluster_covariance
        )

        best_sample_lower, _, _ = solve_qubo(Q=cluster_qubo)
        lower_allocations.append(binary_to_float(bits=best_sample_lower, split=len(cluster)))

    allocations = np.array([x * y for x, group in zip(upper_allocations, lower_allocations) for y in group])
    allocations /= sum(allocations)
    for asset, allocation in zip(assets, allocations):
        print(f"{asset}: {allocation}")



