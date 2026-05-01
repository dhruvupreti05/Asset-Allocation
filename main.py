# Purdue Experimental Math Lab: Kaufmann Group

import numpy as np
import pandas as pd
import networkx as nx

import os
import numpy as np
from dotenv import load_dotenv
from src import *

load_dotenv()

token = os.getenv("API_TOKEN")

num_partitions = 3
gamma=8.0
beta=1.0

"""
Returns
"""
def getReturns(allocations, returns):
    return np.dot(returns, allocations)

"""
Risk
"""
def getRisk(covariance, allocations):
    return np.transpose(allocations) @ covariance @ allocations

if __name__ == "__main__":
    with open('assets.txt', 'r') as file:
        assets = [line.strip() for line in file]

        daily_returns = closing_prices(assets=assets, start="2020-01-01")
        returns = daily_returns.mean() * 252

        covariance_matrix = daily_returns.cov().copy()
        np.fill_diagonal(covariance_matrix.values, 0)

    
        Graph = nx.from_numpy_array(covariance_matrix.to_numpy())
        draw_graph(Graph=Graph, name=".graph.png")

        B, _, _, _ = modularity_matrix(G=Graph)
        Q_communities = build_community_detection_qubo(B, k=num_partitions, gamma=gamma, beta=beta)
        best_sample_communities, _, _ = solve_qubo(Q=Q_communities, token=token, num_repeats=1)

        communities = decode_one_hot(n=len(assets), k=num_partitions, x=best_sample_communities)

        comms = {}
        nodes = list(Graph.nodes())
        for node, c in zip(nodes, communities):
            comms.setdefault(int(c), set()).add(node)

        partitions = [group for group in comms.values() if group]

        draw_graph(Graph=Graph, name=".graph_with_communities.png", labels=communities)

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
        best_sample_upper, _, _ = solve_qubo(Q=Q_upper, token=token)

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

            best_sample_lower, _, _ = solve_qubo(Q=cluster_qubo, token=token)
            lower_allocations.append(binary_to_float(bits=best_sample_lower, split=len(cluster)))

        allocations = np.array([x * y for x, group in zip(upper_allocations, lower_allocations) for y in group])
        allocations /= sum(allocations)
        for asset, allocation in zip(assets, allocations):
            print(f"{asset}: {allocation}")



