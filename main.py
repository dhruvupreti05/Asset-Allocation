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

if __name__ == "__main__":
    with open('assets.txt', 'r') as file:
        assets = [line.strip() for line in file]

        daily_returns = closing_prices(assets=assets, start="2020-01-01")
        returns = daily_returns.mean() * 252

        covariance_matrix = get_covarience(daily_returns=daily_returns)
    
        Graph = nx.from_numpy_array(covariance_matrix.to_numpy())
        draw_graph(Graph=Graph, name=".graph.png")

        community_detection = CommunityDetection(adjacency_matrix=covariance_matrix, k=num_partitions)
        communities = community_detection.run(token=token)

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

            group_average_returns[group_index] = returns.iloc[asset_group].mean()
            group_daily_returns[:, group_index] = daily_returns.iloc[:, asset_group].mean(axis=1).to_numpy()

        partition_covariance_matrix = np.cov(group_daily_returns, rowvar=False)

        upper_allocation = AssetAllocation(returns=list(group_average_returns.values()), covariance=partition_covariance_matrix)
        upper_allocations = upper_allocation.run(token=token)

        lower_allocations = []

        for i, cluster in enumerate(partitions):
            cluster = list(cluster)

            cluster_returns = [returns.iloc[asset] for asset in cluster]
            cluster_covariance = covariance_matrix.iloc[cluster, cluster].to_numpy()

            inner_allocation = AssetAllocation(returns=cluster_returns, covariance=cluster_covariance)
            lower_allocations.append(inner_allocation.run(token=token))
    
        allocations = np.array([x * y for x, group in zip(upper_allocations, lower_allocations) for y in group])
        for asset, allocation in zip(assets, allocations):
            print(f"{asset}: {allocation}")



