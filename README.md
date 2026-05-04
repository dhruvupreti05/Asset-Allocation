# Asset-Allocation

## Introduction:

Modern portfolio management requires solving a fundamental challenge: how should an investor distribute capital across many assets in order to maximize returns while minimizing risk? This question, first formalized by Harry Markowitz in 1952, forms the basis of what is known as the portfolio optimization problem. The classical Markowitz approach seeks to minimize the variance of a portfolio subject to constraints on expected return, but solving this problem exactly becomes computationally expensive as the number of assets grows, since general quadratic optimization problems (QUBOs) are NP-hard. As financial markets grow in complexity and the number of tradable assets increases, there is a growing need for more scalable approaches to portfolio optimization.

Quantum annealing, implemented by hardware such as D-Wave's quantum processors, solves optimization problems by encoding them as Quadratic Unconstrained Binary Optimization (QUBO) problems. The goal is mainly to minimize a function of binary variables with both linear and quadratic terms. Prior research has shown that quantum annealers can produce portfolios achieving more than 80\% of the return of classical optimal solutions while satisfying risk constraints, suggesting real promise for quantum approaches in finance. Separately, researchers have demonstrated that quantum annealers can effectively detect community structure in networks by maximizing a modularity metric, a technique that identifies groups of densely connected nodes within a graph.

This project combines both of these ideas into a two-level hierarchical portfolio allocation framework. Using historical return data from 20 publicly traded stocks downloaded via Yahoo Finance, we first construct a covariance graph representing how asset returns move together. We then apply a quantum annealing-based community detection algorithm — formulated using the modularity matrix — to partition the 20 stocks into four communities of correlated assets. Within each community, we apply a second QUBO-based optimization to determine individual asset allocations. Our research investigates whether this hierarchical quantum approach can produce meaningful portfolio allocations, and examines practical challenges that arise when encoding financial constraints, such as the normalization requirement, into QUBO formulations.

<img width="723" height="623" alt="Screenshot 2026-04-28 at 3 06 36 PM" src="https://github.com/user-attachments/assets/5b74a187-38ff-46fd-a12d-75881bf23afc" />

## Set-Up:

First pull the repo in some directory you are conformatble working in.

``` 
git clone https://github.com/dhruvupreti05/Asset-Allocation.git
```

Following that make sure to change the permissions of the setup file, `chmod +x setup.sh` first then run the setup which will create a virtual environment `./setup.sh`. After that, whenever you want to use the project, activate the virtual environment by running `source ocean/bin/activate` to activate the virtual environment.

## References:

This work was supported by the collaborative efforts of Bhavya Lakhina, Esha Sury, Dhruv Upreti and Shashank Boopathi; mentorship was given by Kale Stahl and Birgit Kaufmann.

