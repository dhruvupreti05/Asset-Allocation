import yfinance as yf
import numpy as np

def closing_prices(assets, start="2020-01-01"):
    closing_prices = yf.download(assets, start=start)["Close"]
    return  closing_prices.pct_change().dropna()

def get_covarience(daily_returns):
    covariance_matrix = daily_returns.cov().copy()
    np.fill_diagonal(covariance_matrix.values, 0)

    return covariance_matrix

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