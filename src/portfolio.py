import yfinance as yf

def closing_prices(assets, start="2020-01-01"):
    closing_prices = yf.download(assets, start=start)["Close"]
    return  closing_prices.pct_change().dropna()