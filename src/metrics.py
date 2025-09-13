import numpy as np
import pandas as pd

def compute_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252/len(returns)) - 1
    sharpe = (returns.mean()*252) / vol if vol > 0 else 0
    sortino = (returns.mean()*252) / (np.sqrt((returns[returns<0]**2).mean())*np.sqrt(252))
    drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    calmar = cagr / abs(drawdown) if drawdown < 0 else np.inf
    return pd.Series({"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar, "MaxDD": drawdown})
