
import numpy as np

class SimpleFactorModel:

    def __init__(self,data):
        self.data=data

    def estimate(self,tickers):
        r=self.data.get_returns(tickers)
        market=r.mean(axis=1)
        betas={}
        for t in tickers:
            betas[t]=r[t].cov(market)/market.var()
        return betas
