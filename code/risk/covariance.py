import numpy as np


class ShrinkageCovariance:

    def __init__(self, data, shrinkage=0.1):
        self.data = data
        self.shrinkage = shrinkage

    def compute(self, tickers):
        r = self.data.get_returns(tickers)
        sample = r.cov().values
        diag = np.diag(np.diag(sample))
        return (1 - self.shrinkage) * sample + self.shrinkage * diag
