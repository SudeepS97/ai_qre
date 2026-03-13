
import numpy as np
import cvxpy as cp

class PortfolioOptimizer:

    def __init__(self, cov, config):

        self.cov=cov
        self.config=config

    def solve(self, alphas, current=None):

        tickers=list(alphas.keys())
        alpha_vec=np.array([alphas[t] for t in tickers])

        cov=self.cov.compute(tickers)

        n=len(tickers)
        w=cp.Variable(n)

        if current:
            curr=np.array([current.get(t,0) for t in tickers])
        else:
            curr=np.zeros(n)

        risk=cp.quad_form(w,cov)
        turnover=cp.norm1(w-curr)

        obj=cp.Maximize(
            alpha_vec@w
            -0.5*risk
            -self.config.turnover_penalty*turnover
        )

        cons=[]
        cons.append(cp.sum(w)==self.config.net_target)
        cons.append(cp.norm1(w)<=self.config.gross_limit)
        cons.append(w<=self.config.max_position)
        cons.append(w>=-self.config.max_position)

        prob=cp.Problem(obj,cons)
        prob.solve()

        return {t:w.value[i] for i,t in enumerate(tickers)}
