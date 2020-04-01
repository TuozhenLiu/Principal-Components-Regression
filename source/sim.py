import numpy as np
from scipy import linalg
from scipy.stats import norm
from source.sigma_ma import sigma_ma


def sim(n, p, rho, mu):
    var = sigma_ma(p, rho)
    uper_mat = linalg.cholesky(var)
    x = norm.rvs(size=(n,p))
    x = np.dot(x, uper_mat)
    x = x + mu
    return x
