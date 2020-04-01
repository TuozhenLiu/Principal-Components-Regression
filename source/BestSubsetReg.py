#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from scipy import linalg
from scipy.stats import norm
import random

def turnbits_rec(p):
    if (p == 1):
        return np.array([[True, False], [True, True]])
    else:
        tmp1 = np.c_[turnbits_rec(p - 1),
                     np.array([False] * (2**(p - 1))).reshape((2**(p - 1), 1))]
        tmp2 = np.c_[turnbits_rec(p - 1),
                     np.array([True] * (2**(p - 1))).reshape((2**(p - 1), 1))]
        return np.r_[tmp1, tmp2]

def turnbits_rec2(p):
    if (p == 1):
        return np.array([[True, False], [True, True]])
    else:
        tmp1 = np.c_[turnbits_rec2(p - 1),
                     np.array([False] * p).reshape(p, 1)]
        tmp2 = np.array([[True]*(p+1)])
        return np.r_[tmp1, tmp2]

def mse(xtx_t, xty_t, beta):
    return (np.sum(np.dot(xtx_t, beta) * beta) - 2 * np.sum(xty_t * beta))


def solve_sym(xtx, xty):
    L = linalg.cholesky(xtx)
    return linalg.lapack.dpotrs(L, xty)[0]

class BestSubsetReg(object):
    def __init__(self, x, y, inter=True, isCp=True, isAIC=True, isCV=True):
        self.__n, self.__p = x.shape
        self.__inter = inter
        if inter:
            self.__x = np.c_[np.ones((self.__n, 1)), x]
            self.__ind_var = turnbits_rec2(self.__p)
        else:
            self.__x = x
            self.__ind_var = turnbits_rec2(self.__p)[1:, 1:]
        self.__y = y
        self.__xTx = np.dot(self.__x.T, self.__x)
        self.__xTy = np.dot(self.__x.T, self.__y)
        self.__b = [
            solve_sym(self.__xTx[ind][:, ind], self.__xTy[ind])
            for ind in self.__ind_var
        ]
        self.__isCp = isCp
        self.__isAIC = isAIC
        self.__isCV = isCV

    def __Cp_AIC(self):
        rss = np.dot(self.__y, self.__y) - [
            np.sum(np.dot(self.__xTx[ind][:, ind], b_) * b_)
            for ind, b_ in zip(self.__ind_var, self.__b)
        ]
        d = np.sum(self.__ind_var, axis=1)
        if self.__isCp:
            self.Cp = rss + 2 * d * rss[-1] / (self.__n - self.__p - self.__inter)
        if self.__isAIC:
            self.AIC = self.__n * np.log(rss) + 2 * d

    def __cvreg(self):
        K = 10
        indexs = np.array_split(np.random.permutation(np.arange(0, self.__n)), K)

        def cvk(ind, index):
            txx = self.__xTx[ind][:, ind] - np.dot(
                (self.__x[index][:, ind]).T, self.__x[index][:, ind])
            txy = self.__xTy[ind] - np.dot(
                (self.__x[index][:, ind]).T, self.__y[index])
            tcoe = solve_sym(txx, txy)
            return np.sum(
                (self.__y[index] - np.dot(self.__x[index][:, ind], tcoe))**2)

        self.cverr = np.sum(np.array([[cvk(ind, index) for index in indexs]
                                      for ind in self.__ind_var]),
                            axis=1) / self.__n

    def output(self, isPrint=True):
        """
        If inter=True, first item is intercept, Otherwise it is X1. 
        If print=False, save results only and do not print.
        """
        if self.__isCp | self.__isAIC:
            self.__Cp_AIC()
            if self.__isCp:
                min_id = np.argmin(self.Cp)
                self.Cp = [self.__ind_var[min_id][0:], self.__b[min_id]]
                if isPrint:
                    print("Cp：\nVariable：", self.Cp[0])
                    print("Coefficient：", self.Cp[1])
            if self.__isAIC:
                min_id = np.argmin(self.AIC)
                self.AIC = [self.__ind_var[min_id][0:], self.__b[min_id]]
                if isPrint:
                    print("AIC：\nVariable：", self.AIC[0])
                    print("Coefficient：", self.AIC[1])
        if self.__isCV:
            self.__cvreg()
            min_id = np.argmin(self.cverr)
            self.cverr = [self.__ind_var[min_id][0:], self.__b[min_id]]
            if isPrint:
                print("Cross Validation：\nVariable：", self.cverr[0])
                print("Coefficient：", self.cverr[1])

