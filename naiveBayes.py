import pandas as pd
import numpy as np
import math
import scipy.stats

data = pd.read_csv("out2.csv", low_memory=False)

y = data['THERE15']
data.drop('THERE15', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)

class LogisticR:
    def __init__(self):
        return

    def getThetaProbs(self):
        self.outcomes = y.drop_duplicates()
        self.theta = pd.Series()
        self.mu = pd.Series()
        self.sigma = pd.Series()
        self.kCount = pd.Series()
        self.kProb = pd.Series()
        for k in self.outcomes:
            self.kCount.loc[k] = len(self.y[self.y == k])
            self.kProb.loc[k] = self.kCount[k]/float(len(self.y))
            self.theta.loc[k] = self.getThetaKSeries(k)
            self.mu.loc[k], self.sigma.loc[k] = self.getMuSigmaSeries(k)

    def getThetaKSeries(self, k):
        kSeries = pd.Series()
        for col in self.dCols:
            kSeries.loc[col] = self.getXSeries(col, k)
        return kSeries

    def getMuSigmaSeries(self, k):
        muSeries = pd.Series()
        sigmaSeries = pd.Series()
        for col in self.cCols:
            muSeries.loc[col], sigmaSeries.loc[col] = self.getAvgVarSeries(col, k)
        return muSeries, sigmaSeries

    def getAvgVarSeries(self, col, k):
        column = self.X[col]
        mu = column[self.y == k].mean()
        if math.isnan(mu):
            mu = 0
        sigma = column[self.y == k].var()
        if math.isnan(sigma):
            sigma = 1
        return mu, sigma

    def getXSeries(self, x, k):
        col = self.X[x]
        categories = col.drop_duplicates()
        xSeries = pd.Series()
        for cat in categories:
            catCount = len(col[(col == cat) & (y == k)])
            xSeries.loc[cat] = catCount/float(self.kCount[k])
        return xSeries

    def train (self, X, y, types):

        # Number of features in model
        self.m = len(X.columns)
        self.mNames = X.columns
        print(self.mNames)
        # Size of dataset
        self.n = len(X.index)

        # Continuous feature list
        self.cCols = X.columns[types == 'c']
        # Disrete feature list
        self.dCols = X.columns[types == 'd']

        self.X = X
        self.y = y

        self.getThetaProbs()

    def predict (self, X):

        out = pd.DataFrame()
        for x in X.index:
            denom = 0
            for k in self.outcomes:
                denom += self.pYProdPXY(k,X.ix[x])
            for k in self.outcomes:
                out.loc[x,k] = self.pYProdPXY(k,X.ix[x])/float(denom)
        print(out)
        return out

    def pYProdPXY (self, k, x):
        prod = 1
        for f in self.dCols:
            prod *= self.theta[k][f][x[f]]
        for f in self.cCols:
            prod *= scipy.stats.norm(self.mu[k][f], self.sigma[k][f]).pdf(x[f])
        prod *= self.kProb[k]
        return prod

lr = LogisticR()
lr.train(data[:2000], y, pd.Series(['d','c','d','d','d','c']))
out = lr.predict(data[2001:3000])
res = out[1].round() == y[2001:3000]
print(res.sum()/float(1000))
