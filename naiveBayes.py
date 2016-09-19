import pandas as pd
import numpy as np
import math
import scipy.stats

data = pd.read_csv("out2.csv", low_memory=False)

y = data['THERE15']
data.drop('THERE15', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)

LAPLACE_SMOOTHING = 1

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
            catCount = len(col[(col == cat) & (y == k)]) + LAPLACE_SMOOTHING
            xSeries.loc[cat] = catCount/float(self.kCount[k]+ (LAPLACE_SMOOTHING*len(categories)))
        return xSeries

    def train (self, X, y, types):

        # Number of features in model
        self.m = len(X.columns)
        self.mNames = X.columns
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
        return out

    def pYProdPXY (self, k, x):
        prod = 1
        for f in self.dCols:
            prod *= self.theta[k][f][x[f]]
        for f in self.cCols:
            prod *= scipy.stats.norm(self.mu[k][f], self.sigma[k][f]).pdf(x[f])
        prod *= self.kProb[k]
        return prod

    def crossValidation (self, X, y, types, k):
        perm = np.random.permutation(len(X))
        X, y = pd.Series(np.array_split(X.iloc[perm], k)), pd.Series(np.array_split(y.iloc[perm], k))
        error = 0
        trueError = 0
        for i in range(k):
            mask = [True]*k
            mask[i] = False
            trainDataX, trainDataY = pd.concat(list(X[mask])), pd.concat(list(y[mask]))
            validateDataX, validateDataY = X[i], y[i]
            self.train(trainDataX, trainDataY, types)
            validateResult = self.predict(validateDataX)
            errSum = 0
            for i in validateDataY.index:
                errSum += validateResult.ix[i][validateDataY[i]]
            e = (validateResult[1].round() == validateDataY).sum()/float(len(validateDataY))
            error += e
            te = errSum/float(len(validateDataY))
            trueError += te
        error = error/float(k)
        trueError = trueError/float(k)
        print(error)
        print(trueError)



lr = LogisticR()
lr.crossValidation(data, y, pd.Series(['d','c','d','d','d','c']), 5)

#lr.train(data[:2000], y, pd.Series(['d'data,'c','d','d','d','c']))
#out = lr.predict(data[2001:3000])
#res = out[1].round() == y[2001:3000]
#print(res.sum()/float(1000))
