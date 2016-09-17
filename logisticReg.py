import pandas as pd
import numpy as np
import math

data = pd.read_csv("out2.csv", low_memory=False)

y = data['THERE15']
data.drop('THERE15', axis=1, inplace=True)
data.drop('AVGTIME', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)

class LogisticR:
    def __init__(self):
        return

    def dErr (self, x, y):
        return x * (self.sigma(x.dot(self.w))-y)

    def sigma (self, val):
        return 1/(1 + math.exp(-val))

    def dot(self, a, b):
        return a.dot(b)

    def sumError (self, X, y):
        total = 0
        for i in range(self.n):
            total += self.dErr(X.loc[i], y.loc[i])
        return total/self.n #TODO: Taking the average prevents overflow error

    def train (self, X, y, alpha):
        X.insert(0, 'INTERCEPT', 1)
        # Number of features in model
        self.m = len(X.columns)
        # Size of dataset
        self.n = len(X.index)
        # Weights vector
        self.w = pd.Series([0] * self.m, index=X.columns)

        for i in range(300):
            #print("Iteration: " + str(i))
            es = self.sumError(X, y)
            es = es.multiply(alpha)
            self.w = self.w - es

    def predict (self, X):
        X.insert(0, 'INTERCEPT', 1)
        return X.dot(self.w).apply(self.sigma)


lr = LogisticR()
lr.train(data[:2000], y, 0.15)
pre = lr.predict(data[2001:3000])
res = pre.round() == y[2001:3000]
print(res.sum()/float(1000))
