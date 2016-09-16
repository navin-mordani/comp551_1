import pandas as pd
import numpy as np
import math

data = pd.read_csv("out2.csv", low_memory=False)

y = data['THERE15']
data.drop('THERE15', axis=1, inplace=True)
data.drop('AVGTIME', axis=1, inplace=True)
data.drop('id', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)

class LogisticR:
    def __init__(self):
        return

    def dErr (self, x, y):
        return x * (y - self.sigma(self.dot(x, self.w)))

    def sigma (self, val):
        return 1/(1 + math.exp(-val))

    def dot(self, a, b):
        return a.dot(b)

    def sumError (self, X, y):
        total = 0
        for i in range(self.n-1):
            total += self.dErr(X.loc[i], y.loc[i])
        return total

    def train (self, X, y, alpha):
        # Number of features in model
        self.m = len(X.columns)
        # Size of dataset
        self.n = len(X.index)
        # Weights vector
        self.w = pd.Series([0.001] * self.m, index=X.columns)

        for i in range(100):
            print("Iteration: " + str(i))
            es = self.sumError(X, y)
            es = es.multiply(alpha)
            self.w = self.w + es
            print(self.w)

    def predict (self, X):
        return X.dot(self.w).apply(self.sigma)


lr = LogisticR()
lr.train(data[:200], y, 0.0001)
pre = lr.predict(data[1001:1200])
print(pre)
