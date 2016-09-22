import pandas as pd
import numpy as np
import math

data = pd.read_csv("out2.csv", low_memory=False)

y = data['THERE15']
data['TOTAL_EVENTS'] = (data['TOTAL_EVENTS'] - data['TOTAL_EVENTS'].mean()) / (data['TOTAL_EVENTS'].max() - data['TOTAL_EVENTS'].min())
data.drop('THERE15', axis=1, inplace=True)
data.drop('AVGTIME', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)
data15 = data.copy()


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
            total += self.dErr(X.iloc[i], y.iloc[i])
        return total#/self.n #TODO: Taking the average prevents overflow error

    def train (self, X, y, alpha, errThres):

        # Number of features in model
        self.m = len(X.columns)
        # Size of dataset
        self.n = len(X.index)
        # Weights vector
        self.w = pd.Series([0] * self.m, index=X.columns)

        errorThresholdAttained = False
        while not errorThresholdAttained:
            #print("Iteration: " + str(i))
            es = self.sumError(X, y)
            es = es.multiply(alpha)
            if (es.abs() <= errThres).all():
                errorThresholdAttained = True
            self.w = self.w - es
            print(es)

    def predict (self, X):
        res= X.dot(self.w).apply(self.sigma)
        print(res)
        print((res >= 0.5 ).sum())
        return res

    def crossValidation (self, X, y, alpha, errThres, k):
        perm = np.random.permutation(len(X))
        X.insert(0, 'INTERCEPT', 1)
        X, y = pd.Series(np.array_split(X.iloc[perm], k)), pd.Series(np.array_split(y.iloc[perm], k))
        error = 0
        trueError = 0
        for i in range(k):
            mask = [True]*k
            mask[i] = False
            trainDataX, trainDataY = pd.concat(list(X[mask])), pd.concat(list(y[mask]))
            validateDataX, validateDataY = X[i], y[i]
            self.train(trainDataX, trainDataY, alpha, errThres)
            validateResult = self.predict(validateDataX)
            trainResult = self.predict(trainDataX)
            validateResult = (validateResult >= 0.5 ) == validateDataY
            trainResult = (trainResult >= 0.5) == trainDataY
            traine = trainResult.sum()/float(len(trainDataY))
            vale = validateResult.sum()/float(len(validateDataY))
            return traine, vale


lr = LogisticR()
data.insert(0, 'INTERCEPT', 1)
data15.insert(0, 'INTERCEPT', 1)
lr.train(data, y, 0.001, 0.003)
t3 = data['THERE13']
t4 = data['THERE14']
data15.drop('THERE12', axis=1, inplace=True)
data15.drop('THERE13', axis=1, inplace=True)
data15.drop('THERE14', axis=1, inplace=True)
data15.insert(0, 'THERE12', t3)
data15.insert(0, 'THERE13', t4)
data15.insert(0, 'THERE14', y)
lr.predict(data15)
#lr.crossValidation(data[:1000], y[:1000], 0.01, 0.003, 5)
'''
trainea = []
valea = []
for i in range(5, 15, 1):
    traine, vale = lr.crossValidation(data[:1000], y[:1000], i/float(100), 0.004, 5)
    trainea.append(traine)
    valea.append(vale)

import matplotlib.pyplot as plt

x = range(1, 10, 1)
# red dashes, blue squares and green triangles
plt.plot(x, trainea, 'r--', x, valea , 'b--')
plt.show()
'''
#pre = lr.predict(data[2001:3000])
#res = pre.round() == y[2001:3000]
#print(res.sum()/float(1000))
