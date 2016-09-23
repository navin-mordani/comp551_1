import pandas as pd
import numpy as np
import math

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
        count = 0
        while not errorThresholdAttained:
            count += 1
            #print("Iteration: " + str(i))
            es = self.sumError(X, y)
            #if count%10 == 0:
            #    print(es.divide(self.n))
            #    print(self.w)
            if es.abs().mean() <= errThres*self.n or count > 400:
                errorThresholdAttained = True
            es = es.multiply(alpha).divide(self.n)
            self.w = self.w - es


    def predict (self, X):
        res= X.dot(self.w).apply(self.sigma)
        return res

    def crossValidation (self, X, y, alpha, errThres, k):
        perm = np.random.permutation(len(X))
        X, y = pd.Series(np.array_split(X.iloc[perm], k)), pd.Series(np.array_split(y.iloc[perm], k))
        train = 0
        validate = 0
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
            train += traine
            validate += vale
        return train/float(k), validate/float(k)

'''
# Prepare data for Logistic and Bayes
data = pd.read_csv("./data/dataForY1LogiAndNaive.csv", low_memory=False)
output = pd.DataFrame(index=data.index)
y = data['THERE15']

#Regularize TOTAL_EVENTS
data['TOTAL_EVENTS'] = (data['TOTAL_EVENTS'] - data['TOTAL_EVENTS'].mean()) / (data['TOTAL_EVENTS'].max() - data['TOTAL_EVENTS'].min())

data.drop('THERE15', axis=1, inplace=True)
data.drop('AVGTIME', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)
data.insert(0, 'INTERCEPT', 1)
data15 = data.copy()
t3 = data['THERE13']
t4 = data['THERE14']
data15.drop('THERE12', axis=1, inplace=True)
data15.drop('THERE13', axis=1, inplace=True)
data15.drop('THERE14', axis=1, inplace=True)
data15.insert(0, 'THERE12', t3)
data15.insert(0, 'THERE13', t4)
data15.insert(0, 'THERE14', y)
lr = LogisticR()
'''
#-------------------------------------------------------------------------------
# Make prediction for 2016
'''
data.insert(0, 'INTERCEPT', 1)
data15 = data.copy()
t3 = data['THERE13']
t4 = data['THERE14']
data15.drop('THERE12', axis=1, inplace=True)
data15.drop('THERE13', axis=1, inplace=True)
data15.drop('THERE14', axis=1, inplace=True)
data15.insert(0, 'THERE12', t3)
data15.insert(0, 'THERE13', t4)
data15.insert(0, 'THERE14', y)

lr.train(data, y, 1, 0.005)

res = lr.predict(data15)
res = (res >= 0.5)
res.to_csv('predictionY1logistic.csv', index=True)
'''

#-------------------------------------------------------------------------------
# Cross validation
'''
data.insert(0, 'INTERCEPT', 1)
traine, vale = lr.crossValidation(data, y, 1, 0.005, 5)
print(traine)
print(vale)
'''

# ------------------------------------------------------------------------------
# Train on part of the data and save prediction to file
'''
#data.insert(0, 'INTERCEPT', 1)
lr.train(data[:2000], y[:2000], 1, 0.005)
res = lr.predict(data)
res.to_csv('ROC.csv', index=False)

'''

#-------------------------------------------------------------------------------
# Produce ROC curve
'''
roc = pd.read_csv("ROC.csv", low_memory=False, header=None)
roc = roc
roc.index = y.index
roc = roc[0]
#TPR
for i in [0.5]:
    TPR = ((roc >= 0.5) != y) & (y == True)
    print(TPR.sum())
    TPR = TPR.sum()/float((y == True).sum())
    print(TPR)

#1 - TNR
for i in [0.5]:
    TNR = ((roc >= 0.5 ) != y) & (y == False)
    print(TNR.sum())
    TNR = TNR.sum()/float((y == False).sum())
    print(1-TNR)
'''
#-------------------------------------------------------------------------------
# Test multiple values
'''
perm = np.random.permutation(len(data))
data, y = data.iloc[perm], y.iloc[perm]
x = [1, 0.1, 0.01]

for i in x:
    traine, vale = lr.crossValidation(data[:100], y[:100], i, 0.0004, 5)
    print(traine)
    print(vale)
'''
