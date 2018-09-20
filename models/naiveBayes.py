import pandas as pd
import numpy as np
import math
import scipy.stats
import argparse
LAPLACE_SMOOTHING = 1

class NaiveB:
    def __init__(self):
        return

    def calculateProbabilities(self):
        self.possibleOutcomes = self.y.drop_duplicates()
        # Discrete data
        self.kProbabilities = pd.Series()
        # Continuous data
        self.kAverages = pd.Series()
        self.kVariances = pd.Series()
        # List of counts for outcomes k
        self.kCount = pd.Series()
        # List of probabilities for outcomes k
        self.kProb = pd.Series()
        for k in self.possibleOutcomes:
            self.kCount.loc[k] = len(self.y[self.y == k])
            self.kProb.loc[k] = self.kCount[k]/float(len(self.y))
            self.kProbabilities.loc[k] = self.getProbabilitiesForK(k)
            self.kAverages.loc[k], self.kVariances.loc[k] = self.getAveragesVariancesForK(k)

    def getProbabilitiesForK(self, k):
        featureProbabilities = pd.Series()
        for col in self.dCols: # For every discrete feature
            featureProbabilities.loc[col] = self.getProbabilitiesForFeature(col, k)
        return featureProbabilities

    def getProbabilitiesForFeature(self, feat, k):
        col = self.X[feat]
        categories = col.drop_duplicates() # Possible values for the feature
        categoryProbabilites = pd.Series()
        for cat in categories:
            catCount = len(col[(col == cat) & (self.y == k)]) + LAPLACE_SMOOTHING # Count of occurences of a category in the feature
            categoryProbabilites.loc[cat] = catCount/float(self.kCount[k] + (LAPLACE_SMOOTHING*len(categories)))
        return categoryProbabilites

    def getAveragesVariancesForK(self, k):
        featureAverages = pd.Series()
        featureVariances = pd.Series()
        for col in self.cCols: # For every Continuous feature
            featureAverages.loc[col], featureVariances.loc[col] = self.getAverageVarianceForFeature(col, k)
        return featureAverages, featureVariances

    def getAverageVarianceForFeature(self, col, k):
        column = self.X[col]
        average = column[self.y == k].mean()
        if math.isnan(average):
            average = 0
        variance = column[self.y == k].var()
        if math.isnan(variance):
            variance = 1
        return average, variance

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

        self.calculateProbabilities()

    def predict (self, X):

        out = pd.DataFrame()
        for x in X.index:
            denom = 0
            for k in self.possibleOutcomes:
                denom += self.pYProdPXY(k,X.ix[x])
            for k in self.possibleOutcomes:
                out.loc[x,k] = self.pYProdPXY(k,X.ix[x])/float(denom)
        return out

    def pYProdPXY (self, k, x):
        prod = 1
        for f in self.dCols:
            prod *= self.kProbabilities[k][f][x[f]]
        for f in self.cCols:
            prod *= scipy.stats.norm(self.kAverages[k][f], self.kVariances[k][f]).pdf(x[f])
        prod *= self.kProb[k]
        return prod

    def calculateError(self, result, Y):
        errSum = 0
        for i in Y.index:
            errSum += result.ix[i][Y[i]]
        error = (result[1].round() == Y).sum()/float(len(Y))
        trueError = errSum/float(len(Y))
        return error, trueError

    def crossValidation (self, X, y, types, k):
        perm = np.random.permutation(len(X))
        X, y = pd.Series(np.array_split(X.iloc[perm], k)), pd.Series(np.array_split(y.iloc[perm], k))
        trainError, trainTrueError = 0, 0
        validateError, validateTrueError = 0, 0
        for i in range(k):
            mask = [True]*k
            mask[i] = False
            trainDataX, trainDataY = pd.concat(list(X[mask])), pd.concat(list(y[mask]))
            validateDataX, validateDataY = X[i], y[i]
            self.train(trainDataX, trainDataY, types)
            validateResult = self.predict(validateDataX)
            trainResult = self.predict(trainDataX)

            trainE, trainTrueE = self.calculateError(trainResult, trainDataY)
            validateE, validateTrueE = self.calculateError(validateResult, validateDataY)
            #print(trainE)
            #print(validateE)
            trainError += trainE
            trainTrueError += trainTrueE
            validateError += validateE
            validateTrueError += validateTrueE

        print('--- Training ---')
        print('Error rate:')
        print(1 - trainError/float(k))
        print('True Error:')
        print(1 - trainTrueError/float(k))
        print('--- Validation ---')
        print('Error rate:')
        print(1 - validateError/float(k))
        print('True Error:')
        print(1 - validateTrueError/float(k))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--trainCSV',type=str, help='CSV file containing training data', required=True)
	parser.add_argument('-s','--submission',type=str, help='Submission File',required=True)
	args = parser.parse_args()
	

	data = pd.read_csv(args.trainCSV, low_memory=False)#("./data/Y1/dataForY1LogiAndNaive.csv", low_memory=False)
	y = data['THERE15']
	data.drop('THERE15', axis=1, inplace=True)
	data.drop('PARTICIPANT ID', axis=1, inplace=True)
	t3 = data['THERE13']
	t4 = data['THERE14']
	print('Running Naive Bayes model...')
	naiveModel = NaiveB()
	naiveModel.train(data, y, pd.Series(['d','c','d','d','d','c']))
	dataB15 = data.copy()
	dataB15.drop('THERE12', axis=1, inplace=True)
	dataB15.drop('THERE13', axis=1, inplace=True)
	dataB15.drop('THERE14', axis=1, inplace=True)
	dataB15.insert(2, 'THERE12', t3)
	dataB15.insert(2, 'THERE13', t4)
	dataB15.insert(2, 'THERE14', y)

	naiveModel.crossValidation(data, y, pd.Series(['d','c','d','d','d','c']),2)
	out = naiveModel.predict(dataB15)
	#print(out)
	res = out[1].round()
	res.to_csv(args.submission, index=True, header=False)#('bayes.csv', index=True, header=False)
	#print(res)
	#print(res.sum()/float(1000))
