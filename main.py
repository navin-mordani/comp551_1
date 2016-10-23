from models import logisticReg as lr
from models import naiveBayes as nb
from models import linearReg as ir
import pandas as pd

# ------------------------------------------------------------------------------
# Prepare data for Logistic and Bayes
data = pd.read_csv("data/Y1/dataForY1LogiAndNaive.csv", low_memory=False)
data = data
output = pd.DataFrame(index=data.index)
y = data['THERE15']
y2 = y.copy()

#Regularize TOTAL_EVENTS
data['TOTAL_EVENTS'] = (data['TOTAL_EVENTS'] - data['TOTAL_EVENTS'].mean()) / (data['TOTAL_EVENTS'].max() - data['TOTAL_EVENTS'].min())

dataB = data.copy()

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
data15.insert(3, 'THERE14', y)
data15.insert(4, 'THERE13', t4)
data15.insert(5, 'THERE12', t3)

dataB.drop('THERE15', axis=1, inplace=True)
dataB.drop('PARTICIPANT ID', axis=1, inplace=True)


# ------------------------------------------------------------------------------
# Logistic Regresssion
print('Running Logisitic Regression model...')
print('This may take some time :( ')
logisticModel = lr.LogisticR()
logisticModel.train(data[:100], y, 1, 0.005)
res = logisticModel.predict(data15[:100])
res = res.round()
output.insert(0, 'Y1_LOGISTIC', res)
'''
# ------------------------------------------------------------------------------
# Naive Bayes
data = pd.read_csv("./data/Y1/dataForY1LogiAndNaive.csv", low_memory=False)
y = data['THERE15']
data.drop('THERE15', axis=1, inplace=True)
data.drop('PARTICIPANT ID', axis=1, inplace=True)

print('Running Naive Bayes model...')
naiveModel = nb.NaiveB()
naiveModel.train(data, y, pd.Series(['d','c','d','d','d','c']))
dataB15 = data.copy()
dataB15.drop('THERE12', axis=1, inplace=True)
dataB15.drop('THERE13', axis=1, inplace=True)
dataB15.drop('THERE14', axis=1, inplace=True)
dataB15.insert(2, 'THERE12', t3)
dataB15.insert(2, 'THERE13', t4)
dataB15.insert(2, 'THERE14', y)

out = naiveModel.predict(dataB15)
res = out[1].round()
output.insert(0, 'Y1_NAIVEBAYES', res)
'''
# ------------------------------------------------------------------------------
# Linear Regression
'''
res = ir.main()
output.insert(2, 'Y2_LINEAR', res)
'''
output.to_csv('predictions.csv', index=True, header=False)
