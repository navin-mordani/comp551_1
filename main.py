import pandas as pd
dataframe = pd.read_csv("Project1_data.csv", low_memory=False)
ids = dataframe['PARTICIPANT ID']
numCols = len(dataframe.columns)
colNames = dataframe.ix[:,0:6].columns.values

for i in range(1,(numCols/6)+14):
    dataframe.insert((i*6), 'PARTICIPANT ID'+str(i), ids)

numCols = len(dataframe.columns)

nf = None;
for i in range(0, numCols/6):
    sub = dataframe.ix[:,i*6:(i*6)+6]
    sub.columns = colNames
    if(nf is None):
        nf = sub
    else:
        nf = nf.append(sub)

nf = nf[pd.notnull(nf['EVENT DATE'])]
nf = nf.sort('PARTICIPANT ID')
nf.to_csv('out.csv', index=False)
