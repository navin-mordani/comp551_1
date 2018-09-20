import pandas as pd
import numpy as np
from datetime import datetime
import pandas as pd
import sys


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
#nf.to_csv('stackedData.csv', index=False)
#nf = pd.read_csv("./data/stackedData.csv", low_memory=False)

users = nf.ix[:,0:1]
users = users.drop_duplicates()

col = ['PARTICIPANT ID', 'GENDER', 'TOTAL_EVENTS', 'THERE15', 'THERE14', 'THERE13', 'THERE12', 'AVGTIME']

outFrame = pd.DataFrame(columns=col, dtype=np.int16)
timeSeries = nf[(nf['EVENT DATE'] == '2015-09-20') & (nf['EVENT TYPE'] == 'Marathon')]

def timeToSeconds (time):
    if time == '-1':
        return 6*3600
    else:
        d=datetime.strptime( time, '%H:%M:%S')
        return d.hour*3600 + d.minute*60 + d.second

for i in users.index:

    participantEvents = nf[nf['PARTICIPANT ID'] == users.ix[i]['PARTICIPANT ID']]
    total = len(participantEvents)
    was2015 = len(participantEvents[(participantEvents['EVENT DATE'] == '2015-09-20') & (participantEvents['EVENT TYPE'] == 'Marathon') ])
    if was2015 >= 2:
        was2015 = 1
    was2014 = len(participantEvents[(participantEvents['EVENT DATE'] == '2014-09-28') & (participantEvents['EVENT TYPE'] == 'Marathon') ])
    if was2014 >= 2:
        was2014 = 1
    was2013 = len(participantEvents[(participantEvents['EVENT DATE'] == '2013-09-22') & (participantEvents['EVENT TYPE'] == 'Marathon')])
    if was2013 >= 2:
        was2013 = 1
    was2012 = len(participantEvents[(participantEvents['EVENT DATE'] == '2012-09-23') & (participantEvents['EVENT TYPE'] == 'Marathon')])
    if was2012 >= 2:
        was2012 = 1
    try:
        if 'M' in participantEvents.iloc[0]['CATEGORY']:
            gender = -1
        else:
            gender = 1
    except:
        gender = 0
    times = participantEvents[ participantEvents['EVENT TYPE']=='Marathon']['TIME']
    times = times.apply(timeToSeconds)
    if len(times) == 0:
        times = 6*3600
    else:
        times = times.mean()

    outFrame.loc[users.ix[i]['PARTICIPANT ID']] = [users.ix[i]['PARTICIPANT ID'], gender, total, was2015, was2014, was2013, was2012, times]


outFrame[['PARTICIPANT ID', 'GENDER', 'TOTAL_EVENTS', 'THERE15', 'THERE14', 'THERE13', 'THERE12']] = outFrame[['PARTICIPANT ID', 'GENDER', 'TOTAL_EVENTS', 'THERE15', 'THERE14', 'THERE13', 'THERE12']].astype(int)
outFrame.to_csv('../data/Y1/dataForY1LogiAndNaive.csv', index=False)
print(outFrame)
