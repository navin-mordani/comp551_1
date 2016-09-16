import pandas as pd
import numpy as np
from datetime import datetime

nf = pd.read_csv("out.csv", low_memory=False)

users = nf.ix[:,0:1]
users = users.drop_duplicates()

col = ['PARTICIPANT ID', 'GENDER', 'TOTAL_EVENTS', 'THERE15', 'THERE14', 'THERE13', 'THERE12', 'AVGTIME']

outFrame = pd.DataFrame(columns=col, dtype=np.int16)

def timeToSeconds (time):
    if time == '-1':
        return 6*3600
    else:
        d=datetime.strptime( time, '%H:%M:%S')
        return d.hour*3600 + d.minute*60 + d.second

for i in users.index:
    try:
        participantEvents = nf[nf['PARTICIPANT ID'] == users.ix[i]['PARTICIPANT ID']]
        total = len(participantEvents)
        was2015 = len(participantEvents[(participantEvents['EVENT DATE'] == '2015-09-20') & (participantEvents['EVENT TYPE'] == 'Marathon')])
        was2014 = len(participantEvents[(participantEvents['EVENT DATE'] == '2014-09-28') & (participantEvents['EVENT TYPE'] == 'Marathon')])
        was2013 = len(participantEvents[(participantEvents['EVENT DATE'] == '2013-09-22') & (participantEvents['EVENT TYPE'] == 'Marathon')])
        was2012 = len(participantEvents[(participantEvents['EVENT DATE'] == '2012-09-23') & (participantEvents['EVENT TYPE'] == 'Marathon')])
        if 'M' in participantEvents.iloc[0]['CATEGORY']:
            gender = 0
        else:
            gender = 1
        times = participantEvents[ participantEvents['EVENT TYPE']=='Marathon']['TIME']
        times = times.apply(timeToSeconds)
        times = times.mean()

        outFrame.loc[users.ix[i]['PARTICIPANT ID']] = [users.ix[i]['PARTICIPANT ID'], gender, total, was2015, was2014, was2013, was2012, times]
    except:
        print('Error PARTICIPANT: ' + str(i))

outFrame[['PARTICIPANT ID', 'GENDER', 'TOTAL_EVENTS', 'THERE15', 'THERE14', 'THERE13', 'THERE12']] = outFrame[['PARTICIPANT ID', 'GENDER', 'TOTAL_EVENTS', 'THERE15', 'THERE14', 'THERE13', 'THERE12']].astype(int)
#nf = nf[nf['EVENT NAME'].str.contains('Marathon Oasis')]
#outFrame.to_csv('out2.csv', index=False)
print(outFrame)
