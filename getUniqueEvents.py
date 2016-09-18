import pandas as pd
import numpy as np
from datetime import datetime

nf = pd.read_csv("out.csv", low_memory=False)
col = ['PARTICIPANT ID', 'GENDER', 'TIME IF COMPLETING A MARATHON', 'AGE']

def timeToSeconds (time):
    if time == '-1':
        return 6*3600
    else:
        d=datetime.strptime( time, '%H:%M:%S')
        return d.hour*3600 + d.minute*60 + d.second

eventTypes = nf.ix[:,3:4]
eventTypes = eventTypes.drop_duplicates()
print (eventTypes)
eventTypes.to_csv('uniqueEventTypes.csv', index=False)
"""
col = ['Event Types']
outFrame = pd.DataFrame(columns=col)
outFrame.to_csv('uniqueEventTypes.csv', index=False)"""
