import pandas as pd
import numpy as np
from datetime import datetime

nf = pd.read_csv("out.csv", low_memory=False)

eventTypes = nf.ix[:,3:4]
eventTypes = eventTypes.drop_duplicates()
print (eventTypes)
eventTypes.to_csv('uniqueEventTypes.csv', index=False)
"""
col = ['Event Types']
outFrame = pd.DataFrame(columns=col)
outFrame.to_csv('uniqueEventTypes.csv', index=False)"""
