from __future__ import division
import pandas as pd
import numpy as np
import csv
import re
from datetime import datetime
import sys

def getTimeInSeconds (timeInHour):
    if timeInHour == '-1':
        return -1
    else:
        hour = int(timeInHour[0:2]);
        minute = int(timeInHour[3:5]);
        seconds = int(timeInHour[6:9]);
        return hour*3600 + minute*60 + seconds

def createEventTypeWithDistanceDictionary():
    with open('uniqueEventTypes.csv', mode='r') as infile:
        reader = csv.reader(infile)
        eventTypeWithDistanceDictionary = {rows[0]:rows[1] for rows in reader}
    return eventTypeWithDistanceDictionary

def getAgeAndGender(Category):
    pattern1 = re.compile("[(M,F)]\d\d-\d\d");
    if pattern1.match(Category):
        gender = Category[0:1];
        ageLowerBound = Category[1:3];
        ageUpperBound = Category[4:6];
        age = (int(ageLowerBound) + int(ageUpperBound))/2;
    else:
        pattern2 = re.compile("[(M,F)]\d\d+[(\+,\-)]")
        if pattern2.match(Category):
            gender = Category[0:1]
            age = Category[1:3]
        elif (Category == 'MALE' or Category == 'M-JUN' or Category == 'Male' or Category == 'NO AGE' or Category == 'No age' or Category == 'CLYDE' or Category == 'NULL' or Category == 'U0-0' or Category == ' '):
            gender = 'M'
            age = -1
        elif (Category == 'M-SEN II' or Category == 'M-SENIOR II'):
            gender = 'M'
            age = 65
        elif (Category == 'Female'):
            gender = 'F'
            age = -1
        elif (Category == 'F-SENIOR I' or Category == 'F-SEN I'):
            gender = 'F'
            age = 65
        elif Category == 'M 30-39':
            gender = 'M'
            age = 34.5
        elif (Category == 'MAITRE F30-39'):
            gender = 'F'
            age = 34.5
        elif (Category == 'M05'):
            gender = 'M'
            age = 5
        else:
            gender = 'M'
            age = -1
    genderAge = {'gender':gender, 'age':age}
    return genderAge

def getTimeOfCompletion(distance,actualTime):
    estimatedTime = 0
    if(actualTime == -1):
        estimatedTime = 6*3600
    elif(distance == '42.2' or distance == '42'):
        estimatedTime = actualTime
    elif(distance == '21.1' or distance == '21'):
        estimatedTime = 1.9755*actualTime + 637.3307
    elif(distance == '10'):
        estimatedTime = 4.3268 * actualTime + 807.3818
    elif(distance == '5'):
        estimatedTime = 9.0869 * actualTime + 641.3391
    elif(distance == '15'):
        estimatedTime = 2.8096 * actualTime + 800.1356
    elif(distance == '1'):
        estimatedTime = 46.8293 * actualTime + 1598.3947
    return estimatedTime;

eventTypeWithDistanceDictionary = createEventTypeWithDistanceDictionary()
result = pd.DataFrame(columns=('Participant Id', 'Gender','Age','TimeOfCompletion'))
i=0
for df in pd.read_csv('out.csv',sep=',', chunksize=1):
    try:
        eventType = df.ix[:,3:4].iloc[0]['EVENT TYPE']
        initialDistanceOfMarathon = eventTypeWithDistanceDictionary.get(eventType)
        if(initialDistanceOfMarathon == '42' or initialDistanceOfMarathon == '21' or initialDistanceOfMarathon =='42.2' or initialDistanceOfMarathon=='21.1'):
            participantId = df.ix[:,0:1].iloc[0]['PARTICIPANT ID']
            category = df.ix[:,5:6].iloc[0]['CATEGORY']
            initialTimeOfCompletion = df.ix[:,4:5].iloc[0]['TIME']
            valueOfAgeAndGender = getAgeAndGender(category)
            age = valueOfAgeAndGender.get('age')
            gender = valueOfAgeAndGender.get('gender')
            if(initialTimeOfCompletion == -1):
                timeInSeconds = getTimeInSeconds('-1')
            else:
                timeInSeconds = getTimeInSeconds(initialTimeOfCompletion)
            estimatedTimeOfCompletion = getTimeOfCompletion(initialDistanceOfMarathon,timeInSeconds)
            result.loc[i] = [participantId,gender,age,estimatedTimeOfCompletion]
            i = i+1
    except:
         print('Error particpant',participantId,category,initialTimeOfCompletion)
result.to_csv('featureExtractionForY2.csv', index=False)
