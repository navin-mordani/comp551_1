import pickle
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plot3d
import pandas as pan
import math
import sys
from numpy import genfromtxt

def getMatrixFromFile(inputFileName):
    matrix = genfromtxt(inputFileName, dtype = 'float',delimiter=',',skip_header = 1)
    matrix = np.round(matrix,2)
    return matrix

def Err(X,y,w):
    prediction = np.dot(X,w)
    difference = np.subtract(y,prediction)
    divisor = (len(X[:,1]))
    return (np.dot(np.transpose(difference),difference))/divisor#Since (y-Xw)^2 = (y-Xw)'(y-Xw)

def GradientDescent(X,y,w):
    prediction = np.dot(X,w)
    difference = prediction - y;
    difference = np.transpose(difference)
    difference = np.dot(difference,X);
    difference = np.transpose(difference);
    return difference

def NormalEquation(X,y,w):
    xTranspose = np.transpose(X)
    xTransposeXInverse = np.linalg.inv(np.dot(xTranspose,X))
    xTransposeY = (np.dot(xTranspose,y))
    w = np.dot(xTransposeXInverse,xTransposeY)
    return w

def LinearRegression(X,y,w):
    alpha = 0.001
    listErr= np.zeros((10000,1),dtype = np.double)
    ErrorTrain = np.zeros((100,1),dtype = np.double)
    ErrorTest = np.zeros((100,1),dtype = np.double)
    #for j in range(100):
    for i in range(10000):
           w = w-2*((alpha*GradientDescent(X,y,w))/len(X[:,1]))
           Error = Err(X,y,w)
           listErr[i,0] = Error
    #    ErrorTrain[j,0] = Err(X[0:2100,:],y[0:2100],w)
    #    ErrorTest[j,0] = Err(X[2100:,:],y[2100:],w)
    fig = plt.figure()
    A = np.arange(10000).reshape(10000,1)
    A = np.arange(100).reshape(100,1)
    #plt.plot(A[:,0],ErrorTrain[:,0],label = 'Training')
    #plt.plot(A[:,0],ErrorTest[:,0], label = 'Test')
    #plt.xlabel('Iterations')
    #plt.ylabel('Loss Function')
    #plt.ylim(0,1)
    #plt.xlim(0,100)
    #plt.legend()
    #plt.show()
    difference = (np.dot(X,w) - y)
    difference = np.abs(difference)
    #for i in range(len(X[:,1])):
    #    print(difference[i])
    return w

def KFoldLinearRegression(X,y,w,K):
    numberOfRecords = len(X[:,1])
    sizeOfFold = numberOfRecords // K
    sumErrMSE = 0
    sumMeanErrAbsolute = 0
    for looper in range(K):
        #print(looper,"   ")
        if K-1 == looper:
            XTest = X[looper * sizeOfFold:,:]
            yTest = y[looper * sizeOfFold:,:]
        else:
            XTest = X[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
            yTest = y[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
        XTrain = np.delete(X,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        yTrain = np.delete(y,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        w = LinearRegression(XTrain,yTrain,w)
        sumErrMSE += Err(XTest,yTest,w)
        difference = (np.dot(XTest,w) - yTest)
        difference = np.abs(difference)
        sumMeanErrAbsolute += np.mean(difference)
        #print("K Fold",(Err(XTest,yTest,w)))
        w = np.zeros((len(X[1,:]),1),dtype = int)
    w = LinearRegression(X,y,w)
    return sumErrMSE / K,sumMeanErrAbsolute / K,w

def KFoldClosedMatrixForm(X,y,w,K):
    numberOfRecords = len(X[:,1])
    sizeOfFold = numberOfRecords // K
    sumErr = 0
    for looper in range(K):

        if K-1 == looper:
            XTest = X[looper * sizeOfFold:,:]
            yTest = y[looper * sizeOfFold:,:]
        else:
            XTest = X[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
            yTest = y[looper * sizeOfFold:(looper + 1) * sizeOfFold,:]
        XTrain = np.delete(X,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        yTrain = np.delete(y,np.s_[looper * sizeOfFold:(looper + 1) * sizeOfFold],0)
        w = NormalEquation(XTrain,yTrain,w)
        sumErr += Err(XTest,yTest,w)
        #print("K Fold",(Err(XTest,yTest,w)))
        w = np.zeros((len(X[1,:]),1),dtype = int)
    return sumErr / K


def Normalize(X):
    meanX = np.mean(X,axis = 0,dtype=np.float)
    standardDeviation = np.std(X,axis = 0,dtype = np.float)
    standardDeviation[standardDeviation == 0]=1
    X = (X-meanX)/standardDeviation
    return X

def correctTimeFormat(timeInSeconds):
    hours = int(timeInSeconds//3600)
    mins  = int((timeInSeconds%3600)//60)
    secs  = int(timeInSeconds-(hours*3600)-(mins*60))
    formattedTimeHours = "{0:0=2d}".format(hours)
    formattedTimeMins = "{0:0=2d}".format(mins)
    formattedTimeSecs = "{0:0=2d}".format(secs)
    formattedTime = str(formattedTimeHours)+':'+str(formattedTimeMins)+':'+str(formattedTimeSecs)
    return formattedTime


def correlation(X,y,wLinear):
    yPredicted = np.dot(X, wLinear)
    a = yPredicted - np.mean(yPredicted)
    b = y - np.mean(y)
    #print(np.sum((a) * (b)) / math.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
    plt.scatter(y,yPredicted)
    plt.ylim(0,10)
    plt.xlim(0,10)
    plt.xlabel('yActual')
    plt.ylabel('yPredicted')
    plt.grid()
    return np.sum((a) * (b)) / math.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

def main():
    #sys.stdout=open("test.txt","w")
    X = getMatrixFromFile('./data/Y2/featureMatrixFinal.csv')
    y = getMatrixFromFile('./data/Y2/vectorY.csv')
    #The next 2 lines take the traspose of y to generate y vector
    y = y[np.newaxis]
    y = np.transpose(y)

    #Convert the time in seconds to hours for matrix X and output vector Y
    X[:,0:3] = X[:,0:3]/3600
    y = y/3600


    plt.plot(X[:,0],)
    #Normalize matrix X

    X = Normalize(X)

    #Add bias vector column to all 1s in the Matrix X
    bias = np.ones((len(X[:,1]),1))
    X = np.append(X,bias,axis = 1)
    perm = np.random.permutation(len(X))
    X,y = X[perm],y[perm]
    w = np.zeros((len(X[1,:]),1),dtype = int)# initialize w = zero vector
    K = 10
    #KFoldClosedMatrixForm(X,y,w,K)
    MSELinear,meanAbsoluteLinear,wLinear = KFoldLinearRegression(X,y,w,K)
    mseLinear = MSELinear[0];
    #print mseLinear
    print('The Mean Square Error for Linear Regression Using K-Fold cross-validation is ',MSELinear)
    print('The Mean Absolute Error for Linear Regression Using K-Fold cross-validation is ',meanAbsoluteLinear)
    print('The Correlation between predicted output and the actual output is ', correlation(X,y,wLinear))
    print('The Hypothesis is ',wLinear[0],' * AVG-TIME-YEAR-3 + ',wLinear[1],' * AVG-TIME-YEAR-2 +'
    ,wLinear[2],' * AVG-TIME-YEAR-1 + ',wLinear[3],' * AGE + ',wLinear[4],' * GENDER + ',wLinear[5])
    fullX = getMatrixFromFile('./data/Y2/XMatrix.csv')
    allIds = fullX[:,len(fullX[1,:]) - 1]
    fullX = fullX[:,1:len(fullX[1,:]) - 1]
    fullX = Normalize(fullX)
    bias = np.ones((len(fullX[:,1]),1))
    fullX = np.append(fullX,bias,axis = 1)
    #print(fullX)
    yPredicted = np.dot(fullX,wLinear)
    finalPrediction = pan.Series();
    for i in range(len(fullX)):
        finalPrediction.loc[int(allIds[i])]=correctTimeFormat(yPredicted[i]*3600)
    return finalPrediction
