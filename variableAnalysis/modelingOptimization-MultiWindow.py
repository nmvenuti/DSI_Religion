# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:32:44 2016

@author: nmvenuti

Modeling grid search
"""

#Import packages
import pandas as pd
import numpy as np
import glob

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import time
#from sknn import mlp


startTime=time.time()

################################
#####Import and clean data######
################################

#Define data filepath
dataPath='./github/nmvenuti/DSI_Religion/variableAnalysis/First Run-Cuts/'


#Get data frame for each cut
signalDF2=pd.read_csv(dataPath+'cocowindow_2/run0/masterOutput.csv')
signalDF3=pd.read_csv(dataPath+'cocowindow_3/run0/masterOutput.csv')
signalDF4=pd.read_csv(dataPath+'cocowindow_4/run0/masterOutput.csv')
signalDF5=pd.read_csv(dataPath+'cocowindow_5/run0/masterOutput.csv')
signalDF6=pd.read_csv(dataPath+'cocowindow_6/run0/masterOutput.csv')


def addRank(signalDF):
    #Add in group ranking
    groupNameList=['WBC', 'PastorAnderson', 'NaumanKhan', 'DorothyDay', 'JohnPiper', 'Shepherd',
    'Rabbinic', 'Unitarian', 'MehrBaba']
    groupRankList=[1,2,3,4,4,4,6,7,8]
    
    groupRankDF=pd.DataFrame([[groupNameList[i],groupRankList[i]] for i in range(len(groupNameList))],columns=['groupName','rank'])
    
    signalDF['groupName']=signalDF['groupId'].map(lambda x: x.split('_')[0])
    
    signalDF=signalDF.merge(groupRankDF, on='groupName')
    return(signalDF)

signalDF2=addRank(signalDF2)
signalDF3=addRank(signalDF3)
signalDF4=addRank(signalDF4)
signalDF5=addRank(signalDF5)
signalDF6=addRank(signalDF6)


##################################################################
#####Hyperparameter optimzation for MLP RF and SVM Regression#####
##################################################################

def modelAnalysis(signalDF):
    xList=['perPos','perNeg','judgementFrac','avgSD', 'avgEVC']
    yList=['rank']
    signalDF=signalDF[signalDF['files']>5]
    signalDF=signalDF.dropna()
    
    #Set up test train splits
    trainIndex=[x for x in signalDF['groupId'] if 'train' in x]
    testIndex=[x for x in signalDF['groupId'] if 'test' in x]
    
    signalTrainDF=signalDF[signalDF['groupId'].isin(trainIndex)]
    signalTestDF=signalDF[signalDF['groupId'].isin(testIndex)]
    
    yActual=signalTestDF['rank'].tolist()
    
    #SVM
    svmAccuracy=0
    svmParam=[0]
    cList=[0.01,0.05,0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5]
    epsilonList=[0.01,0.05,0.1,0.5,1.0]
    kernelList=['linear','poly','rbf','sigmoid']
    degreeList=[0,1,2,3,4]
    coefList=[0,1,2,3]
    for C in cList:
        for epsilon in epsilonList: 
            for kernel in kernelList:
                for degree in degreeList:
                    for coef in coefList:
                        signalSVR=svm.SVR(C=C,epsilon=epsilon,kernel=kernel,degree=degree,coef0=coef,max_iter=100000)
                        signalSVR.fit(signalTrainDF[xList],signalTrainDF[yList])
    
                        #Predict New Data
                        yPred=signalSVR.predict(signalTestDF[xList])
                        
                        #Get accuracy
                        x=float(len([i for i in range(len(yPred)) if abs(yActual[i]-yPred[i])<1])/float(len(yPred)))
                        
                        #Add to parameter list
                        if x>svmAccuracy:
                            svmAccuracy=x
                            svmParam=[C,epsilon,kernel,degree,coef,svmAccuracy]
                            
    
    #Random Forest Regressor
    rfAccuracy=0
    estimatorList=[10,25,50,100,150,200]
    depthList=[5,10,15,20,25,30]
    featureList=['auto','sqrt','log2']
    splitList=[1,2,3]
    rfParam=[0]
    for estimator in estimatorList:
        for depth in depthList:
            for feature in featureList:
                for split in splitList:
                    rfModel=RandomForestRegressor(n_estimators=estimator,max_depth=depth,
                                                  min_samples_split=split, max_features=feature,
                                                  random_state=0,n_jobs=-1)
        
                    rfModel.fit(signalTrainDF[xList],signalTrainDF[yList])
        
                    #Predict New Data
                    yPred=rfModel.predict(signalTestDF[xList])
                    
                    #Get accuracy
                    x=float(len([i for i in range(len(yPred)) if abs(yActual[i]-yPred[i])<1])/float(len(yPred)))
                            
                    
                    #Add to parameter list
                    if x>rfAccuracy:
                        rfAccuracy=x
                        rfParam=[estimator,depth,split,feature,rfAccuracy]
    #Perform same analysis with scaled data
    #Scale the data
    sc = StandardScaler()
    sc=sc.fit(signalTrainDF[xList])
    signalStdTrainDF= pd.DataFrame(sc.transform(signalTrainDF[xList]),columns=xList)
    signalStdTestDF = pd.DataFrame(sc.transform(signalTestDF[xList]),columns=xList)
    svmAccuracy=0
    svmStdParam=[0]
    for C in cList:
        for epsilon in epsilonList: 
            for kernel in kernelList:
                for degree in degreeList:
                    for coef in coefList:
                        signalSVR=svm.SVR(C=C,epsilon=epsilon,kernel=kernel,degree=degree,coef0=coef,max_iter=100000)
                        signalSVR.fit(signalStdTrainDF[xList],signalTrainDF[yList])
                        
                        #Predict New Data
                        yPred=signalSVR.predict(signalStdTestDF[xList])
                        
                        #Get accuracy
                        x=float(len([i for i in range(len(yPred)) if abs(yActual[i]-yPred[i])<1])/float(len(yPred)))
                        
                        #Add to parameter list
                        if x>svmAccuracy:
                            svmAccuracy=x
                            svmStdParam=[C,epsilon,kernel,degree,coef,svmAccuracy]


    #STDRandom Forest Regressor
    rfAccuracy=0
    rfStdParam=[0]
    for estimator in estimatorList:
        for depth in depthList:
            for feature in featureList:
                for split in splitList:
                    rfModel=RandomForestRegressor(n_estimators=estimator,max_depth=depth,
                                                  min_samples_split=split, max_features=feature,
                                                  random_state=0,n_jobs=-1)
        
                    rfModel.fit(signalStdTrainDF[xList],signalTrainDF[yList])
        
                    #Predict New Data
                    yPred=rfModel.predict(signalStdTestDF[xList])
                    
                    #Get accuracy
                    x=float(len([i for i in range(len(yPred)) if abs(yActual[i]-yPred[i])<1])/float(len(yPred)))
                            
                    
                    #Add to parameter list
                    if x>rfAccuracy:
                        rfAccuracy=x
                        rfStdParam=[estimator,depth,split,feature,rfAccuracy]

    return(svmParam+rfParam+svmStdParam+rfStdParam)                                  
                                
                

accuracyList=[]
accuracyList.append([2]+modelAnalysis(signalDF2))
accuracyList.append([3]+modelAnalysis(signalDF3))
accuracyList.append([4]+modelAnalysis(signalDF4))
accuracyList.append([5]+modelAnalysis(signalDF5))
accuracyList.append([6]+modelAnalysis(signalDF6))
pd.DataFrame(accuracyList).to_csv('testOutput.csv')
