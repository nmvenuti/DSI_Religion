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
import os

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from sknn import mlp


startTime=time.time()

################################
#####Import and clean data######
################################

def addRank(signalDF):
    #Add in group ranking
    groupNameList=['WBC', 'PastorAnderson', 'NaumanKhan', 'DorothyDay', 'JohnPiper', 'Shepherd',
    'Rabbinic', 'Unitarian', 'MehrBaba']
    groupRankList=[1,2,3,4,4,4,6,7,8]
    
    groupRankDF=pd.DataFrame([[groupNameList[i],groupRankList[i]] for i in range(len(groupNameList))],columns=['groupName','rank'])
    
    signalDF['groupName']=signalDF['groupId'].map(lambda x: x.split('_')[0])
    
    signalDF=signalDF.merge(groupRankDF, on='groupName')
    return(signalDF)

#Define data filepath
rawPath='./github/nmvenuti/DSI_Religion/pythonOutput/run1/cleanedOutput'

#Get raw files
rawFileList=[]
for dirpath, dirnames, filenames in os.walk(rawPath):
    for filename in [f for f in filenames ]:
        if 'masterOutput.csv' in filename:
            rawFileList.append(os.path.join(dirpath, filename))

#Create list of lists with coco,cv,netAng,45,SC
cleanFileList=[[int(x) for x in y.replace('/','_').split('_') if x.isdigit()]+[y] for y in rawFileList]
#Convert to dataframe
fileDF=pd.DataFrame(cleanFileList,columns=['coco','cv','netAng','SC','filepath'])

#Check for incomplete runs and complete failed runs
#incomplete
fileDF['id']=fileDF['coco'].map(str)+'_'+fileDF['cv'].map(str)+'_'+fileDF['SC'].map(str)+'_'+fileDF['netAng'].map(str)
print(fileDF['id'].value_counts()[fileDF['id'].value_counts()<3])

#complete fails
neededCuts=[str(coco)+'_'+str(cv)+'_'+str(sw)+'_'+str(ang) for  coco in [2,3,4,5,6] for cv in [2,3,4,5,6] for sw in [0,10,20,30] for ang in [30,45,60,75]
if str(coco)+'_'+str(cv)+'_'+str(sw)+'_'+str(ang) not in set(fileDF['id']) ]

for cut in neededCuts:
    print(cut.split('_'))

resultsList=[]
failedFiles=[]
for iteration in range(len(cleanFileList)):
    #only pull files with startcount ==0
    if cleanFileList[iteration][3]==0:
        try:
            #Get data frame for each cut
            signalDF=pd.read_csv(cleanFileList[iteration][4])
    
            signalDF=addRank(signalDF)
            
            #Set up modeling parameters
            xList=['perPos','perNeg','judgementFrac','avgSD', 'avgEVC','perPosDoc','perNegDoc','judgementCount']
            yList=['rank']
            signalDF=signalDF[signalDF['files']>5]
            signalDF=signalDF.dropna()
            
            #Set up test train splits
            trainIndex=[x for x in signalDF['groupId'] if 'train' in x]
            testIndex=[x for x in signalDF['groupId'] if 'test' in x]
            
            signalTrainDF=signalDF[signalDF['groupId'].isin(trainIndex)]
            signalTestDF=signalDF[signalDF['groupId'].isin(testIndex)]
            
            yActual=signalTestDF['rank'].tolist()
            
                                    
            
            #Random Forest Regressor
            rfModel=RandomForestRegressor(n_estimators=10,max_depth=10,
                                          min_samples_split=1, max_features='auto',
                                          random_state=0,n_jobs=-1)
            
            rfModel.fit(signalTrainDF[xList],signalTrainDF[yList])
            
            #Predict New Data
            yPred=rfModel.predict(signalTestDF[xList])
            
            #Get accuracy
            rfAccuracy=float(len([i for i in range(len(yPred)) if abs(yActual[i]-yPred[i])<1])/float(len(yPred)))
            rfMAE=np.mean(np.abs(yActual-yPred))        
            #Perform same analysis with scaled data
            #Scale the data
            sc = StandardScaler()
            sc=sc.fit(signalTrainDF[xList])
            signalStdTrainDF= pd.DataFrame(sc.transform(signalTrainDF[xList]),columns=xList)
            signalStdTestDF = pd.DataFrame(sc.transform(signalTestDF[xList]),columns=xList)
            signalSVR=svm.SVR(C=3,epsilon=0.1,kernel='rbf',max_iter=100000)
            signalSVR.fit(signalStdTrainDF[xList],signalTrainDF[yList])
            
            #Predict New Data
            yPred=signalSVR.predict(signalStdTestDF[xList])
            
            #Get accuracy
            svmAccuracy=float(len([i for i in range(len(yPred)) if abs(yActual[i]-yPred[i])<1])/float(len(yPred)))
            svmMAE=np.mean(np.abs(yActual-yPred))
            
            resultsList.append(['_'.join(map(str,cleanFileList[iteration][0:4]))]+cleanFileList[iteration][0:4]+[rfAccuracy,rfMAE,svmAccuracy,svmMAE])
        except ValueError:
            print(cleanFileList[iteration][4]+' failed')
            failedFiles.append(cleanFileList[iteration][4])

resultsDF=pd.DataFrame(resultsList)
resultsDF.columns=['id','cocowindow','cvWindow','netAngle','startCount','rfAccuracy','rfMAE','svmAccuracy','svmMAE']
resultsDF.to_csv(rawPath+'summaryOutput-full.csv')
#Summarize data                                

        
        
#summaryDF=resultsDF.groupby(['id']).agg({'rfAccuracy':{'meanRF':'mean','minRF':'min','maxRF':'max'},
#'svmAccuracy':{'meanSVM':'mean','minSVM':'min','maxSVM':'max'}})
#summaryDF.to_csv(rawPath+'summaryOutput.csv')
#summaryDF.reset_index(inplace=True)
#summaryDF=pd.DataFrame(np.array(summaryDF))
#summaryDF.columns=['id','meanRF','minRF','maxRF','meanSVM','minRF','meanSVM']
#summaryDF.describe()
#summaryDF['cocoWindow']=summaryDF['id'].map(lambda x: int(x.split('_')[0]))
#summaryDF['cvWindow']=summaryDF['id'].map(lambda x: int(x.split('_')[1]))
#summaryDF['startCount']=summaryDF['id'].map(lambda x: int(x.split('_')[2]))
#summaryDF['netAngle']=summaryDF['id'].map(lambda x: int(x.split('_')[3]))
#summaryDF.drop('id',inplace=True,axis=1)



#Plot accuracy versus different parameters
ax=sns.pairplot(resultsDF.drop('id', axis=1))

