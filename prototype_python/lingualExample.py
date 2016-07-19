# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:12:17 2016

@author: nmvenuti
Example of lingual package
"""
import time
start=time.time()
import sys, os
import multiprocessing as mp

os.chdir('./github/nmvenuti/DSI_Religion/')
#from joblib import Parallel, delayed
#import multiprocessing as mp
import os.path
import pandas as pd
sys.path.append('./prototype_python/')
import lingual as la

#Extract randomized files
fileDF=pd.read_csv('./data_dsicap/test_train/fileSplit_0.csv')
fileList=fileDF.values.tolist()
fileList=[[fileList[i][1],fileList[i][2],fileList[i][3]] for i in range(len(fileList))]


#Get set of subgroups
subgroupList=[ list(y) for y in set((x[0],x[2]) for x in fileList) ]

#Select files for first set of values in subgroupList
testFiles=list(fileDF[(fileDF['group']==subgroupList[0][0])&(fileDF['subgroup']==subgroupList[0][1])]['filepath'])

startTime=time.time()
#Test lingual object
loTest=la.lingualObject(testFiles)

#Test get coco
loTest.getCoco()

#Test DSM
loTest.getDSM()

#Test keywords with default and judgements method
loTest.setKeywords()
loTest.keywords

loTest.setKeywords('judgement')
loTest.keywords

#Test context vectors
loTest.getContextVectors()

#Test get average semantic density
loTest.getSD()

#Test judgements
loTest.getJudgements()

#Test sentiment
loTest.sentimentLookup()

#Test network
loTest.setNetwork()

#Test evc
loTest.evc()


print(str(time.time()-startTime))


#define function for multiple filecuts
def fullRun(fileList):
    #Test lingual object
    loTest=la.lingualObject(fileList)
    
    #Test get coco
    loTest.getCoco()
    
    #Test DSM
    loTest.getDSM()
    
    #Test keywords
    loTest.setKeywords()
    
    #Test context vectors
    loTest.getContextVectors()
    
    #Test get average semantic density
    loTest.getSD()
    
    #Test judgements
    loTest.judgements()
    
    #Test sentiment
    loTest.sentimentLookup()
    
    #Test network
    loTest.setNetwork()
    
    #Test evc
    loTest.evc()
    
    return('complete')

paramList=[list(fileDF[(fileDF['group']==subgroupList[i][0])&(fileDF['subgroup']==subgroupList[i][1])]['filepath']) for i in range(10)]
#Time ten runs
startTime=time.time()
testOuptut=[fullRun(x) for x in paramList]
print(str(time.time()-startTime))
#99.7856340408


#time ten paralleized runs
startTime=time.time()
xPool=mp.Pool(processes=3)    
outputList=[xPool.apply_async(fullRun, args=(x,)) for x in paramList]
masterOutput=[p.get() for p in outputList] 
print(str(time.time()-startTime))