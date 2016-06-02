# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:23:11 2016

@author: nmvenuti
"""

import gc, sys, os
os.chdir('../DSI_Religion/')
import os.path
import pandas as pd
import numpy as np
import nltk
import time
sys.path.append('./python/')
import semanticDensity as sd
stemmer = nltk.stem.snowball.EnglishStemmer()


###############################
#####Raw File List Extract#####
###############################

#Get file list for documents
rawPath = './data_dsicap/'
groupList=['DorothyDay','JohnPiper','MehrBaba','NaumanKhan','PastorAnderson',
           'Rabbinic','Shepherd','Unitarian','WBC']
rawFileList=[]
for groupId in groupList:
    for dirpath, dirnames, filenames in os.walk(rawPath+groupId+'/raw'):
        for filename in [f for f in filenames ]:
            if '.txt' in filename:
                rawFileList.append([groupId,os.path.join(dirpath, filename)])
                
###############################                
#####Set up random binning#####
###############################
                
#Loop through each group and create sub bins
groupSize=10
testSplit=0.1
fileList=[]
for groupId in groupList:
    subGroup=[x for x in rawFileList if groupId == x[0]]
    randomSample=list(np.random.choice(range(len(subGroup)),size=len(subGroup),replace=False))
    splitIndex=int((1-testSplit)*len(subGroup))
    groupId=['train'+ "%02d" %int(i/groupSize) if i<splitIndex else 'test'+ "%02d" %int((i-splitIndex)/groupSize) for i in randomSample]
    
    fileList=fileList+[[subGroup[i][0],subGroup[i][1],groupId[i]] for i in range(len(subGroup))]

fileDF=pd.DataFrame(fileList,columns=['group','filepath','subgroup'])


#Get set of subgroups
subgroupList=[ list(y) for y in set((x[0],x[2]) for x in fileList) ]

#Make output directory
runDirectory='./pythonOutput/'+ time.strftime("%c")
os.makedirs(runDirectory)

#Print file splits to runDirectory
fileDF.to_csv(runDirectory+'/fileSplits,csv')

timeIn=time.time()
################################
#####Perform group analysis#####
################################

for groupId in subgroupList:
    #Create sub directory
    folderPath=runDirectory+'/'+groupId[0]+'/'+groupId[1]
    os.makedirs(folderPath)
    
    #Get list of subfiles
    subFileList=[x[1] for x in fileList if x[0]==groupId[0] and x[2]==groupId[1]]
    
    tokenList = sd.tokenize(subFileList)
    
    #######################            
    ###Semantic analysis###
    #######################
    
    #Get word coCo
    CoCo, TF, docTF = sd.coOccurence(tokenList,6)
    
    #Get DSM
    DSM=sd.DSM(CoCo,50)
    
    
    #Get context vectors
    #Bring in wordlist
    wordList=['god','love','hate']
    wordList=[stemmer.stem(word) for word in wordList]
    CVDict=sd.contextVectors(tokenList, DSM, wordList, 6)
    
    #Run cosine sim
    cosineSimilarity=sd.averageCosine(CVDict,subFileList,wordList,1000)
    pd.DataFrame(cosineSimilarity).to_csv(folderPath+'/contextVectors.csv')

timeOut=time.time()

estimatedRunTime=(timeOut-timeIn)


