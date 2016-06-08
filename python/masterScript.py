# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:23:11 2016

@author: nmvenuti
"""

import gc, sys, os
os.chdir('./github/nmvenuti/DSI_Religion/')
import os.path
import pandas as pd
import numpy as np
import nltk
import time
import pickle
sys.path.append('./python/')
import semanticDensity as sd
import syntacticParsing as sp
import sentimentAnalysis as sa
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
targetWordCount=10
cocoWindow=6
svdInt=50
cvWindow=6
simCount=1000
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
masterOutput=[]
for groupId in subgroupList:
#    print('testing')
#groupId=subgroupList[0]
    #Create sub directory
    #folderPath=runDirectory+'/'+groupId[0]+'/'+groupId[1]
    #os.makedirs(folderPath)
    
    #Get list of subfiles
    subFileList=[x[1] for x in fileList if x[0]==groupId[0] and x[2]==groupId[1]]
    
    tokenList = sd.tokenize(subFileList)
    
    ########################
    ###Sentiment Analysis###
    ########################
    sentimentList=sa.sentimentLookup(tokenList)
    #pd.DataFrame(sentimentList,columns=['perPos','perNeg','perPosDoc','perNegDoc']).to_csv(folderPath+'/sentiment.csv')
    
    ########################################
    ###POS Tagging and Judgement Analysis###
    ########################################
    
    judgementList=[sp.judgements(sp.readText(fileName)) for fileName in subFileList]
    judgementAvg=list(np.mean(np.array(judgementList),axis=0))
    #pd.DataFrame(judgementList,columns=['judgementCount','judgementFrac']).mean().to_csv(folderPath+'/judgements.csv')
    
    txtString=' '.join([sp.readText(fileName) for fileName in subFileList])
    wordList=sp.targetWords(txtString,targetWordCount)
    
    #######################            
    ###Semantic analysis###
    #######################
    
    #Get word coCo
    CoCo, TF, docTF = sd.coOccurence(tokenList,cocoWindow)
    
    #Get DSM
    DSM=sd.DSM(CoCo,svdInt)
    #pickle.dump(DSM,open(folderPath+'/testDSM.pickle', 'wb') )
    
    
    #Get context vectors
    #Bring in wordlist
    
    wordList=[stemmer.stem(word) for word in wordList]
    CVDict=sd.contextVectors(tokenList, DSM, wordList, cvWindow)
    
    #Run cosine sim
    cosineSimilarity=sd.averageCosine(CVDict,subFileList,wordList,simCount)
    avgSD=np.mean([x[1] for x in cosineSimilarity])
    #pd.DataFrame(cosineSimilarity).to_csv(folderPath+'/contextVectors.csv')
    
    #Append outputs to masterOutput
    masterOutput.append(['_'.join(groupId)]+sentimentList+judgementAvg+[avgSD])

#Create output file
outputDF=pd.DataFrame(masterOutput,columns=['groupId','perPos','perNeg','perPosDoc','perNegDoc','judgementCount','judgementFrac','avgSD'])
outputDF.to_csv(runDirectory+'/masterOutput.csv')
timeOut=time.time()

estimatedRunTime=(timeOut-timeIn)

