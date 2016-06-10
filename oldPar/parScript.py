# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:23:11 2016

@author: nmvenuti
"""
import pip
import sys, os
#os.chdir('./github/nmvenuti/DSI_Religion/')
import os.path
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
#from joblib import Parallel, delayed
sys.path.append('./python/')
import nltk
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')
import semanticDensity as sd
import syntacticParsing as sp
import sentimentAnalysis as sa



stemmer = nltk.stem.snowball.EnglishStemmer()

##########################
#####Define Functions#####
##########################
def textAnalysis(paramList):
    startTime=time.time()
    groupId=paramList[0]
    fileList=paramList[1]
    targetWordCount=paramList[2]
    cocoWindow=paramList[3]
    svdInt=paramList[4]
    cvWindow=paramList[5]
    simCount=paramList[6]
    #Get list of subfiles
    subFileList=[x[1] for x in fileList if x[0]==groupId[0] and x[2]==groupId[1]]
    
    tokenList = sd.tokenize(subFileList)
    
    ########################
    ###Sentiment Analysis###
    ########################
    sentimentList=sa.sentimentLookup(tokenList)
    
    ########################################
    ###POS Tagging and Judgement Analysis###
    ########################################
    
    judgementList=[sp.judgements(sp.readText(fileName)) for fileName in subFileList]
    judgementAvg=list(np.mean(np.array(judgementList),axis=0))
    
    txtString=' '.join([sp.readText(fileName) for fileName in subFileList])
    wordList=sp.targetWords(txtString,targetWordCount)
    
    #######################            
    ###Semantic analysis###
    #######################
    
    #Get word coCo
    CoCo, TF, docTF = sd.coOccurence(tokenList,cocoWindow)
    
    #Get DSM
    DSM=sd.DSM(CoCo,svdInt)
    
    #Get context vectors
    #Bring in wordlist
    
    wordList=[stemmer.stem(word) for word in wordList]
    CVDict=sd.contextVectors(tokenList, DSM, wordList, cvWindow)
    
    #Run cosine sim
    cosineSimilarity=sd.averageCosine(CVDict,subFileList,wordList,simCount)
    avgSD=np.mean([x[1] for x in cosineSimilarity])
    
    endTime=time.time()
    timeRun=endTime-startTime
    #Append outputs to masterOutput
    return(['_'.join(groupId)]+[len(subFileList),timeRun]+sentimentList+judgementAvg+[avgSD])

def runMaster(rawPath,groupList,groupSize,testSplit,targetWordCount,cocoWindow,svdInt,cvWindow,simCount,nCores):
    ###############################
    #####Raw File List Extract#####
    ###############################
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
    fileDF.to_csv(runDirectory+'/fileSplits.csv')
    
    
    ################################
    #####Perform group analysis#####
    ################################
    
    #Create paramList
    paramList=[[x,fileList,targetWordCount,cocoWindow,svdInt,cvWindow,simCount] for x in subgroupList]
    
    #Parallelize calulation
#    masterOutput=Parallel(n_jobs=nCores)(delayed(textAnalysis)(x) for x in paramList)  
    xPool=mp.Pool(processes=4)    
    outputList=[xPool.apply_async(textAnalysis, args=(x,)) for x in paramList]
    masterOutput=[p.get() for p in outputList]    
    #Create output file
    outputDF=pd.DataFrame(masterOutput,columns=['groupId','files','timeRun','perPos','perNeg','perPosDoc','perNegDoc','judgementCount','judgementFrac','avgSD'])
    outputDF.to_csv(runDirectory+'/masterOutput.csv')
    



#Set inital conditions and run
if __name__ == '__main__':
    rawPath = './data_dsicap/'
    groupList=['DorothyDay','JohnPiper','MehrBaba','NaumanKhan','PastorAnderson',
               'Rabbinic','Shepherd','Unitarian','WBC']
    groupSize=10
    testSplit=0.1
    targetWordCount=10
    cocoWindow=6
    svdInt=50
    cvWindow=6
    simCount=1000
    coreString=os.environ['SLURM_JOB_CPUS_PER_NODE']
    coreString=''.join([c if c.isdigit() else ' ' for c in coreString])
    nCores=reduce(lambda x, y: x*y,[int(x) for x in coreString.split() if x.isdigit()])
    
    
    startTime=time.time()
    runMaster(rawPath,groupList,groupSize,testSplit,targetWordCount,cocoWindow,svdInt,cvWindow,simCount,nCores)
    endTime=time.time()
    print((endTime-startTime)/60)#.82

