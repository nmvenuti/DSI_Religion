# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:23:11 2016

@author: nmvenuti
"""
import time
start=time.time()
import sys, os
#os.chdir('./github/nmvenuti/DSI_Religion/')
import os.path
import numpy as np
import pandas as pd
from datetime import datetime
sys.path.append('./python/')
import nltk
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')
import semanticDensity as sd
import syntacticParsing as sp
import sentimentAnalysis as sa
import networkQuantification as nq

end=time.time()
sys.stdout = open("output.txt", "a")
print(str(datetime.now()))
print('finished loading packages after '+str(end-start)+' seconds')
sys.stdout.flush()


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
    print('finished sentiment'+'_'.join(groupId))
    sys.stdout.flush()
    
    ########################################
    ###POS Tagging and Judgement Analysis###
    ########################################
    
    judgementList=[sp.judgements(sp.readText(fileName)) for fileName in subFileList]
    judgementAvg=list(np.mean(np.array(judgementList),axis=0))
    print('finished judgement'+'_'.join(groupId))

    sys.stdout.flush()      
    txtString=' '.join([sp.readText(fileName) for fileName in subFileList])
    wordList=sp.targetWords(txtString,targetWordCount)

    print('finished target words'+'_'.join(groupId))
    sys.stdout.flush()    
    #######################            
    ###Semantic analysis###
    #######################
    
    #Get word coCo
    CoCo, TF, docTF = sd.coOccurence(tokenList,cocoWindow)
    print('finished coco'+'_'.join(groupId))
    sys.stdout.flush()
      
    #Get DSM
    DSM=sd.DSM(CoCo,svdInt)
    print('finished coco'+'_'.join(groupId))
    sys.stdout.flush() 
    
    #Get context vectors
    #Bring in wordlist
    
    wordList=[stemmer.stem(word) for word in wordList]
    CVDict=sd.contextVectors(tokenList, DSM, wordList, cvWindow)

    print('finished coco'+'_'.join(groupId))
    sys.stdout.flush()
     
    #Run cosine sim
    cosineSimilarity=sd.averageCosine(CVDict,subFileList,wordList,simCount)
    avgSD=np.mean([x[1] for x in cosineSimilarity])
    print('finished Semantic density'+'_'.join(groupId))
    sys.stdout.flush() 
    
    ############################
    ###Network Quantification###
    ############################
    avgEVC=nq.getNetworkQuant(DSM,wordList)
    print('finished network quant'+'_'.join(groupId))
    sys.stdout.flush() 
   
    endTime=time.time()
    timeRun=endTime-startTime
    print('finished running'+'_'.join(groupId)+' in '+str(end-start)+' seconds')
    sys.stdout.flush()
    #Append outputs to masterOutput
    return(['_'.join(groupId)]+[len(subFileList),timeRun]+sentimentList+judgementAvg+[avgSD]+[avgEVC])   

def runMaster(rawPath,groupList,crossValidate,groupSize,testSplit,targetWordCount,cocoWindow,svdInt,cvWindow,simCount):
    ###############################
    #####Raw File List Extract#####
    ###############################

    rawFileList=[]
    for groupId in groupList:
        for dirpath, dirnames, filenames in os.walk(rawPath+groupId+'/raw'):
            for filename in [f for f in filenames ]:
                if '.txt' in filename:
                    rawFileList.append([groupId,os.path.join(dirpath, filename)])

    #Make output directory
#    runDirectory='./pythonOutput/'+ time.strftime("%c")
#    runDirectory='./pythonOutput/cocowindow_'+str(cocoWindow)+time.strftime("%c").replace(' ','_')
    runDirectory='./pythonOutput/cocowindow_'+str(cocoWindow)+'_cvwindow_'+str(cvWindow)
    os.makedirs(runDirectory)
    end=time.time()
    #print('finished loading packages after '+str(end-start)+' seconds')
    sys.stdout.flush()
    
    
    #Perform analysis for each fold in cross validation
    for fold in range(crossValidate):                
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
        outputDirectory=runDirectory+'/run'+str(fold)
        os.makedirs(outputDirectory)
        
        ##print file splits to runDirectory
        fileDF.to_csv(outputDirectory+'/fileSplits.csv')

        end=time.time()
        #print('finished randomly creating subgroups '+str(end-start)+' seconds')
        sys.stdout.flush()        
        
        ################################
        #####Perform group analysis#####
        ################################
        
        #Create paramList
        paramList=[[x,fileList,targetWordCount,cocoWindow,svdInt,cvWindow,simCount] for x in subgroupList]
        
        #Run calculation
    	print('begin text analysis')
    	sys.stdout.flush()
        masterOutput=[textAnalysis(x) for x in paramList]
    	print('completed text analysis')
    	sys.stdout.flush()  
        #Create output file
        outputDF=pd.DataFrame(masterOutput,columns=['groupId','files','timeRun','perPos','perNeg','perPosDoc','perNegDoc','judgementCount','judgementFrac','avgSD','avgEVC'])
        outputDF.to_csv(outputDirectory+'/masterOutput.csv')

        
    



#Set inital conditions and run
if __name__ == '__main__':
    rawPath = './data_dsicap/'
    groupList=['DorothyDay','JohnPiper','MehrBaba','NaumanKhan','PastorAnderson',
               'Rabbinic','Shepherd','Unitarian','WBC']
    crossValidate=1
    groupSize=10
    testSplit=0.1
    targetWordCount=10
#    cocoWindow=6
    svdInt=50
#    cvWindow=6
    simCount=1000
    
    
    
    startTimeTotal=time.time()
    #Try hyper-parameter optimization on window range from 5 to 15
    for cocoWindow in range(4,5):
        for cvWindow in range(4,5):
            runMaster(rawPath,groupList,crossValidate,groupSize,testSplit,targetWordCount,cocoWindow,svdInt,cvWindow,simCount)
    endTimeTotal=time.time()
    print('finished entire run in :'+str((endTimeTotal-startTimeTotal)/60)+' minutes')
    sys.stdout.flush()

