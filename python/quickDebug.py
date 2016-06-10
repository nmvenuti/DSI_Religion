# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:40:25 2016

@author: nmvenuti
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:23:11 2016

@author: nmvenuti
"""
import time
start=time.time()
import sys, os
os.chdir('./github/nmvenuti/DSI_Religion/')
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



stemmer = nltk.stem.snowball.EnglishStemmer()

subFileList=['./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20131124.pdf.txt',
'./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20070624.pdf.txt',
'./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20110130.pdf.txt',
'./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20091108.pdf.txt',
'./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20110904.pdf.txt',
'./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20100328.pdf.txt',
'./data_dsicap/WBC/raw/WestboroBaptist_Sermon_20090510.pdf.txt'
]



groupSize=10
testSplit=0.1
targetWordCount=10
cocoWindow=6
svdInt=50
cvWindow=6
simCount=1000


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

############################
###Network Quantification###
############################
avgEVC=nq.getNetworkQuant(DSM,wordList)


