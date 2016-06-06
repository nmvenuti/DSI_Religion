# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:11:13 2016

@author: nmvenuti
Judgements Syntactic Parsing Development
"""

#Import packages
import nltk
import nltk.data
import string
import pandas as pd

#Set up inital parameters
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nounList=['NN','NNS','NNP','NNPS']
adjList=['JJ','JJR','JJS']
toBeList=['be', 'am', 'is', 'are', 'was', 'were', 'been', 'has', 
'have', 'had', 'do', 'did', 'does', 'can', 'could', 'shall', 
'should', 'will', 'would', 'may', 'might', 'must']
tagFilterList=['JJ','JJR','JJS','RB','RBR','RBS','WRB']


#Define packages

def readText(filepath):
    rawText=unicode(open(filepath).read(), "utf-8", errors="ignore")
    tokens=nltk.word_tokenize(rawText)
    tokenList=[]
    for token in tokens:
        try:
            tokenList.append(str(token))
        except:
            tokenList.append('**CODEC_ERROR**')
    return(' '.join(tokenList))

def judgements(txtString):
    judgementCount=0
    sentList=list(tokenizer.tokenize(txtString))
    for sent in sentList:
        tagList=nltk.pos_tag(nltk.word_tokenize(sent))
        #Check for noun
        if len([x for x in tagList if x[1] in nounList])>0:
            #Check for adj
            if len([x for x in tagList if x[1] in adjList])>0:
                #Check for to-be verb
                if len([x for x in tagList if str.lower(x[0]) in toBeList])>0:
                    judgementCount=judgementCount+1
    judgementPercent=float(judgementCount)/len(sentList)
    #Return metrics
    return([judgementCount,judgementPercent])

def targetWords(txtString,wordCount):
    tagList=nltk.pos_tag(nltk.word_tokenize(txtString))
    targetDict={}
    for tag in tagList:
        if tag[1] in tagFilterList:
            word=str.lower(''.join([c for c in tag[0] if c not in string.punctuation]))
            try:
                targetDict[word]=targetDict[word]+1
            except:
                targetDict[word]=1
    targetDF=pd.DataFrame([[k,v] for k,v in targetDict.items()],columns=['word','count'])
    targetDF.sort(['count'],inplace=True,ascending=False)
    sortedTargetList=list(targetDF['word'])[:wordCount]
    return(sortedTargetList)