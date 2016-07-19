# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:18:57 2016

@author: nmvenuti
Lingual (lingustic singal) Class

Consolidation of all code used in lingustic analysis in single class to 
improve processing speeds and make analysis easier.

"""
#Import packages
import nltk
import nltk.data
from nltk.tag.perceptron import PerceptronTagger
import string
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
import random
import math
import igraph
import scipy.spatial.distance as ssd


###################################
######Set up inital parameters#####
###################################

#Set function parameters for package
tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
sentTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Get sentimentWord dict and remove duplicates. Store in lists
posFilePath='./refData/positive-words.txt'
negFilePath='./refData/negative-words.txt'
posWords=list(set(unicode(open(posFilePath).read(), "utf-8", errors="ignore")))
negWords=list(set(unicode(open(negFilePath).read(), "utf-8", errors="ignore")))

stemmer = nltk.stem.snowball.EnglishStemmer()
punctuation = set(string.punctuation)
stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

tagger = PerceptronTagger()



#Part of speech lists
nounList=['NN','NNS','NNP','NNPS']
adjList=['JJ','JJR','JJS']
toBeList=["is","was","am","are","were","been","be","being"]
tagFilterList=['JJ','JJR','JJS','RB','RBR','RBS','WRB']


#Define generic packages
def get_cosine(vec1,vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator/denominator)


#Define main class
class lingualObject(object):
    '''Class to extract lingusitic singals
    
    Paramameters
    ============
    filelist: list
        List of files used
    cocoWindow: int, default 6    
    cvWindow: int, default 6
    netAngle: float, default 30
    useStem: boolean
        True/False if stemming is used
    useStopwords: boolean
        True/False if stopwords are kept in vocabulary
        
    Methods
    =======
    Preprocessing:
        creates tokens and pos-tagged documents
    Keyword identiication:
        identifies keywords
    Co-Occurences:
        develops co-occurrence dictionary based on co-occurrence window
    DSM:
        reduces cocoDict based on dimensions
    Context Vectors:
        developes context vector dictionary based on keyword list
    Semantic Density Analysis:
        returns semantic density of keywords
    Network Analysis:
        returns network quantification of keywords
    Sentiment Analysis:
        returns sentiment ranking for documents
    Judgement Analysis:
        returns judgement quantification for documents
        
    '''
    #Initialize and perform preprocessing
    def __init__(self,fileList,useStem=True,useStopwords=True):
        #Define parameters
        self.fileList=fileList       
        self.useStem=useStem
        self.useStopwords=useStopwords
        
        ######################
        ###Get text objects###
        ######################
        
        #define rawText, tokens, and sentences
        self.rawText={}
        self.tokens={}
        self.sentences={}
        
        #extract raw text from each file in fileList and create tokens
        for fileName in fileList:   
            #Extract raw text and update for encoding issues            
            rawData=unicode(open(fileName).read(), "utf-8", errors="ignore")
            textList=nltk.word_tokenize(rawData)
            tokenList=[]
            for token in textList:
                try:
                    tokenList.append(str(token))
                except:
                    tokenList.append('**CODEC_ERROR**')
            
            #Create clean text string and save as rawText
            txtString=' '.join(tokenList)
            self.rawText[fileName]=txtString
            
            #Break text into sentences and extract as sentences
            sentList=list(sentTokenizer.tokenize(txtString))
            self.sentences[fileName]=sentList
            
            #Create tokens
            #Convert all text to lower case
            textList=[word.lower() for word in tokenList]
            
            #Remove punctuation
            textList=[word for word in textList if word not in punctuation]
            textList=["".join(c for c in word if c not in punctuation) for word in textList ]
            
            #convert digits into NUM
            textList=[re.sub("\d+", "NUM", word) for word in textList]  
            
            #Stem words if useStem True
            newStopWords=stopWords
            if useStem:
                textList=[stemmer.stem(word) for word in textList]
                newStopWords=[stemmer.stem(word) for word in stopWords]
            
            #Remove blanks
            textList=[word for word in textList if word!= ' ']
             
            #Remove stopwords if useStopwords ==False
            if not useStopwords: 
                newStopWords.append("")
                textList=[word for word in textList if word not in newStopWords]
            
            #Extract tokens
            self.tokens[fileName]=textList
            
    ########################
    ###Create cocoDict,TF###
    ########################
    def getCoco(self,k=6):
        self.cocoWindow=k
        self.cocoDict={}
        self.TF={}
        self.docTF={}
            
        #Loop through each file
        for fileName in self.fileList:
            for i in range(len(self.tokens[fileName])):
                #Adjust window to contain words k in front or k behind
                lowerBound=max(0,i-k)
                upperBound=min(len(self.tokens[fileName]),i+k)
                coCoList=self.tokens[fileName][lowerBound:i]+self.tokens[fileName][i+1:upperBound+1]
                window=self.tokens[fileName][i]
                
                #Add window to coCoDict if not present
                if window not in self.cocoDict.keys():
                    self.cocoDict[window]={}
                
                #Add words to coCoDict for window
                for word in coCoList:
                    try:
                        self.cocoDict[window][word]+=1
                    except KeyError:
                        self.cocoDict[window][word]=1
                        
                # Make TF Dictionary
                try:
                    self.TF[window]+=1
                except KeyError:
                    self.TF[window]=1
                    
                # Get TF Counts by File
                if fileName in self.docTF.keys():
                    if window in self.docTF[fileName].keys():
                        self.docTF[fileName][window]+=1
                    else:
                        self.docTF[fileName][window]=1
                else:
                    self.docTF[fileName]={}
                    self.docTF[fileName][window]=1
    
    ################
    ###Create DSM###
    ################
    def getDSM(self,k=50):
        #Define DSM
        self.DSM={}

        #Convert cocoDict to list of lists
        cocoDictList = []
        for key in self.cocoDict.keys():
            cocoDictList.append(self.cocoDict[key])
        
        #Vectorize cocoDictList
        v = DictVectorizer(sparse=True)
        dsm = v.fit_transform(cocoDictList)
        
        #perform truncated svd
        svd = TruncatedSVD(n_components=k, random_state=42)
        coCoSVD = svd.fit_transform(dsm)
        
        #create DSM dict
        for i in range(len(v.get_feature_names())):
            self.DSM[v.get_feature_names()[i]]={}
            for j in range(k):
                self.DSM[v.get_feature_names()[i]][j]=coCoSVD[i][j]

    
    #####################
    ###Set Up keywords###
    #####################
    def setKeywords(self,method='adjAdv',wordCount=10,startCount=0):
        #Need to set up other keyword methods
        #Get total text string
        txtString=''.join([x for x in self.rawText.values()])
        
        #Get total tag list
        tagList=tagger.tag(nltk.word_tokenize(txtString))
        
        #Define target dict
        targetDict={}
        
        #Loop through each tag in list and get count of tag and word
        for tag in tagList:
            if tag[1] in tagFilterList:
                word=str.lower(''.join([c for c in tag[0] if c not in string.punctuation]))
                #Filter out codecerrors
                if word != 'codecerror':
                    try:
                        targetDict[word]=targetDict[word]+1
                    except:
                        targetDict[word]=1
        
        #Create data frame with counts and sort
        targetDF=pd.DataFrame([[k,v] for k,v in targetDict.items()],columns=['word','count'])
        targetDF.sort(['count'],inplace=True,ascending=False)
        
        #Create keywords based on startCount and wordCount
        self.keywords=list(targetDF['word'])[startCount:wordCount+startCount]
    
    ######################################
    ###Get context vectors for keywords###
    ######################################
    def getContextVectors(self,k=6):
        
        #Define cv dict
        self.cvDict={}
        
        for fileName in self.fileList:
            self.cvDict[fileName]={}
            for i in range(len(self.tokens[fileName])):
                #Adjust window to contain words k in front or k behind
                lowerBound=max(0,i-k)
                upperBound=min(len(self.tokens[fileName]),i+k)
                cvList=self.tokens[fileName][lowerBound:i]+self.tokens[fileName][i+1:upperBound+1]
                window=self.tokens[fileName][i]
                
                #Check if window in wordlist
                if window in self.keywords:            
                    #Add entry for cvDict if window not yet present
                    if window not in self.cvDict[fileName].keys():
                        self.cvDict[fileName][window]={}
                    
                    #Create context vector            
                    contextVector={}
                    
                    for word in cvList:
                        for key in self.DSM[word].keys():                    
                            #Update context vector
                            try:
                                contextVector[key]=contextVector[key]+self.DSM[word][key]
                            except:
                                contextVector[key]=self.DSM[word][key]
                    
                    #Add context vector to cvDict
                    cvIndex=len(self.cvDict[fileName][window])+1
                    self.cvDict[fileName][window][cvIndex]=contextVector

    ##############################################
    ###Get average semantic density of keywords###
    ##############################################
    def getSD(self,sim=1000):
        subCV={}
        cosineResults=[]
        for fileName in self.fileList:
            for word in self.cvDict[fileName].keys():
                #Add word if not in keys
                if word not in subCV.keys():
                    subCV[word]={}
                for i in range(len(self.cvDict[fileName][word])):
                    subCV[word][len(subCV[word])+1]=self.cvDict[fileName][word][i+1]
        for searchWord in self.keywords:
            try: 
                if len(subCV[searchWord])>1:
                    consineSim=np.zeros(sim)
                    for i in range(sim):
                        x=random.randrange(0, len(subCV[searchWord]))
                        y=random.randrange(0, len(subCV[searchWord]))
                    
                        consineSim[i]=get_cosine(subCV[searchWord][x+1],subCV[searchWord][y+1])
                    approx_avg_cosine=np.average(consineSim)
                else:
                    approx_avg_cosine=-1
            except KeyError:
                approx_avg_cosine = np.NaN
            cosineResults.append([searchWord,approx_avg_cosine])
        return cosineResults
    
    ##########################################
    ###Get Judgement counts and percentages###
    ##########################################
    def judgements(self):
        #Define judgement list
        judgementList=[]
        
        #Loop through each document
        for fileName in self.fileList:
            judgementCount=0    
            #Loop through each sentence in file
            for sent in self.sentences[fileName]:
                tagList=tagger.tag(nltk.word_tokenize(sent))
                
                #Look for combination of noun-adj-to_be verb in order        
                nounFlag=False
                adjFlag=False
        
                for tag in tagList:
                    #Check if noun flag activated
                    if nounFlag:
                        #If noun flag activated check if adj flag activated
                        if adjFlag:
                            #If adjective flag activated check if word is to-be verb
                            if tag[0] in toBeList:
                                #If true, count as judgement, exit loop
                                judgementCount=judgementCount+1
                                break
                        #if adj flag not activated check if tag is adjective
                        else:
                            if tag[1] in adjList:
                                adjFlag=True
                    #if noun flag not activated check if tag is noun
                    else:
                        if tag[1] in nounList:
                            nounFlag=True
        
            judgementPercent=float(judgementCount)/len(self.sentences[fileName])
            judgementList.append([fileName,judgementCount,judgementPercent])
        
        #Return judgement numbers for each document
        return(judgementList)
    
    ###############    
    ###Sentiment###
    ###############
    def sentimentLookup(self):
        fileSentiment=[]
        #Get sentiment for each document
        for filename in self.tokens.keys():
            
            #initialize counters
            wordCount=0.0
            posCount=0.0
            negCount=0.0
            
            #Get counts
            for token in self.tokens[filename]:
                #Add to word count
                wordCount=wordCount+1        
                
                #Check if positive
                if token in posWords:
                    posCount=posCount+1
                
                #Check if negative
                if token in negWords:
                    negCount=negCount+1
            
            #Calculate percentages and append to list
            posPer=posCount/wordCount
            negPer=negCount/wordCount
            fileSentiment.append([posPer,negPer])
        
        #Calculate average sub-group level word sentiment percent
        wordSentiment=np.mean(np.array(fileSentiment),axis=0)
        
        #Calculate sub-group level doc sentiment percent
        posDocCount=float(len([x for x in fileSentiment if x[0]>x[1]]))
        posDocPer=posDocCount/len(fileSentiment)
        negDocPer=1-posDocPer
        output=[wordSentiment[0],wordSentiment[1],posDocPer,negDocPer]   
        
        return(output)
    
    ######################    
    ###Network Analysis###
    ######################
    def setNetwork(self,netAngle=30):
        #Get list of values in DSM
        dsmList=[x.values() for x in self.DSM.values()]
        
        #Calculate distances for each set of values in dsm
        cosineNP=ssd.cdist(dsmList,dsmList,metric='cosine')
        
        adj = cosineNP.copy()
        
        #Apply thresholds
        adj[np.abs(cosineNP) >= math.cos(math.radians(netAngle))] = 0 # Converting 3threshold to radians to a cosine value
        
        adj[np.abs(cosineNP) < math.cos(math.radians(netAngle))] = 1 # Converting threshold to radians to a cosine value
        
        adjList = pd.DataFrame(adj,columns=self.DSM.keys(),index=self.DSM.keys()).values.tolist()
        
        #Create network graph
        net = igraph.Graph.Adjacency(adjList, mode = "undirected")
        self.network=net
    
    def evc(self):
        
        #Get eigenvector centrality
        ev_centrality = igraph.Graph.evcent(self.network)
        
        #Get mean eigenvector centrality for words in target list
        meanEVC=np.mean([ev_centrality[i] for i in range(len(self.DSM.keys())) if self.DSM.keys()[i] in self.keywords])
        return(meanEVC)