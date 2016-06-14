# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:17:50 2016

@author: nmvenuti

File is used to consolidate output reports from Rivanna 
and perform variable analysis on each output type
"""

#Import packages
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

################################
#####Import and clean data######
################################

#Define data filepath
dataPath='./github/nmvenuti/DSI_Religion/variableAnalysis/Third Run-Cuts/'


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

#Index([u'Unnamed: 0', u'groupId', u'files', u'timeRun', u'perPos', u'perNeg', u'perPosDoc', u'perNegDoc', 
#u'judgementCount', u'judgementFrac', u'avgSD', u'avgEVC', u'groupName', u'rank'], dtype='object')

##################################
#####Review Consolidated Data#####
##################################

def plotVariables(signalDF,idNumber):
    
    #Subset to groups of proper length
    signalDF=signalDF[signalDF['files']>5]    
    
    #Create box plots
    with PdfPages('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/variable importance results-window '+str(idNumber)+'.pdf') as pdf:
    
        
        ax = sns.boxplot(x='rank',y='judgementFrac',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Average Fraction of Judgements versus Rank')
        pdf.savefig()
        plt.close() 
    
        ax = sns.boxplot(x='rank',y='judgementCount',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Average Number of Judgements versus Rank') 
        pdf.savefig()
        plt.close() 
     
    
        ax = sns.boxplot(x='rank',y='avgEVC',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Eigenvector Centrality versus Rank')
        pdf.savefig()
        plt.close() 
        
        
        ax = sns.boxplot(x='rank',y='avgSD',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Context Vector Similarity versus Rank') 
        pdf.savefig()
        plt.close() 
        
        ax = sns.boxplot(x='rank',y='perPos',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Fraction of Positive Words versus Rank') 
        pdf.savefig()
        plt.close()     
        
        ax = sns.boxplot(x='rank',y='perNeg',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Fraction of Negative Words versus Rank') 
        pdf.savefig()
        plt.close() 
        
        ax = sns.boxplot(x='rank',y='perPosDoc',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Fraction of Positive Documents versus Rank') 
        pdf.savefig()
        plt.close() 
        
        ax = sns.boxplot(x='rank',y='perNegDoc',data=signalDF)
        fig= ax.get_figure()
        plt.suptitle('Fraction of Negative Documents versus Rank') 
        pdf.savefig()
        plt.close() 

plotVariables(signalDF2,2)
plotVariables(signalDF3,3)
plotVariables(signalDF4,4)
plotVariables(signalDF5,5)
plotVariables(signalDF6,6)