# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:17:50 2016

@author: nmvenuti

File is used to consolidate output reports from Rivanna 
and perform variable analysis on each output type
"""

#Import packages
import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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

###################################################
###Pull in data to generate median summary files###
###################################################

resultsDF=pd.read_csv(rawPath+'-summaryOutput-full.csv')
resultsDF.drop('Unnamed: 0',axis=1, inplace=True)
testDF=resultsDF[['cocowindow','startCount','netAngle','cvWindow','rfAccuracy','rfMae','svmAccuracy','svmMae']].groupby(['cocowindow','startCount','netAngle','cvWindow']).median().reset_index()
testDF.to_csv(rawPath+'resultMedians.csv')

#Top 10 random forest medians
testDF.sort('rfAccuracy',ascending=True).head(n=10).to_csv(rawPath+'bestRF-mae.csv')

#Top 10 SVM medians
testDF.sort('svmAccuracy',ascending=True).head(n=10).to_csv(rawPath+'bestSVM-mae.csv')

###########################################################
###Get outputs of files for visual analysis of variables###
###########################################################

#Get raw files
rawFileList=[]
for dirpath, dirnames, filenames in os.walk(rawPath):
    for filename in [f for f in filenames ]:
        if 'masterOutput.csv' in filename:
            rawFileList.append(os.path.join(dirpath, filename))

#Create list of lists with coco,cv,netAng,SC
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

#Merge all files into one dataframe if start count=0
totalDF = pd.DataFrame()
combineList = []
for row in range(len(fileDF)):
    if fileDF['SC'].ix[row]==0:
        storeDF = pd.read_csv(fileDF['filepath'].ix[row],index_col=None)
        for col in ['coco','cv','netAng','SC']:
            storeDF[col]=fileDF[col].ix[row]
        combineList.append(storeDF)
smallDF = pd.concat(combineList)

#Add rank
smallDF=addRank(smallDF)




#Plot variables based on parameters
#Scatter
def scatter3d(x,y,z,xTitle,yTitle,zTitle,colorTitle, cs, colorsMap='gray_r',saveFig=False):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    ax.set_zlabel(zTitle)
#    ax.w_xaxis.gridlines.set_lw(3.0)
#    ax.w_yaxis.gridlines.set_lw(3.0)
#    ax.w_zaxis.gridlines.set_lw(3.0)
    scalarMap.set_array(cs)
    cbar=fig.colorbar(scalarMap)
    cbar.set_label(colorTitle)
    if saveFig:
        fig.savefig('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/'+zTitle+'.png',bbox_inches='tight')  # save the figure to file  

#x=np.array(totalDF['coco'])
#y=np.array(totalDF['cv'])
#z=np.array(totalDF['avgSD'])
#c=np.array(totalDF['rank'])
#scatter3d(x,y,z,'Co-Occurrence Window','cv','sd', 'rank', c)


#Density curve
def surface3D(df,x,y,z,xTitle,yTitle,zTitle,plotTitle,colorsMap='gray_r',saveFig=False):
    # re-create the 2D-arrays
    x1 = np.linspace(df[x].min(), df[x].max(), len(df[x].unique()))
    y1 = np.linspace(df[y].min(), df[y].max(), len(df[y].unique()))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((df[x], df[y]), df[z], (x2, y2), method='cubic')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=colorsMap,
        linewidth=0, antialiased=False)
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    ax.set_zlabel(zTitle)
    #ax.set_zlim(-1.01, 1.01)
    
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(plotTitle)
    if saveFig:
        fig.savefig('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/'+plotTitle+'.png',bbox_inches='tight')   # save the figure to file   

    # ~~~~ MODIFICATION TO EXAMPLE ENDS HERE ~~~~ #
    
#    plt.show()
#Average Semantic Density verus Co-Occurrence and Coco Window
surface3D(smallDF,'coco','cv','avgSD','Co-Occurrence Window','Context Vector Window', 'Average Semantic Density','Average Semantic Density versus Hyperparameters',plt.cm.coolwarm,True)

#Average Eigenvector Centrality versus Network Angle and Coco Window
surface3D(smallDF,'netAng','coco','avgEVC','Network Angle','Co-Occurrence Window', 'Average Eigenvector Centrality','Average Eigenvector Centrality versus Hyperparameters',plt.cm.coolwarm,True)


#Semantic Density versus rank 1 graphic
x=np.array(smallDF['coco'])
y=np.array(smallDF['cv'])
z=np.array(smallDF['avgSD'])
c=np.array(smallDF['rank'])
scatter3d(x,y,z,'Co-Occurrence Window','Context Vector Window','Average Semantic Density', 'Rank', c,plt.cm.coolwarm,True)

#multiple graphics
rankList=list(set(smallDF['rank']))
rankList.sort()

fig = plt.figure(figsize=(24,12))
for i in range(len(rankList)):
    df=smallDF[smallDF['rank']==rankList[i]]
    x=np.array(df['coco'])
    y=np.array(df['cv'])
    z=np.array(df['avgSD'])
#    figY=int(np.floor(i/4.)+1)
#    figX=np.mod(i,4)+1
    
    ax = fig.add_subplot(2, 4, i+1, projection='3d')
    ax.scatter(x, y, z)
    ax.set_zlim(0,1.)
    ax.set_xlabel('Co-Occurrence Window')
    ax.set_ylabel('Context Vector Window')
    ax.set_zlabel('Average Semantic Density')
    pltTitle='Average Semantic Density- Rank '+str(rankList[i])
    plt.title(pltTitle)
    plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()

fig.savefig('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/scatterPlots/multipleAVGSD.png',bbox_inches='tight')

fig = plt.figure(figsize=(24,12))
for i in range(len(rankList)):
    df=smallDF[smallDF['rank']==rankList[i]]
    x=np.array(df['coco'])
    y=np.array(df['cv'])
    z=np.array(df['avgEVC'])
#    figY=int(np.floor(i/4.)+1)
#    figX=np.mod(i,4)+1
    
    ax = fig.add_subplot(2, 4, i+1, projection='3d')
    ax.scatter(x, y, z)
    ax.set_zlim(0,1.)
    ax.set_xlabel('Co-Occurrence Window')
    ax.set_ylabel('Context Vector Window')
    ax.set_zlabel('Average Eigenvector Centrality')
    pltTitle='Average Eigenvector Centrality- Rank '+str(rankList[i])
    plt.title(pltTitle)
    plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()

fig.savefig('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/scatterPlots/multipleAVGEVC.png',bbox_inches='tight')

#multiple graphics
#for i in set(smallDF['rank']):
#    df=smallDF[smallDF['rank']==i]
#    x=np.array(df['coco'])
#    y=np.array(df['cv'])
#    z=np.array(df['avgEVC'])
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.scatter(x, y, z)
#    ax.set_zlim(0,1.)
#    ax.set_xlabel('Co-Occurrence Window')
#    ax.set_ylabel('Context Vector Window')
#    ax.set_zlabel('Average Eigenvector Centrality')
#    pltTitle='Average Eigenvector Centrality- Rank '+str(i)
#    plt.title(pltTitle)
#    fig.savefig('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/scatterPlots/'+pltTitle+'.png')

#Eigen Vector Centrality versus rank
x=np.array(smallDF['coco'])
y=np.array(smallDF['netAng'])
z=np.array(smallDF['avgEVC'])
c=np.array(smallDF['rank'])
scatter3d(x,y,z,'Co-Occurrence Window','Network Angle','Average Eigenvector Centrality', 'Rank', c,'gray_r',True)


with PdfPages('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/3DSurfacePlots-mean color scale.pdf') as pdf:
    #Average semantic density versus coco window and context vector window
    surface3D(smallDF,'coco','cv','avgSD','Co-Occurrence Window','Context Vector Window', 'Average Semantic Density','Average Semantic Density versus Windows',plt.cm.coolwarm)
    pdf.savefig()
    plt.close()     
    
    #Average Eigenvector Centrality versus Network Angle and Coco Window
    surface3D(smallDF,'netAng','coco','avgEVC','Network Angle','Co-Occurrence Window', 'Average Eigenvector Centrality','Average Eigenvector Centrality versus Windows',plt.cm.coolwarm)
    pdf.savefig()
    plt.close()
    
    #Semantic Density versus rank
    x=np.array(smallDF['coco'])
    y=np.array(smallDF['cv'])
    z=np.array(smallDF['avgSD'])
    c=np.array(smallDF['rank'])
    scatter3d(x,y,z,'Co-Occurrence Window','Context Vector Window','Average Semantic Desnity', 'Rank', c,plt.cm.coolwarm)
    pdf.savefig()
    plt.close()
    
    #Eigen Vector Centrality versus rank
    x=np.array(smallDF['coco'])
    y=np.array(smallDF['netAng'])
    z=np.array(smallDF['avgEVC'])
    c=np.array(smallDF['rank'])
    scatter3d(x,y,z,'Co-Occurrence Window','Network Angle','Average Eigenvector Centrality', 'Rank', c,plt.cm.coolwarm)
    pdf.savefig()
    plt.close()
#generate 3D plot




#fig = plt.figure(figsize=(9,6))
#ax=fig.gca(projection='3d')
#surf=ax.plot_surface(x,y,z,rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True)
#ax.set_xlabel('strike')
#ax.set_ylabel('time-to-maturity')
#ax.set_zlabel('implied volatility')
#fig.colorbar(surf,shrink=0.5,aspect=5)

#Index([u'Unnamed: 0', u'groupId', u'files', u'timeRun', u'perPos', u'perNeg', u'perPosDoc', u'perNegDoc', 
#u'judgementCount', u'judgementFrac', u'avgSD', u'avgEVC', u'groupName', u'rank'], dtype='object')

##################################
#####Review Consolidated Data#####
##################################

#def plotVariables(signalDF,idNumber):
#    
#    #Subset to groups of proper length
#    signalDF=signalDF[signalDF['files']>5]    
#    
#    #Create box plots
#    with PdfPages('./github/nmvenuti/DSI_Religion/variableAnalysis/outputs/variable importance results-window '+str(idNumber)+'.pdf') as pdf:
#    
#        
#        ax = sns.boxplot(x='rank',y='judgementFrac',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Average Fraction of Judgements versus Rank')
#        pdf.savefig()
#        plt.close() 
#    
#        ax = sns.boxplot(x='rank',y='judgementCount',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Average Number of Judgements versus Rank') 
#        pdf.savefig()
#        plt.close() 
#     
#    
#        ax = sns.boxplot(x='rank',y='avgEVC',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Eigenvector Centrality versus Rank')
#        pdf.savefig()
#        plt.close() 
#        
#        
#        ax = sns.boxplot(x='rank',y='avgSD',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Context Vector Similarity versus Rank') 
#        pdf.savefig()
#        plt.close() 
#        
#        ax = sns.boxplot(x='rank',y='perPos',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Fraction of Positive Words versus Rank') 
#        pdf.savefig()
#        plt.close()     
#        
#        ax = sns.boxplot(x='rank',y='perNeg',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Fraction of Negative Words versus Rank') 
#        pdf.savefig()
#        plt.close() 
#        
#        ax = sns.boxplot(x='rank',y='perPosDoc',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Fraction of Positive Documents versus Rank') 
#        pdf.savefig()
#        plt.close() 
#        
#        ax = sns.boxplot(x='rank',y='perNegDoc',data=signalDF)
#        fig= ax.get_figure()
#        plt.suptitle('Fraction of Negative Documents versus Rank') 
#        pdf.savefig()
#        plt.close() 
#
#plotVariables(signalDF2,2)
#plotVariables(signalDF3,3)
#plotVariables(signalDF4,4)
#plotVariables(signalDF5,5)
#plotVariables(signalDF6,6)