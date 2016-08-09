
# coding: utf-8

# In[2]:

import pandas as pd
import xml.etree.ElementTree as ET
import os, sys
import numpy as np
import pandas as pd
import re
import random
import nltk
tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
import string
import datetime
import imp


# ## Function to Load Dorothy Day File Metadata

# In[ ]:

def getDorothyDaymetadata(dataloc):

    files = os.listdir(dataloc)

    files = [file for file in files if '.txt' in file]

    fileData = pd.DataFrame(files, columns = ['fileName'])

    firstlines = []
    for file in fileData.fileName:
        text = rawText=open(dataloc+"/"+file, encoding = 'latin1').read()
        lineone = text[0:text[1:len(text)].find('\n')+1]
        lineone = re.sub('\nThe Catholic Worker, ','',lineone)
        firstlines.append(lineone)

    fileData['firstline'] = firstlines

    # Building Parsing for Dates in DD Files
    date_est = []
    for i in range(0,len(fileData)):
        if fileData.loc[i,'firstline'][0] == '\n':
            # print(fileData.loc[i,'firstline'][0])
            # print("Drop")
            date_est.append('unclear date')
        else:
            # print(fileData.loc[i,'firstline'][0])
            # print("Keep")
            pieces = fileData.loc[i,'firstline'].split(" ")
            if len(pieces[1]) != 5:
                date_est.append('unclear date')
            else:
                date_est.append(pieces[0]+', '+pieces[1][0:4])
                # print(pieces)

    fileData['date_est'] = date_est

    datelist = []
    for date in date_est:
        try:
            datelist.append(datetime.datetime.strptime(date, '%B, %Y'))
        except ValueError:
            temp = re.split('[-,/]',date)
            temp = temp[0]+','+temp[len(temp)-1]
            try: 
                temp = datetime.datetime.strptime(temp, '%B, %Y')
                datelist.append(temp)
            except ValueError:
                try: 
                    temp = datetime.datetime.strptime(temp, '%b, %Y')
                    datelist.append(temp)
                except ValueError:
                    datelist.append("unclear date")

    fileData['date_clean'] = datelist
    
    fileData[fileData.date_clean == 'unclear date'] = np.NaN
    fileData = fileData.dropna()
    fileData = fileData[fileData.date_clean > datetime.datetime(1900, 1, 1, 0, 0)]
    del fileData['date_est']
    del fileData['firstline']
    fileData.reset_index(inplace=True, drop=True)
    
    fileData.sort_values(by='date_clean', ascending=True, inplace=True)
    
    return fileData


# ## Function to Load File Metadata when Date 1st Item in FileName e.g. 2 March 2017

# In[ ]:

def getSimplemetadata(dataloc):
    files = os.listdir(dataloc)
    files = [file for file in files if '.txt' in file]
    fileData = pd.DataFrame(files, columns = ['fileName'])
    temp = [file.split('_')[0] for file in files]
    fileData['date'] = temp
    # Convert Str Dates to Date-Time Objects
    dates_clean = [datetime.datetime.strptime(date,'%d %B %Y') for date in fileData.date]
    fileData['date_clean'] = dates_clean
    fileData.sort_values(by='date_clean', ascending=True, inplace=True)
    return fileData


# ## Function to Load WBC File Metadata

# In[1]:

def getWBCmetadata(dataloc):
    files = os.listdir(dataloc)
    files = [file for file in files if '.txt' in file]
    fileData = pd.DataFrame(files, columns = ['fileName'])
    temp = [file.split('_')[2][0:8] for file in files]
    fileData['date'] = temp
    # Convert Str Dates to Date-Time Objects
    dates_clean = [datetime.datetime.strptime(date,'%Y%m%d') for date in fileData.date]
    fileData['date_clean'] = dates_clean
    fileData.sort_values(by='date_clean', ascending=True, inplace=True)
    return fileData


# ## Function to Load Ghandi File Metadata

# In[ ]:



