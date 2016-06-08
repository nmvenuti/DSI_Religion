
# coding: utf-8

# In[1]:

import gc, sys, os
os.chdir('..')
import os.path
import pandas as pd
import numpy as np
import nltk
import time
sys.path.append('./python/')
import semanticDensity as sd
# import syntacticParsing as sp


# In[10]:

# ! pip install python-igraph


# In[2]:

import igraph


# In[3]:

import pickle
import timeit
import datetime
import math


# In[4]:

dsm = pickle.load(open('testDSM.pickle', "rb" ))


# In[5]:

del dsm['']


# In[6]:

cosines = np.zeros((len(dsm.keys()),len(dsm.keys())))


# In[7]:

cosinesDF = pd.DataFrame(cosines, columns = dsm.keys(), index = dsm.keys())


# In[8]:

start = datetime.datetime.now()
for key1 in dsm.keys():
    vec1 = dsm[key1]
    for key2 in dsm.keys():
        if cosinesDF.loc[key1,key2] == 0.0:
            vec2 = dsm[key2]
            dist = sd.get_cosine(vec1,vec2)
            cosinesDF.loc[key1,key2] = dist
            cosinesDF.loc[key2,key1] = dist
runtime = datetime.datetime.now() - start
print(str(runtime))


# In[9]:

pickle.dump(cosinesDF, open( "testAdjMatrix.pickle", "wb" ) )


# In[61]:

cosinesDF = pickle.load(open('testAdjMatrix.pickle', "rb" ))


# In[68]:

cosinesDF[0:10]


# In[63]:

adj = cosinesDF.copy()


# In[64]:

start = datetime.datetime.now()
adj[cosinesDF >= math.cos(math.radians(30))] = 0 # Converting 30 degree threshold to radians to a cosine value
runtime = datetime.datetime.now() - start
print(str(runtime))


# In[66]:

adj[cosinesDF < math.cos(math.radians(30))] = 1 # Converting 30 degree threshold to radians to a cosine value


# In[67]:

adj[0:10]


# In[71]:

adjList = adj.values.tolist()


# In[74]:

net = igraph.Graph.Adjacency(adjList, mode = "undirected")


# In[ ]:

# net <- graph.adjacency(adjacency,mode='undirected')


# In[93]:

edgelist = igraph.Graph.get_edgelist(net) # Not Used


# In[96]:

len(edgelist)


# In[ ]:

# edgelist <- get.edgelist(net)


# In[85]:

ev_centrality = igraph.Graph.evcent(net) # Works, but NEED TO FILTER TO TARGET WORDS ONLY


# In[91]:

ev_centrality[0:10]


# In[ ]:




# In[ ]:

# ev.centrality <- evcent(net)$vector[target_words]


# In[89]:

sg_centrality = igraph.Graph.cent(net)


# In[ ]:

centrality <- subgraph.centrality(net)[target_words]


# In[ ]:

return 


# In[ ]:

data.frame(subgraph_centrality=log(mean(centrality, na.rm=T)),
         eigenvector_centrality=mean(ev.centrality, na.rm=T),
         group=group_id)


# In[ ]:




# In[ ]:




# In[ ]:



