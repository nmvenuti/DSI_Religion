#Create DSM function 
createDSM<-function(filepath,datafile_name,wordCo){
  #Subset vectors only greater than min matches
  minMatches = 10
  countWords <- wordCo[,.(length(context)), 
                                 by=target][V1 > minMatches]$target
  wordCo <- wordCo[target %in% countWords &
                                         context %in% countWords]
  wordCo[,c("target", "context"):=
                     list(as.factor(target), as.factor(context))]
  
  #run Distributional semantic model(DSM)
  rawDSM <- dsm(target=wordCo$target,
                feature=wordCo$context,
                score=wordCo$freq,
                N=100)
  
  #Subset DSM for, project DSM into lower-dimenstional subspace
  rawDSM <- subset(rawDSM, nnzero >= 10, nnzero >= 10, T)
  dsmProj <- dsm.projection(rawDSM, "svd", n=300)
  col(dsmProj)
  nrow(dsmProj)
  #Transpose and convert dsm into indexical dataframe, only take first 300
  dsmProjName<-dsmProj_filename(filepath, datafile_name)
  #Save DSM projections
  save(dsmProj, file=dsmProjName)
}



#Quantify context vectors function
quantifyContextPar<-function(search_word,filepath,datafile_name,target_corpus,dsmProj,minMatches=25,window_length=15,sim_count=1000){
  # print("Start of quantify Context")
  dsmProj <- data.frame(dsmProj)
  dsmProj <- t(dsmProj)
  #for each document in corpus
  for (i in 1:length(target_corpus)){
    
    #extract text
    text_string=target_corpus[[i]]$content
    n=length(text_string)
    doc_context_vector_df=data.frame(text_string)
    colnames(doc_context_vector_df)=c('target')
    doc_context_vector_df$context_vector=""
    
    for (j in 1:n){
      
      #Set window bounds
      lower_bound=j-window_length
      upper_bound=j+window_length
      if(lower_bound<1){
        lower_bound=1
      }
      if(upper_bound>n){
        upper_bound=n
      }
      
      #Identify target word
      id_word=text_string[j]
      
      #Get context vector ids and extract target word
      context_names=text_string[lower_bound:upper_bound]
      # context_names=context_names[-(j-lower_bound)]
      
      #Remove words stripped from dsm
      context_names=context_names[context_names %in% colnames(dsmProj)]
      
      #Add in section to extract context vectors
      doc_context_vector_df$context_vector[j]=paste(context_names,collapse = "-") 
      
    }
    
    if(exists('context_vector_df')){
      context_vector_df<-rbind(context_vector_df,doc_context_vector_df)
    }else{
      context_vector_df<-data.frame(doc_context_vector_df)
      colnames(context_vector_df)<-colnames(doc_context_vector_df)
    }
  }
  
    
  #Extract words
  # print(search_word)
  
  context_vector_subset<-context_vector_df$context_vector[context_vector_df$target==search_word]
  if(length(context_vector_subset>0)){
    #Calculate context vectors
    cosineSim=0
    for (cvIndex in 1:sim_count){
      x=ceiling(runif(sim_count,min = 0,max = length(context_vector_subset)))
      y=ceiling(runif(sim_count,min = 0,max = length(context_vector_subset)))
      
      cvWordsX=as.vector(strsplit(context_vector_subset[x],"-"))[[1]]
      cvWordsY=as.vector(strsplit(context_vector_subset[y],"-"))[[1]]
      
      #Get list of unique words
      unique_words<-unique(c(cvWordsX,cvWordsY))

      #Create empty vector for cv
      cvX<-rep(0,nrow(dsmProj))
      cvY<-rep(0,nrow(dsmProj))
      
      #Calculate context vector
      for (word in unique_words){
        if( word %in% colnames(dsmProj)){
          if(word %in% cvWordsX){
            cvX=dsmProj[,word]+cvX
          }
          if(word %in% cvWordsY){
            cvY=dsmProj[,word]+cvY
          }
        }
      }
      #Add cosine sim for random pairs to cosine sim
      cosineSim=cosineSim+cosine(as.vector(cvX),as.vector(cvY))[[1,1]]
    }
    cvCosineSim=cosineSim/sim_count
    
  }else{
  #Add Na if word not found in context vectors
  cvCosineSim=NA
  }
  # print("end of quant cont")
  
  return(cvCosineSim)
}


