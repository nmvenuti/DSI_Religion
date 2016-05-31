#Modeling library
#Will use artifical neural networdks, ordered logistic, random forest, and support vector machines
#Load test data for analysis
setwd("~/GitHub/dsicapstone-predicting_extremism/nmvenuti_sandbox")
filepath='C:/Users/nmvenuti/Documents/GitHub/dsicapstone-predicting_extremism/nmvenuti_sandbox/spring break updates/ref_complete1'

####################
#####Data Input#####
####################
library('stringr')
library('data.table')

# dat<-data.frame()

dataFiles <- list.files(paste0(filepath))

#Define signal type
signals<-c('judgements','semACOM','semContext','sentiment','network')

#Extract LDA
signalLDA<-data.frame(read.csv(paste0(filepath,'/','signal_LDA.csv')))
ldaDTB<-data.table(signalLDA)
setnames(ldaDTB,'GroupID','group')
dat<-ldaDTB[,lapply(.SD,mean),by="group"]
dat$X<-NULL

#Get all inputs except semco
for (signalID in signals){
  if(signalID!='semContext'){
    for (inputFile in dataFiles[str_detect(dataFiles,paste0(signalID))]) {
      
      #Read in data
      inputData<-data.frame(read.csv(paste0(filepath,'/',inputFile)))
      inputData$parameter<-signalID
      inputData$X<-NULL
      ifelse(exists('signalData'),signalData<-rbind(signalData,inputData),signalData<-inputData)
    }
    dat<-merge(x = dat, y = signalData, by = "group", all = TRUE)
    signalData<-NULL
  }
}

#Add semco columns
dat[,c('semco1','semco2','semco3','semco4','semco5','semco6','semco7','semco8','semco9','semco10')]<-NA

#Get semco files
dataFiles <- list.files(paste0(filepath))

#Get all semco inputs
for (inputFile in dataFiles[str_detect(dataFiles,'semContext')]) {
  
  #Read in data
  signalData<-data.frame(read.csv(paste0(filepath,'/',inputFile),stringsAsFactors = F))
  groupName<-signalData$groupName[1]
  dat$semco1[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='X0']
  dat$semco2[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V2']
  dat$semco3[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V3']
  dat$semco4[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V4']
  dat$semco5[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V5']
  dat$semco6[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V6']
  dat$semco7[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V7']
  dat$semco8[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V8']
  dat$semco9[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V9']
  dat$semco10[dat$group==groupName]<-signalData$t.cvCosineSim.[signalData$X=='V10']
}

nrow(dat)
length(dat)
dat<-data.frame(dat)

#Add average semco
dat$averageSemco<-apply(dat[,c('semco1','semco2','semco3','semco4','semco5','semco6','semco7','semco8','semco9','semco10')],1,mean)
topicModel<-dat[grepl("Topic", names(dat))]

# Enable 1-hot encoding for probable topics
for (i in 1:nrow(dat)){
  # dat[i,'ProbableTopic']<-which.max(x[i,])[[1]]
  columnName<-attributes(which.max(topicModel[i,]))$names
  dat[i,colnames(x)]<-0
  dat[i,columnName]<-1
}

# dat$ProbableTopic<-as.factor(dat$ProbableTopic)

#Create group lookup table
groupName<-c('WBC', 'PastorAnderson', 'NaumanKhan', 'DorothyDay', 'JohnPiper', 'Shepherd',
'Rabbinic', 'Unitarian', 'MehrBaba')
groupRank<-c(1,2,3,4,4,4,6,7,8)
groupRank<-cbind.data.frame(groupName,groupRank)

#Add in response variable for groups
dat$groupRank<-999
for (name in groupName){
  dat$groupRank[grep(name,dat$group)]=print(groupRank$groupRank[groupRank$groupName==name])
}

#Split into test and training

testDat<-dat[grep("test", dat$group),]
trainDat<-dat[grep("train", dat$group),]


##################
#####Function#####
##################

testVariables<-function(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees,MSE=F){
  #Create formula
  dataFormula<-paste(c(y," ~ ",paste(x, collapse ="+")),collapse = "")
  
  
  
  # ##################################
  # #####Artifical Neural Network#####
  # ##################################
  library(neuralnet)
  netModel <- neuralnet(as.formula(dataFormula), trainDat, hidden=hiddenLayers,threshold=thresholdValue,learningrate=learningrate,algorithm = 'rprop+' ,act.fct=activationFunction)
  netSummary<-summary(netModel)
  netTrainResults<-compute(netModel,trainDat[,x])
  netTestResults<-compute(netModel,testDat[,x])

  #Get results

  nn_accuracy<-ifelse(MSE,mean((testDat$groupRank-netTestResults$net.result)^2),nrow(testDat[abs(testDat$groupRank-netTestResults$net.result)<1,])/nrow(testDat))

  
  ##########################
  #####Ordered logistic#####
  ##########################
  
  # #Load library
  # library(MASS)
  # 
  # #Create data for ol
  # olTestDat<-testDat
  # olTrainDat<-trainDat
  # 
  # #Ensure response is factor
  # olTestDat[,y]<-factor(olTestDat[,y],ordered=TRUE)
  # olTrainDat[,y]<-factor(olTrainDat[,y],ordered=TRUE)
  # 
  # #Create model
  # olModel<-polr(as.formula(dataFormula),data=olTrainDat,Hess=TRUE)
  # 
  # #Extract summary data
  # olSummary<-summary(olModel)
  # 
  # #Create prediction
  # olTestResults<-as.integer(predict(olModel,olTestDat[,x]))
  # olTrainResults<-as.integer(predict(olModel,olTrainDat[,x]))
  # 
  # 
  # #Get results
  # ol_accuracy<-ifelse(MSE,mean((testDat$groupRank-olTestResults)^2),nrow(testDat[abs(testDat$groupRank-olTestResults)<1,])/nrow(testDat))
  
  ####Should we be using factors or continuos here (kinda feelin contiuous)
  
  #######################
  #####Random Forest#####
  #######################
  library(randomForest)
  rfModel<-randomForest(as.formula(dataFormula),data = trainDat,ntree=trees)
  rfSummary<-summary(rfModel)
  rfTrainResults<-predict(rfModel,newdata = trainDat)
  rfTestResults<-predict(rfModel,newdata = testDat)
  
  #Output train and test accuracy
  rf_accuracy<-ifelse(MSE,mean((testDat$groupRank-rfTestResults)^2),nrow(testDat[abs(testDat$groupRank-rfTestResults)<1,])/nrow(testDat))
  
  
  #################################
  #####Support Vector Machines#####
  #################################
  library(e1071)
  svmModel=svm(trainDat[,x],trainDat[,y], kernel ="radial", degree = 3, probability = TRUE)
  svmSummary<-summary(svmModel)
  svmTrainResults<-predict(svmModel,trainDat[,x])
  svmTestResults<-predict(svmModel,testDat[,x])
  
  olTestResults<-0
  ol_accuracy<-0

  #Output train and test accuracy
  svm_accuracy<-ifelse(MSE,mean((testDat$groupRank-svmTestResults)^2),nrow(testDat[abs(testDat$groupRank-svmTestResults)<1,])/nrow(testDat))
  output<-data.frame(c(nn_accuracy,ol_accuracy,rf_accuracy,svm_accuracy))
  
  results<-data.frame(cbind(testDat$groupRank,netTestResults$net.result,olTestResults,rfTestResults,svmTestResults))
  colnames(results)<-c('Actual','ANN','OL','RF','SVM')
  
  output$test<-c('ANN','OL','RF','SVM')
  colnames(output)<-c('accuracy','test')
  totalOutput<-list(output,results)
  return(totalOutput)
}
#Set up variable testing
y<-'groupRank'
hiddenLayers=1
thresholdValue=0.1
learningrate=0.01
activationFunction='logistic' #can also use tanh
trees<-100

setwd(paste0(filepath,'/Output'))

strName<-'Full model'
x<-c("X.PosWords",'X.NegWords' ,'X.PosDoc','X.NegDoc','acom','averageSemco','avgPercJ','avgNumJ',colnames(topicModel),'eigenvector_centrality','subgraph_centrality')
model1<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model1[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model1[2],file = paste0('model_output-',strName,'.csv'))
          
strName<-'Full model-Semantic'
x<-c("X.PosWords",'X.NegWords',colnames(topicModel))
model2<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model2[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model2[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Full model-Performative'# need to add networks in'
x<-c('acom','averageSemco','avgPercJ','avgNumJ','eigenvector_centrality','subgraph_centrality')
model3<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model3[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model3[2],file = paste0('model_output-',strName,'.csv'))

#Individual predictors
strName<-'Coupled sentiment-words'
x<-c("X.PosWords",'X.NegWords')
model4<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model4[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model4[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Coupled sentiment-docs'
x<-c('X.PosDoc','X.NegDoc')
model5<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model5[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model5[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Total sentiment'
x<-c("X.PosWords",'X.NegWords' ,'X.PosDoc','X.NegDoc')
model6<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model6[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model6[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Topic models'
x<-c(colnames(topicModel))
model7<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model7[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model7[2],file = paste0('model_output-',strName,'.csv'))

strName<-'ACOM'
x<-c('acom')
model8<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model8[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model8[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Semco'
x<-c('averageSemco')
model9<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model9[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model9[2],file = paste0('model_output-',strName,'.csv'))

#Judgements
strName<-'Perceived judgements'
x<-c('avgPercJ')
model10<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model10[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model10[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Number judgements'
x<-c('avgNumJ')
model11<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model11[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model11[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Total judgements'
x<-c('avgPercJ','avgNumJ')
model12<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model12[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model12[2],file = paste0('model_output-',strName,'.csv'))

strName<-'Networks'
x<-c('eigenvector_centrality','subgraph_centrality')
model13<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model13[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model13[2],file = paste0('model_output-',strName,'.csv'))

#Best-Need to add in networks
strName<-'optimal model'
x<-c("X.PosWords",'X.NegWords','acom','averageSemco','avgPercJ','avgNumJ','eigenvector_centrality','subgraph_centrality')
model14<-testVariables(x,y,testDat,trainDat,hiddenLayers,thresholdValue,learningrate,activationFunction,trees, MSE = F)
write.csv(model14[1],file = paste0('model_accuracy-',strName,'.csv'))
write.csv(model14[2],file = paste0('model_output-',strName,'.csv'))
model14[1]

#Create general output file
modelOutput<-data.frame('ANN'=0,'OL'=0,'RF'=0,'SVM'=0)
for(i in 1:14){
  modelExtract<-data.frame(get(paste0('model',i))[1])$accuracy
  modelOutput<-rbind(modelOutput,modelExtract)
}
modelOutput<-modelOutput[2:15,]
modelOutput$ModelName<-c('Full model','Full model-Semantic','Full model-Performative','Coupled sentiment-words','Coupled sentiment-docs',
              'Total Sentiment','Topic Models','ACOM','Semco','Perceived Judgements','Number Judgements','Total Judgements',
              'Networks','Optimal Model')

modelOutput<-modelOutput[,c(5,1,3,4)]
write.csv(modelOutput,'modelSummary.csv')

#Test gradient boosting
library(gbm)
dataFormula<-paste(c(y," ~ ",paste(x, collapse ="+")),collapse = "")
gbmTest<-gbm(groupRank ~ X.PosWords+X.NegWords+acom+averageSemco+avgPercJ+avgNumJ+eigenvector_centrality+subgraph_centrality, data = trainDat, n.trees = 10000)
gbmPredictTest<-predict.gbm(gbmTest,testDat, n.trees = 10000)
gbmPredictTest
nrow(testDat[abs(testDat$groupRank-gbmPredictTest)<1,])/nrow(testDat)
# 0.8265306122

