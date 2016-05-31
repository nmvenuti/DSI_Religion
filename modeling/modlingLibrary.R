#Modeling library
#Will use artifical neural networdks, ordered logistic, random forest, and support vector machines
#Load test data for analysis
setwd("~/GitHub/dsicapstone-predicting_extremism/nmvenuti_sandbox")
filepath='C:/Users/nmvenuti/Documents/GitHub/dsicapstone-predicting_extremism/nmvenuti_sandbox/spring break updates/Test_files'

####################
#####Data Input#####
####################
library('stringr')

# dat<-data.frame()

dataFiles <- list.files(paste0(filepath))

#Get all inputs except semco
for (inputFile in dataFiles[str_detect(dataFiles,paste0('signal'))]) {
  
  #Read in data
  signalData<-data.frame(read.csv(paste0(filepath,'/',inputFile)))
  signalData$parameter<-gsub('.csv','',gsub('signal_', "", inputFile))
  ifelse(exists('dat'),dat<-cbind(dat,signalData),dat<-signalData)
}

#Add semco columns
dat[,c('semco1','semco2','semco3','semco4','semco5','semco6','semco7','semco8','semco9','semco10')]<-NA

#Get semco files
dataFiles <- list.files(paste0(filepath,'/semco'))

#Get all inputs except semco
for (inputFile in dataFiles[str_detect(dataFiles,paste0('signal'))]) {
  
  #Read in data
  signalData<-data.frame(read.csv(paste0(filepath,'/semco','/',inputFile),stringsAsFactors = F))
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
length(dataFiles)
#Remove first column (not used)
dat<-data.frame(dat[,-1])


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



#Test data and parameters
# dat=read.csv('test_data_models.csv')
y<-'groupRank'
# x<-c("X.PosWords", "X.NegWords", "X.PosDoc","X.NegDoc")
x<-c('acom',"subgraph_centrality","eigenvector_centrality","X.PosWords","X.NegWords" )

hiddenLayers=1
thresholdValue=0.1
learningrate=0.01
activationFunction='logistic' #can also use tanh
trees<-100

###############
#####Setup#####
###############

#Create formula
dataFormula<-paste(c(y," ~ ",paste(x, collapse ="+")),collapse = "")



##################################
#####Artifical Neural Network#####
##################################
library(neuralnet)
netModel <- neuralnet(as.formula(dataFormula), trainDat, hidden=hiddenLayers,threshold=thresholdValue,learningrate=learningrate,algorithm = 'rprop+' ,act.fct=activationFunction)
netSummary<-summary(netModel)
netTrainResults<-compute(netModel,trainDat[,x])
netTestResults<-compute(netModel,testDat[,x])

#Get results
nrow(trainDat[abs(trainDat$groupRank-netTrainResults$net.result)<1,])/nrow(trainDat)
# 0.6608695652
nrow(testDat[abs(testDat$groupRank-netTestResults$net.result)<1,])/nrow(testDat)
#0.6632653061

##########################
#####Ordered logistic#####
##########################

#Load library
library(MASS)

#Create data for ol
olTestDat<-testDat
olTrainDat<-trainDat

#Ensure response is factor
olTestDat[,y]<-factor(olTestDat[,y],ordered=TRUE)
olTrainDat[,y]<-factor(olTrainDat[,y],ordered=TRUE)

#Create model
olModel<-polr(as.formula(dataFormula),data=olTrainDat,Hess=TRUE)

#Extract summary data
olSummary<-summary(olModel)

#Create prediction
olTestResults<-as.integer(predict(olModel,olTestDat[,x]))
olTrainResults<-as.integer(predict(olModel,olTrainDat[,x]))


#Get results
nrow(trainDat[abs(trainDat$groupRank-olTrainResults)<1,])/nrow(trainDat)

nrow(testDat[abs(testDat$groupRank-olTestResults)<1,])/nrow(testDat)


# #Save model and summary information
# save(olModel,olSummary,file="olOutput.RData")

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
nrow(trainDat[abs(trainDat$groupRank-rfTrainResults)<1,])/nrow(trainDat)

nrow(testDat[abs(testDat$groupRank-rfTestResults)<1,])/nrow(testDat)


#################################
#####Support Vector Machines#####
#################################
library(e1071)
svmModel=svm(trainDat[,x],trainDat[,y], kernel ="radial", degree = 3, probability = TRUE)
svmSummary<-summary(svmModel)
svmTrainResults<-predict(svmModel,trainDat[,x])
svmTestResults<-predict(svmModel,testDat[,x])

#Output train and test accuracy
nrow(trainDat[abs(trainDat$groupRank-svmTrainResults)<1,])/nrow(trainDat)

nrow(testDat[abs(testDat$groupRank-svmTestResults)<1,])/nrow(testDat)
