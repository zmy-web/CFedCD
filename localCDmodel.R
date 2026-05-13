rm(list = ls())
options(java.parameters='-Xmx10g')
# install.packages('BiocManager',repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')
# options('BioC_mirror'='https://mirrors.tuna.tsinghua.edu.cn/bioconductor/')
# BiocManager::install(c('graph','RBGL','Rgraphviz'))
# install.packages(c('kpcalg'), repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')
# BiocManager::install("GO.db")
options(digits = 4)
#library(kpcalg)
library(pcalg)
library(bnlearn)
library("RBGL")
library("graph") #biocLite
library("RBGL")#biocLite
library("Rgraphviz")#biocLite
library(igraph)
library(GO.db)
library(lattice)
library(caret)
library(pROC)
library(modEvA)
library(bnlearn)

traintest<- readRDS("...data/traintest.rds")
alltrain<- traintest$train
alltest<- traintest$test

for(j in 2:length(alltrain)){
  alltrain[,j]<- as.factor(alltrain[,j])
  alltest[,j]<- as.factor(alltest[,j])
}

ofn <- setdiff(colnames(alltrain)[-1],c('age','sex','race'))
bl1 = tiers2blacklist(list('sex',ofn))
bl2 =tiers2blacklist(list(ofn[-1],'label'))
bl3 =tiers2blacklist(list('age',ofn))
bl4 =tiers2blacklist(list('race',ofn))

bl=data.frame(rbind(bl1,bl2,bl3,bl4,c('sex','age'),c('age','sex'),c('sex','race'),
                    c('race','age'),c('age','race'),c('race','sex')))


#
prcCI <- function(pred_scores,true_labels) {
  library(plyr)
  library(PRROC)
  #
  data <- data.frame(score = pred_scores, label = true_labels)
  #
  set.seed(100)  
  bootstrap_samples <- 1000  # 
  auprc_values <- rep(0, bootstrap_samples)
  
  for (i in 1:bootstrap_samples) {
    boot_data <- data[sample(1:nrow(data), replace = TRUE), ]
    
    pr_obj <- pr.curve(scores.class0 = boot_data$score, weights.class0 = as.numeric(as.character(boot_data$label)))
    auprc_values[i] <- pr_obj$auc.integral
  }
  
  # 
  ci <- quantile(auprc_values, probs = c(0.025, 0.975))
  return(print(paste(round(ci[1],4),round(ci[2],4),sep='~')))
}

#from-to 
modstr <- function(fnet){
  chd <- as.vector(unique(fnet$to))
  pad <- as.vector(unique(fnet$from))
  ####
  allnode <- colnames(alltrain)[-1]
  networknode <- c(chd,pad)
  noselnode <- setdiff(allnode,networknode)
  ####
  sing <- c(setdiff(pad ,chd),noselnode)
  p1 <- paste('[',sing, ']' ,sep = '',collapse = '')#
  #
  p2 <- c()
  for(j in 1:length(chd)){
    sf <- subset(fnet,to==chd[j])
    if(nrow(sf)==1){
      p2[j] <- paste('[',chd[j],'|',sf$from,']', sep = "",collapse = "")
    }else{
      p0 <- paste(sf$from, collapse = ':')
      p2[j]<- paste( '[' ,chd[j],'|',p0, ']' , sep ="",collapse ="")
    }
  }
  modelstring = paste(p1,paste(p2, sep = '',collapse = ""), sep = "")
  return (modelstring)
}


st= sort(unique(as.numeric(alltrain$hospitalid)))
scoreM <- c()#
adjmat <- list()
strength<- list()
predList <- list()
measM <- data.frame(matrix(0,nrow = length(st),ncol =9,
                           dimnames =list(c(st),
                                          c('Sensitivity','Specificity','Precision','Accuracy','F1','AUROC','AUPRC',"ROCCI",'PRCCI')) ))
for(s in 1:length(st)){

  cat("now is runing is:",s)
  #
  train =subset(alltrain,hospitalid==st[s])
  test = subset(alltest,hospitalid==st[s])
  
  set.seed(100)
  bynet <- hc(train[,-1],score = 'k2',blacklist =bl,
                 whitelist = NULL,maxp = 10)#
  streng <- arc.strength(bynet, train[,-1], criterion = "mc-x2")#mc-x2
  sig.arcs <- subset(streng, strength < 0.05)
  strength[[s]]<- sig.arcs
  # # 
  modelstring = modstr(fnet = sig.arcs)

  bynet<- model2network(modelstring)
  
  scoreM[s]=score(bynet, train[,-1], type = "k2")
  
  ##step1：
  adjmat[[s]]<- amat(bynet)#
  
  ##step2：
  fitted <- bn.fit(bynet,method='bayes', train[,-1])#
  pre <- predict(fitted,data=test[,-1],node='label',method ='bayes-lw',prob = TRUE)#0/1#
  pred<- attr(pre,'prob')[2,]#
  predList[[s]] <- pred
  AUROC <- pROC::auc(test[,'label'],pred)
  CI=ci.auc(test[,'label'],pred)
  ROCCI=paste(round(CI[1],4),round(CI[3],4),sep = '~')
  AUPRC=modEvA::AUC(obs = test$label,pred=pred,curve = 'PR',simplif = TRUE,main='PR curve')
  PRCCI=prcCI(pred,test$label)
  #
  rocobj <- roc(test[,'label'],pred)
  rt <- coords(rocobj, "best")
  preds <- ifelse(pred>rt$threshold,1,0)
  R_P <- data.frame(refer=test[,'label'],pred=preds)
  tp <- nrow(R_P[which((R_P$refer==1)&(R_P$pred==1)),])
  fp <- nrow(R_P[which((R_P$refer==0)&(R_P$pred==1)),])
  fn <- nrow(R_P[which((R_P$refer==1)&(R_P$pred==0)),])
  tn <- nrow(R_P[which((R_P$refer==0)&(R_P$pred==0)),])
  Sensitivity = tp/(tp+fn)
  Recall= tp/(tp+fn)
  Specificity = tn/(fp+tn)
  Precision = tp/(tp+fp)
  Accuracy = (tp+tn)/(tp+fp+fn+tn)
  F1 = 2*Precision*Recall/(Precision+Recall)
  measM[s,] <- data.frame(Sensitivity,Specificity,Precision,Accuracy,F1,AUROC,AUPRC,ROCCI,PRCCI)
}
measM

names(strength)<- st
names(adjmat)<- st#
names(predList)<- st
localres <- measM
scoreM #
scoreDT = data.frame(sid= st,score=scoreM)

write.csv(localres,".../data/localMeares.csv")#
write.csv(scoreDT,".../data/locscoreDT.csv")
saveRDS(adjmat,".../data/localadjmat.rds")
saveRDS(strength,".../data/localstrength.rds")
saveRDS(predList,".../data/LocalpredList.rds")

