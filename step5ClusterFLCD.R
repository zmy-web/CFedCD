

rm(list = ls())

# install.packages('BiocManager',repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')
# options('BioC_mirror'='https://mirrors.tuna.tsinghua.edu.cn/bioconductor/')
# BiocManager::install(c('graph','RBGL','Rgraphviz'))
# install.packages(c('kpcalg'), repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')
# BiocManager::install("GO.db")
options(digits = 4)
library(kpcalg)
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
library(reshape2)
library(plyr)
library(PRROC)
library(reticulate)
library(Rcpp)
#reticulate::repl_python()
#py_available()

# reticulate::py_install("pandas")
# reticulate::py_install("matplotlib")
#py_config()

source_python("D:/Program Files/R/Rfile/SC_FL_CD/notears.py")


###################################################################
path='.../SC_FL_CD/1031/'
traintest<- readRDS(paste(path,'traintest.rds',sep = ''))
alltrain<- traintest$train
alltest<- traintest$test
dim(alltrain)
for(j in 2:length(alltrain)){
  alltrain[,j]<- as.factor(alltrain[,j])
  alltest[,j]<- as.factor(alltest[,j])
}



st= sort(unique(as.numeric(alltrain$hospitalid)))


#step2

########################################################
#Extract the model's adjacency matrix, edge weights, and network scores from the local training data."

adjmat <- readRDS(paste(path,'localadjmat.rds',sep=''))
strength<- readRDS(paste(path,'localstrength.rds',sep=''))
scoreM <- read.csv(paste(path,'locscoreDT.csv',sep=''))
origscore <- scoreM$score

#Different model aggregation strategies or sensitivity analyses.

#mothod 1: max-min normalization 
score <- (origScore-min(origScore))/(max(origScore)-min(origScore))

##mothod 2:Reciprocal normalization
#score <- (1/abs(origScore))/sum(1/abs(origScore))

#mothod 3: Sigmoid-zscore normalization
#score <- 1/(1+exp(-scale(origScore)))

#Client IDs categorized by groups
si= c(1:4,10,11,13:16,18:22,24)
#si=c(7:9,12,17,23,25,26)
#si=c(5,6)
#si=27:30
N=58# num of nodes
summat <- matrix(0,nrow =N,ncol=N)
wsum <-0

for (s in 1:length(si)) {
  net <- strength[[si[s]]]#
  numEdg=nrow(net)#
  numtolSam= nrow(alltrain)#
  numLocSam=nrow(subset(alltrain,hospitalid==st[si[s]]))#
  rS <- numLocSam/numtolSam#
  idscore= score[si[s]]#
  adjM <- adjmat[[si[s]]]#
  nodes= colnames(alltrain)[-1]
  
  #1：from->to
  g <- graph_from_data_frame(d = net, directed = TRUE,vertices =nodes )

  
  adj_mi_mat <- as_adjacency_matrix(g, attr = NULL, sparse = FALSE)
  #wiadj<- idscore/(numLocSam*numEdg)*adj_mi_mat #1
  wiadj<- idscore*(rS*numEdg/(N*(N-1)))*adj_mi_mat #2
  summat<- summat+wiadj
  wsum <- wsum+idscore*(rS*numEdg/(N*(N-1)))

  ########## Sensitivity analysis,method 4: Additive model#############################
  # edge_comp <- numEdg / (N * (N - 1))
  # # Sample Proportion: ni / N
  # sample_proportion <- rS
  # # Additive 
  # omega <- (1/3)*idscore + (1/3)*edge_comp + (1/3)*rS
  # wiadj=omega*adj_mi_mat 
  # summat<- summat+wiadj
  # wsum <- wsum+omega
 
}

avgsummat <- round(summat/wsum,4)#




#write.csv(avgsummat,paste(path,'WMC3141.csv',sep=''),row.names = TRUE)


#*******************************************************
#step3：,python
#*******************************************************
prcCI <- function(pred_scores,true_labels) {
  #library(plyr)
  #library(PRROC)
  
  data <- data.frame(score = pred_scores, label = true_labels)
  
 
  set.seed(100)  # 
  bootstrap_samples <- 1000  #
  auprc_values <- rep(0, bootstrap_samples)
  
  for (i in 1:bootstrap_samples) {
    boot_data <- data[sample(1:nrow(data), replace = TRUE), ]
    
    pr_obj <- pr.curve(scores.class0 = boot_data$score, weights.class0 = as.numeric(as.character(boot_data$label)))
    auprc_values[i] <- pr_obj$auc.integral
  }
  

  ci <- quantile(auprc_values, probs = c(0.025, 0.975))
  return(print(paste(round(ci[1],4),round(ci[2],4),sep='~')))
}

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
dataset=avgsummat
cat("sum:", sum(dataset))

######################################################
# c1=c(0.5,1,1.5,2)
# c2=c(0.5,1,2,4)
#######################################################
#l1=0.1,l2=1,thed=0.1
W_est = notears_linear(as.matrix(dataset), lambda1=0.5, lambda2=1, loss_type='l2',
                      # h_tol=as.numeric(1e-8), rho_max=as.numeric(1e+16),
                       w_threshold = 0.1)#

res = data.frame(W_est)
rownames(res)<-colnames(dataset) 
colnames(res)<-colnames(dataset) 
#summary(res)

#

Wmet=res

convert_to_binary <- function(matrix) {
  binary_matrix <- ifelse(matrix != 0, 1, 0)
  return(binary_matrix)
}

binary_matrix <- convert_to_binary(Wmet)
#head(binary_matrix)

#write.csv(binary_matrix,paste(path,'C4353adjM.csv',sep=''),row.names = T)
#summary(binary_matrix)


#*******************************************************
#step4
#*******************************************************
#library(reshape2)
#。
#
grid01 <- expand.grid(from=rownames(binary_matrix), to=colnames(binary_matrix))
# 
adj_long <- reshape2::melt(binary_matrix)
# 
arc_df <- cbind(grid01, weight = adj_long$value)
# 
arc_df <- arc_df[arc_df$weight != 0, ]
head(arc_df,n=10)

# arc_df <- rbind(tmp,arc_df)
#***********from-->to

modelstring = modstr(fnet = arc_df)
baydag =model2network(modelstring)
graphviz.plot(baydag,shape = "circle" ,main = 'fusion' ,layout = 'dot',
              highlight = list(nodes='label' ,col='blue',fill='yellow'))

###*********************************************###

###*********************************************###
#step5：Model Performance Evaluation

measM <- data.frame(matrix(0,nrow = length(si),ncol =9,
                           dimnames =list(c(st[si]),
                                          c('Sensitivity','Specificity','Precision','Accuracy','F1','AUROC','ROCCI','AUPRC','PRCCI')) ))
prelist <- list()
for(s in 1:length(si)){
  
  train = subset(alltrain,hospitalid==st[si[s]])
  test = subset(alltest,hospitalid==st[si[s]])
  #
  set.seed(100)
  fitted <- bn.fit(baydag, train[,-1],method='bayes')
  pre <- predict(fitted,data=test[,-1],node='label',method ='bayes-lw',prob = TRUE)#
  pred<- attr(pre,'prob')[2,]#
  
  prelist[[s]]<- pred
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
  measM[s,] <- data.frame(Sensitivity,Specificity,Precision,Accuracy,F1,AUROC,ROCCI,AUPRC,PRCCI)
}
measM
colMeans(measM[,1:6]) 

names(prelist) <- st[si]


write.csv(arc_df,paste(path,'K4/C41Net.csv',sep=''))
write.csv(measM,paste(path,'K4/measMC41.csv',sep = ''))
saveRDS(prelist,paste(path,'K4/prelistC41.rds',sep = ''))



