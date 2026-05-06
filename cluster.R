
rm(list = ls())
library(cluster)
library(factoextra)
library(klaR)
dt <- read.csv('.../DigestVector.csv')
dim(dt)

df <- scale(dt) 

c1<- fviz_nbclust(df, kmeans, method = "silhouette") + #c("silhouette", "wss", "gap_stat"),
  geom_vline(xintercept = 4, linetype = 2,color='red')+
  theme(plot.title=element_text(hjust=0.5))+
  labs(title = 'silhouette')#
c2<- fviz_nbclust(df, kmeans, method = "wss") + 
  geom_vline(xintercept = 4, linetype = 2,color='red')+
  theme(plot.title=element_text(hjust=0.5))+
  labs(title = 'wss')

c3<- fviz_nbclust(df, kmeans, method = "gap_stat") + 
  geom_vline(xintercept = 4, linetype = 2,color='red')+
  theme(plot.title=element_text(hjust=0.5))+
  labs(title = 'gap_stat')#


library(cowplot)

plot_grid(c1,c2,c3,ncol = 3,labels = LETTERS[1:3])


set.seed(100)

K=4
km_result <- kmeans(df,centers= K, nstart = 100)

print(km_result)

dd <- cbind(dt, cluster = km_result$cluster)


#table(dd$cluster)
dd$cluster
#进行可视化展示
cid <- data.frame(sid=rownames(dt),ci=dd$cluster)
head(cid)
C1 =which(cid$ci==1)
C2 =which(cid$ci==2)
C3 =which(cid$ci==3)
C4 =which(cid$ci==4)



p4 <-fviz_cluster(km_result, data = df,
                  palette = rainbow(8)[1:4],#rainbow(K),
                  ellipse.type = "euclid",
                  star.plot = TRUE, 
                  repel = TRUE,
                  #ggtheme = theme_minimal(),
                  main = paste('cluster K=',K,sep = ''),
                  xlab = ,
                  ylab = 
)+theme_classic()+
  #theme_minimal() +
  theme(plot.title=element_text(hjust=0.5))  #
p4




