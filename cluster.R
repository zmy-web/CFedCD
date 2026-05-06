
rm(list = ls())
library(cluster)
library(factoextra)
library(klaR)
dt <- read.csv('.../DigestVector.csv')
dim(dt)
# 数据进行标准化
df <- scale(dt) 
# 查看数据的前五行
head(df, n = 5)

#确定最佳聚类数目

c1<- fviz_nbclust(df, kmeans, method = "silhouette") + #c("silhouette", "wss", "gap_stat"),
  geom_vline(xintercept = 4, linetype = 2,color='red')+
  theme(plot.title=element_text(hjust=0.5))+
  labs(title = 'silhouette')#最大化
c2<- fviz_nbclust(df, kmeans, method = "wss") + 
  geom_vline(xintercept = 4, linetype = 2,color='red')+
  theme(plot.title=element_text(hjust=0.5))+
  labs(title = 'wss')

c3<- fviz_nbclust(df, kmeans, method = "gap_stat") + 
  geom_vline(xintercept = 4, linetype = 2,color='red')+
  theme(plot.title=element_text(hjust=0.5))+
  labs(title = 'gap_stat')#最大化


library(cowplot)

plot_grid(c1,c2,c3,ncol = 3,labels = LETTERS[1:3])


#可以发现聚为四类最合适，当然这个没有绝对的，从指标上看，选择坡度变化不明显的点最为最佳聚类数目。
#设置随机数种子，保证实验的可重复进行
set.seed(100)
#利用k-mean是进行聚类
K=4
km_result <- kmeans(df,centers= K, nstart = 100)
#查看聚类的一些结果
print(km_result)
#提取类标签并且与原始数据进行合并
dd <- cbind(dt, cluster = km_result$cluster)

#查看每一类的数目
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
  theme(plot.title=element_text(hjust=0.5))  #这一句让标题居中，ggplot2默认的标题是左对齐的
p4




