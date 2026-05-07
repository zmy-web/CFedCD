# CFedCD
Clustering-Based Federated Causal Discovery for Multicenter Clinical Data Analysis

The CFedCD framework integrates advanced representation learning and federated optimization techniques to address inherent data heterogeneity and privacy constraints in distributed causal learning tasks. Each client independently extracts high-dimensional feature summaries from local Electronic Medical Records (EMRs) using a deep ensemble model, which captures complex data distributions while preserving privacy. These locally computed summaries are then aggregated at the server side, where K-means clustering is applied to group clients with similar data characteristics into federated clusters. Within each cluster, the collaborative construction of cluster-specific causal graphs is facilitated through adaptive aggregation strategies and regularization techniques to mitigate the impact of distribution shifts.

This repository provides a self-contained, end-to-end pipeline for the CFedCD framework. The workflow is decoupled into four main stages: privacy-preserving feature extraction, server-side clustering, local structure learning, and federated structural fusion.

############################################################################

Repository Structure & Key Scripts
DeepSets.py: A deterministic structural encoder used to generate client-level "digest vectors."

cluster.R: The server-side implementation of the K-means clustering algorithm for client grouping.

localCDmodel.R: An automated local causal discovery script that generates adjacency matrices and structural scores.

ClusterFLCD.R: The federated aggregation engine that coordinates cluster-specific consensus.

CBAMN.py: The Cycle-Breaking Algorithm based on Modified NOTEARS, used for structural optimization and DAG enforcement.

############################################################################

Step 1: Privacy-Preserving Digest Extraction
Script: DeepSets.py
Description: Each client independently processes their discretized local Electronic Medical Records (EMRs). The Deep Sets model maps these sets into a latent space to produce digest vectors.

Step 2: Server-Side Client Clustering
Script: cluster.R
Description: The server aggregates all client-level digest vectors. It executes K-means clustering to partition the heterogeneous clients into distributionally consistent sub-population clusters.The primary objective of the clustering phase is to partition the highly heterogeneous multicenter dataset into sub-populations, where the data distribution within each cluster exhibits higher consistency, thereby facilitating more reliable structure learning. 

Step 3: Local Causal Structure Learning
Script: localCDmodel.R
Description: Each client runs this script on their raw local data to automatically perform causal structure learning and output the standardized adjacency matrices, edge strengths, and network scores.

Step 4: Federated Fusion & DAG Optimization
Script: ClusterFLCD.R & CBAMN.py
Description: Within each cluster, the server performs a weighted aggregation of local DAGs based on their quality and sample size. Since the superposition of DAGs may introduce cycles, the CBAMN algorithm is applied to the cluster-specific Weighted Adjacency Matrix (WAM).


