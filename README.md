# CFedCD
Clustering-Based Federated Causal Discovery for Multicenter Clinical Data Analysis

The CFedCD framework integrates advanced representation learning and federated optimization techniques to address inherent data heterogeneity and privacy constraints in distributed causal learning tasks. Each client independently extracts high-dimensional feature summaries from local Electronic Medical Records (EMRs) using a deep ensemble model, which captures complex data distributions while preserving privacy. These locally computed summaries are then aggregated at the server side, where K-means clustering is applied to group clients with similar data characteristics into federated clusters. Within each cluster, the collaborative construction of cluster-specific causal graphs is facilitated through adaptive aggregation strategies and regularization techniques to mitigate the impact of distribution shifts.

step1: The Deep Sets model is utilized to extract digest descriptors from the discretized local clients’ training data in a privacy-preserving manner. 

step2: The primary objective of the clustering phase is to partition the highly heterogeneous multicenter dataset into sub-populations, where the data distribution within each cluster exhibits higher consistency, thereby facilitating more reliable structure learning. 

step3:Within each cluster, clients are approximated as having homogeneous distributions (representing clinically relevant patient subpopulations), enabling the server to coordinate structural learning on the original discretized data for the collaborative construction of a personalized federated directed graph.
