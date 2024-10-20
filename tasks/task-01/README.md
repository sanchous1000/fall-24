# Laboratory work No. 1.Clastorization

As part of the laboratory work, various clustering algorithms will have to be implemented.

The following algorithms were considered at the lecture:
* Hierarchical clustering
* Statistical clustering
* EM algorithm
* K -means
* Dbscan
* Network of Kohonen

## Exercise

1. Choose datasets for clustering (2 pcs), for example, on [kaggle] (https://www.kaggle.com/datasets?&tags=13304-Clustering); \
1.1.Remove the classes label from the data, if any; \
1.2.visualize data; \
1.3.Determine the "type" of clusters; \
1.4.form a hypothesis about the number of clusters;
2. To implement a hierarchical algorithm; \
2.1.Build a dendrogram for each Dataset; \
2.2.determine the optimal number of clusters for each dataset;
3. To realize algorithms:
* EM algorithm;
* Dbscan;
4. For each clustering algorithm and for each dataset:
* Based on the results of clustering, calculate the average intraclaster distance;
* Based on the results of clustering, calculate the average interclature distance;
* measure the speed of clustering;
5. For algorithms of hierarchical clustering, statistical algorithms and dbscan, take [reference] (https://scikit-learn.org/stable/) implementation and calculate metrics on selected datasets:
* average intraclaster distance;
* average interclature distance;
* clustering speed;
6. Show comparisons of metrics on developed and reference algorithms.