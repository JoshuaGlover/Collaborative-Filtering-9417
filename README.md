# Collaborative Filtering: Matrix Factorisation vs Neural Networks

https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223

Model-based
Association algorithms:
In general, we can find frequent itemsets or sequences in all item purchased by users to do frequent itemsets mining and find frequent n itemsets or sequences of related items that meet the support threshold. If the user has purchased some items in the frequent n item set or sequence, then we can recommend other items in the frequent item set or sequence to the user according to certain rating criteria, which may include support, confidence and promotion.
Apriori，FP Tree, PrefixSpan

Clustering:
Using clustering algorithm for collaborative filtering is somewhat similar to the previous collaborative filtering based on users or items. We can cluster by users or by items based on a certain distance measure. If clustering is based on users, users can be divided into different target groups according to a certain distance measurement method, and items with high ratings of the same target group can be recommended to the target users. Based on clustering of items, similar items with high user ratings are recommended to users.
K-Means, BIRCH, DBSCAN, spectral clustering

Classification:
If we divide the ratings into several segments according to the user's ratings, the problem becomes a classification problem. For example, most directly, a rating threshold is set. If the rating is higher than the threshold, then it is recommended. If the rating is lower than the threshold, it is not recommended. We can turn the problem into a dichotomy problem. Although there are many algorithms for classification problems, logical regression is currently the most widely used one. Why is it logical regression rather than support vector machines that seem more promising? Because logistic regression has a strong explanatory capability, we have a definite probability of recommending each item. At the same time, we can engineer the characteristics of the data to achieve the goal of optimization. At present, logistic regression is very mature for collaborative filtering in big tech companies like Netflix and Google.
logical regression and naïve Bayesian

Regression:
It seems more natural to use regression algorithm for collaborative filtering than classification algorithm. Our ratings can be a continuous value instead of a discrete value. Through the regression model, we can get the target user's prediction rating for a certain commodity.
Ridge regression, tree regression and vector-support machine

Matrix factorization:
Matrix factorization is a widely used method for collaborative filtering. Because the traditional singular value factorization SVD requires that the matrix must be dense without missing data, but our user item scoring matrix is a typical sparse matrix, it is more complex to directly use the traditional SVD for collaborative filtering.
SVD, FunkSVD, BiasSVD, SVD++

Neural networks:
Using neural networks and deep learning to do collaborative filtering should be a trend in the future. At present, the most widely-used two-layer neural network is restricted Boltzmann machine (RBM). In the current Netflix algorithm competition, the RBM algorithm performs very well. Of course, it should be even better to use deep neural networks for collaborative filtering, and it should be a trend in the future for tech companies to use deep learning methods for collaborative filtering.
RBM

Graph models:
Using graph models for collaborative filtering, the similarity between users is considered in a graph model. The commonly used algorithms are SimRank algorithm and Markov model algorithm. For SimRank algorithm, its basic idea is that two objects referenced by similar objects also have similarity. The idea of the algorithm is somewhat similar to the famous PageRank. However, Markov model algorithm is based on the Markov chain, and its basic idea is to find the similarity which is difficult to find by ordinary distance measurement algorithm based on conductivity.
SimRank and Markov

NLP:
The hidden semantic model is mainly based on NLP and involves the semantic analysis of user behaviors to make rating recommendations. The main approaches include LSA and LDA
