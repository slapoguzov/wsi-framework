import scipy.cluster.hierarchy as hac
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from vector_clustering import VectorsClustering


class HierarchyVectorsClustering(VectorsClustering):

    def fit(self, vectors: [int, float]) -> [int, int]:
        pca = PCA(n_components=2)
        vectors_ = list(zip(*vectors))[1]
        x_pca = pca.fit_transform(vectors_)
        clusterModel = AgglomerativeClustering(n_clusters=2, affinity="cosine", linkage="complete")
        cluster = clusterModel.fit_predict(vectors_)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = x_pca[:, 0]
        y = x_pca[:, 1]

        ax.scatter(x, y, c=cluster)

        for i, txt in enumerate(x):
            ax.annotate(str(i), (x[i], y[i]))


        plt.show()


        return [(i, label) for i, label in enumerate(cluster)]

    #kmeans
    def fit_kmeans(self, vectors: [int, float]) -> [int, int]:
        pca = PCA(n_components=2)
        vectors_ = list(zip(*vectors))[1]
        x_pca = pca.fit_transform(vectors_)
        kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=0)
        cluster = kmeans.fit_predict(x_pca)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = x_pca[:, 0]
        y = x_pca[:, 1]
        # points = x_pca[:, 2:4]
        # # color is the length of each vector in `points`
        # color = np.sqrt((points ** 2).sum(axis=1)) / np.sqrt(2.0)
        # rgb = plt.get_cmap('jet')(color)

        ax.scatter(x, y, c=cluster)

        for i, txt in enumerate(x):
            ax.annotate(str(i), (x[i], y[i]))


        plt.show()


        return [(i, label) for i, label in enumerate(cluster)]

