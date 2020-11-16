from pandas import np
from sklearn.cluster import DBSCAN

from clustering.utils import show_two_dimensions_plot
from vector_clustering import VectorsClustering


class DBSCANClustering(VectorsClustering):

    def fit(self, vectors: [int, float], eps=39, min_samples=6) -> [int, int]:
        vectors_ = list(zip(*vectors))[1]
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster = cluster_model.fit_predict(vectors_)

        show_two_dimensions_plot(vectors_, cluster, "eps=" + str(eps) + " min_samples=" + str(min_samples))

        #self.search(vectors)

        return [(i, label) for i, label in enumerate(cluster)]

    def search(self, vectors: [int, float]):
        for eps in np.logspace(1.4, 1.8, 20):
            for min_samples in [4, 5, 6]:
                vectors_ = list(zip(*vectors))[1]
                cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
                cluster = cluster_model.fit_predict(vectors_)
                print(eps, min_samples)
                show_two_dimensions_plot(vectors_, cluster, "eps=" + str(eps) + " min_samples=" + str(min_samples))
