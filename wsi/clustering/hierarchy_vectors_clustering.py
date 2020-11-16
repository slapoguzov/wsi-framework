from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import pca

from clustering.utils import show_two_dimensions_plot
from vector_clustering import VectorsClustering


class HierarchyVectorsClustering(VectorsClustering):

    def fit(self, vectors: [int, float]) -> [int, int]:
        vectors_ = list(zip(*vectors))[1]
        cluster_model = AgglomerativeClustering(n_clusters=2, affinity="cosine", linkage="complete")
        cluster = cluster_model.fit_predict(vectors_)

        show_two_dimensions_plot(vectors_, cluster)

        return [(i, label) for i, label in enumerate(cluster)]

    def fit_kmeans(self, vectors: [int, float]) -> [int, int]:
        vectors_ = list(zip(*vectors))[1]
        x_pca = pca.fit_transform(vectors_)
        kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=0)
        cluster = kmeans.fit_predict(x_pca)

        show_two_dimensions_plot(vectors_, cluster)

        return [(i, label) for i, label in enumerate(cluster)]
