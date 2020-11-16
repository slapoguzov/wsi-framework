from sklearn.cluster import AffinityPropagation

from clustering.utils import show_two_dimensions_plot
from vector_clustering import VectorsClustering


class AffinityPropagationClustering(VectorsClustering):

    def fit(self, vectors: [int, float]) -> [int, int]:
        vectors_ = list(zip(*vectors))[1]
        cluster_model = AffinityPropagation(damping=0.96, max_iter=10000, convergence_iter=15)
        cluster = cluster_model.fit_predict(vectors_)

        show_two_dimensions_plot(vectors_, cluster)

        return [(i, label) for i, label in enumerate(cluster)]
