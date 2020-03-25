import scipy.cluster.hierarchy as hac

from vector_clustering import VectorsClustering


class HierarchyVectorsClustering(VectorsClustering):

    def fit(self, vectors: [int, float]) -> [int, int]:
        cluster = hac.fclusterdata([vec for i, vec in vectors], 0.02, criterion="distance")
        return [(i, label) for i, label in enumerate(cluster)]

