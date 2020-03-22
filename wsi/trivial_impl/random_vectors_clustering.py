import numpy as np
from nltk.tokenize import TreebankWordTokenizer

from vector_clustering import VectorsClustering


class RandomVectorsClustering(VectorsClustering):
    VECTOR_SIZE = 768
    twt = TreebankWordTokenizer()

    def fit(self, vectors: [int, float]) -> [int, int]:
        return [(i, np.random.randint(1, 3)) for i, _ in vectors]

