from abc import abstractmethod
from typing import List, Dict

from pandas import np
from sklearn.decomposition import PCA

from word_embeddings import WordEmbeddings
from wsi import Word


class EmbeddingsDimensionReducer(WordEmbeddings):
    delegate: WordEmbeddings
    dimensions: int = 10
    pca: PCA = {}

    def __init__(self, delegate: WordEmbeddings, dimensions: int = 10):
        self.delegate = delegate
        self.dimensions = dimensions
        self.pca = PCA(n_components=dimensions)

    def convert(self, text: str) -> Dict[Word, List[float]]:
        embeddings = self.delegate.convert(text)
        vectors = enumerate(embeddings.values())
        vectors_ = list(zip(*vectors))[1]
        transformed = {}
        if len(vectors_) < self.dimensions:
            transformed = np.array(vectors_)
            transformed.resize(len(vectors_), self.dimensions, refcheck=False)
        else:
            transformed = self.pca.fit_transform(vectors_)
        result: Dict[Word, List[float]] = {}
        for i, (word, _) in enumerate(embeddings.items()):
            result[word] = transformed[i]
        return result

