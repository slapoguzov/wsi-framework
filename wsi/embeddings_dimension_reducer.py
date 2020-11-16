from abc import abstractmethod
from typing import List, Dict

from sklearn.decomposition import PCA

from word_embeddings import WordEmbeddings
from wsi import Word


class EmbeddingsDimensionReducer(WordEmbeddings):
    delegate: WordEmbeddings
    pca: PCA = {}

    def __init__(self, delegate: WordEmbeddings, dimensions: int = 10):
        self.delegate = delegate
        self.pca = PCA(n_components=dimensions)

    def convert(self, text: str) -> Dict[Word, List[float]]:
        embeddings = self.delegate.convert(text)
        vectors = enumerate(embeddings.values())
        vectors_ = list(zip(*vectors))[1]
        transformed = self.pca.fit_transform(vectors_)
        result: Dict[Word, List[float]] = {}
        for i, (word, _) in enumerate(embeddings.items()):
            result[word] = transformed[i]
        return result

