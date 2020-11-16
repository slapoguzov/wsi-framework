from abc import ABC
from typing import List, Tuple, Dict

import numpy as np

from vector_clustering import VectorsClustering
from word_embeddings import WordEmbeddings
from wsi import Word, Sense


class WordSenseClustering(ABC):
    word_usages: List[Tuple[Word, str]] = []
    word_senses: Dict[str, Dict[Word, Sense]] = []
    word_embeddings: WordEmbeddings
    vectors_clustering: VectorsClustering

    def __init__(self,
                 word_usages: List[Tuple[Word, str]],
                 word_embeddings: WordEmbeddings,
                 vectors_clustering: VectorsClustering) -> None:
        self.word_usages = word_usages
        self.word_embeddings = word_embeddings
        self.vectors_clustering = vectors_clustering
        self.word_senses = self.resolve()
        super().__init__()

    def get_sense(self, word: Word, text: str) -> Sense:
        return self.word_senses[text][word]

    def resolve(self) -> Dict[str, Dict[Word, Sense]]:
        words_texts_vectors = [(word, text, self.get_word_vector(word, text)) for word, text in self.word_usages]
        vectors = [(i, vector) for i, (_, _, vector) in enumerate(words_texts_vectors)]
        cluster_groups = self.vectors_clustering.fit(vectors)
        result: Dict[str, Dict[Word, Sense]] = {}
        for i, group_id in cluster_groups:
            (word, text, _) = words_texts_vectors[i]
            if text in result:
                result[text].update({word: Sense(group_id, text)})
            else:
                result.update({text: {word: Sense(group_id, text)}})
        return result

    def get_word_vector(self, word: Word, text: str) -> List[float]:
        vectors = self.word_embeddings.convert(text)
        if word in vectors:
            return vectors[word]
        for key in vectors.keys():
            if word.start < key.end and word.end > key.start:
                return vectors[key]

        return []

    def get_mean_vector(self, text: str) -> List[float]:
        vectors = self.word_embeddings.convert(text)
        return np.mean(list(vectors.values()), axis=0)