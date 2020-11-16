import itertools
from abc import ABC
from typing import List, Tuple, Dict

import numpy as np

from vector_clustering import VectorsClustering
from word_embeddings import WordEmbeddings
from wsi import Word, Sense


class WordSenseClustering(ABC):
    word_usages: Dict[str, List[Word]] = {}
    word_senses: Dict[str, Dict[Word, Sense]] = []
    word_embeddings: WordEmbeddings
    vectors_clustering: VectorsClustering

    def __init__(self,
                 word_usages: Dict[str, List[Word]],
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
        words_texts_vectors = [(words, text, self.word_embeddings.convert(text))
                               for text, words in self.word_usages.items()]
        words_vectors = [(text_id, word, self.get_word_vector(word, vectors))
                         for text_id, (words, _, vectors) in enumerate(words_texts_vectors)
                         for word in words]
        result: Dict[str, Dict[Word, Sense]] = {}
        for word_text, w_v in itertools.groupby(words_vectors, lambda w_v: w_v[1].text):
            grouped_words_vectors = list(w_v)
            vectors = [(i, vector) for i, (_, _, vector) in enumerate(grouped_words_vectors)]
            cluster_groups = self.vectors_clustering.fit(vectors)
            for i, group_id in cluster_groups:
                (text_id, word, _) = grouped_words_vectors[i]
                (_, text, _) = words_texts_vectors[text_id]
                print("[wsc] assign", word, " id =", i, "to group_id", group_id, "in text", text)
                if text in result:
                    result[text].update({word: Sense(group_id, text)})
                else:
                    result.update({text: {word: Sense(group_id, text)}})
        return result

    @staticmethod
    def get_word_vector(word: Word, vectors: Dict[Word, List[float]]) -> List[float]:
        if word in vectors:
            return vectors[word]
        for key in vectors.keys():
            if word.start < key.end and word.end > key.start:
                return vectors[key]

        return []

    def get_mean_vector(self, text: str) -> List[float]:
        vectors = self.word_embeddings.convert(text)
        return np.mean(list(vectors.values()), axis=0)