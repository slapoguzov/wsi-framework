from abc import ABC
from typing import List, Tuple, Dict

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
        self.word_senses = self.resolve()
        self.word_embeddings = word_embeddings
        self.vectors_clustering = vectors_clustering
        self.resolve()
        super().__init__()

    def get_sense(self, word: Word, text: str) -> Sense:
        return self.word_senses[text][word]

    def resolve(self) -> Dict[str, Dict[Word, Sense]]:
        words_texts_vectors = [(word, text, self.word_embeddings.convert(text)) for word, text in self.word_usages]
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
