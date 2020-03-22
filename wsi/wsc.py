from abc import abstractmethod, ABC
from typing import List, Tuple, Dict

from trivial_impl.random_vectors_clustering import RandomVectorsClustering
from trivial_impl.random_word_embeddings import RandomWordEmbeddings
from vector_clustering import VectorsClustering
from word_embeddings import WordEmbeddings
from wsi import Word, Sense


class WordSenseClustering(ABC):
    word_usages: List[Tuple[Word, str]] = []
    word_senses: Dict[Word, Dict[str, Sense]] = []
    word_embeddings: WordEmbeddings
    vectors_clustering: VectorsClustering

    def __init__(self,
                 word_usages: List[Tuple[Word, str]],
                 word_embeddings: WordEmbeddings = RandomWordEmbeddings(),
                 vectors_clustering: VectorsClustering = RandomVectorsClustering()) -> None:
        self.word_usages = word_usages
        self.word_senses = self.resolve()
        self.word_embeddings = word_embeddings
        self.vectors_clustering = vectors_clustering
        self.resolve()
        super().__init__()

    def get_sense(self, word: Word, text: str) -> Sense:
        return self.word_senses[word][text]

    def resolve(self) -> Dict[Word, Dict[str, Sense]]:
        words_texts_vectors = [(word, text, self.word_embeddings.convert(text)) for word, text in self.word_usages]
        vectors = [(i, vector) for i, (_, _, vector) in enumerate(words_texts_vectors)]
        cluster_groups = self.vectors_clustering.fit(vectors)
        result: Dict[Word, Dict[str, Sense]] = {}
        for i, group_id in cluster_groups:
            (word, text, _) = words_texts_vectors[i]
            result.update({word: {text: Sense(group_id, text)}})
        return result
