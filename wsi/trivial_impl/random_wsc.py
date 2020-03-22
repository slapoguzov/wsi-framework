from typing import List, Tuple

from trivial_impl.random_vectors_clustering import RandomVectorsClustering
from trivial_impl.random_word_embeddings import RandomWordEmbeddings
from wsc import WordSenseClustering
from wsi import Word


class RandomWsc(WordSenseClustering):
    def __init__(self, word_usages: List[Tuple[Word, str]]) -> None:
        self.word_embeddings = RandomWordEmbeddings()
        self.vectors_clustering = RandomVectorsClustering()
        super(RandomWsc, self).__init__(word_usages, RandomWordEmbeddings(), RandomVectorsClustering())
