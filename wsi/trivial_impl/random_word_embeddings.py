from typing import List, Dict

import numpy as np
from nltk.tokenize import TreebankWordTokenizer

from word_embeddings import WordEmbeddings
from wsi import Word


class RandomWordEmbeddings(WordEmbeddings):
    VECTOR_SIZE = 768
    twt = TreebankWordTokenizer()

    def convert(self, text: str) -> Dict[Word, List[float]]:
        spans = self.twt.span_tokenize(text)
        result: Dict[Word, List[float]] = {}

        for start, end in spans:
            token = text[start:end]
            result[Word(token, start, end)] = np.float_(np.random.rand(self.VECTOR_SIZE))
        return result

