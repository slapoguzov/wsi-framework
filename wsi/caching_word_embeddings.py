import hashlib
import json
from typing import List, Dict

from numpy import ndarray

from word_embeddings import WordEmbeddings
from wsi import Word


class CachingWordEmbeddings(WordEmbeddings):
    cache_path: str = ""
    cache_file = {}
    cache: Dict[str, Dict[Word, List[float]]] = {}
    delegate: WordEmbeddings = {}

    def __init__(self, delegate: WordEmbeddings, cache_path: str = "word_embeddings_cache"):
        self.delegate = delegate
        self.cache_file = open(cache_path, mode='a+', encoding='utf-8')
        self.cache_file.seek(0)
        self.cache = self.read_cache()

    def convert(self, text: str) -> Dict[Word, List[float]]:
        if not self.contains_in_cache(text):
            self.write_to_cache(text, self.delegate.convert(text))
        return self.read_from_cache(text)

    def write_to_cache(self, text: str, vectors: Dict[Word, List[float]]):
        value = json.dumps(tuple(vectors.items()), cls=WordEncoder)
        key = self.get_hash(text)
        print("write to cache", key, text)
        self.cache_file.write(f'{key} {value}\n')
        self.cache[key] = vectors
        pass

    def read_from_cache(self, text: str) -> Dict[Word, List[float]]:
        return self.cache[self.get_hash(text)]

    def contains_in_cache(self, text: str) -> bool:
        return self.cache.keys().__contains__(self.get_hash(text))

    @staticmethod
    def get_hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def read_cache(self) -> Dict[str, Dict[Word, List[float]]]:
        result: Dict[str, Dict[Word, List[float]]] = {}
        for line in self.cache_file.readlines():
            (key, value) = line.split(" ", 1)
            result[key] = dict(json.loads(value, object_hook=as_word))
        return result


class WordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Word):
            return {"__type__": "Word", 'text': obj.text, 'start': obj.start, 'end': obj.end}
        if isinstance(obj, ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def as_word(dct):
    if '__type__' in dct and dct['__type__'] == 'Word':
        return Word(dct['text'], dct['start'], dct['end'])
    return dct
