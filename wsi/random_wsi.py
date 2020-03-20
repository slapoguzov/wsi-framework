import random

from wsi.wsi import WordSenseInduction, Word, Sense


class RandomWsi(WordSenseInduction):
    def resolve(self, word: Word, text: str) -> Sense:
        return Sense(random.randint(0, 100), text)
