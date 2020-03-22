from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: int
    end: int

    def __key(self):
        return self.text, self.start, self.end

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Word):
            return self.__key() == other.__key()
        return NotImplemented


@dataclass
class Sense:
    id: int
    description: str = ""


class WordSenseInduction(ABC):
    @abstractmethod
    def resolve(self, word: Word, text: str) -> Sense:
        """
        finds sense of the word in the text
        :param word: the word
        :param text: the text that contains the word
        """
        pass
