from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: int
    end: int


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
