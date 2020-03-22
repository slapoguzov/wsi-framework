from abc import abstractmethod, ABC
from typing import List, Dict

from wsi import Word


class WordEmbeddings(ABC):
    @abstractmethod
    def convert(self, text: str) -> Dict[Word, List[float]]:
        """ converts the text to vectors """
        pass

