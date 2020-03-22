from abc import abstractmethod, ABC


class VectorsClustering(ABC):
    @abstractmethod
    def fit(self, vectors: [int, float]) -> [int, int]:
        """ groups vectors by clusters """
        pass

