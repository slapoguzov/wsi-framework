from wsc import WordSenseClustering
from wsi import WordSenseInduction, Word, Sense


class WsiBasedWsc(WordSenseInduction):
    cluster: WordSenseClustering

    def fit(self, cluster: WordSenseClustering):
        self.cluster = cluster

    def resolve(self, word: Word, text: str) -> Sense:
        return self.cluster.get_sense(word, text)