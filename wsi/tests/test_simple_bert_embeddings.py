import unicodedata
import unittest

from bert.simple_bert_embeddings import SimpleBertEmbeddings
from wsi import Word


class TestSimpleBertEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.testable = SimpleBertEmbeddings(bert_model_path="../../data/bert_rus")

    def test_convert_with_same_words(self):
        # given
        text = "Каток выехал на каток."
        # when
        actual = self.testable.convert(text=text)
        print(actual.keys())
        # then
        self.assertTrue(len(actual[Word('каток', 0, 5)]) > 0)
        self.assertTrue(len(actual[Word('выехал', 6, 12)]) > 0)
        self.assertTrue(len(actual[Word('на', 13, 15)]) > 0)
        self.assertTrue(len(actual[Word('каток', 16, 21)]) > 0)
        self.assertTrue(len(actual[Word('.', 21, 22)]) > 0)

    def test_convert_with_sharps(self):
        # given
        text = "Поделись со мной."
        # when
        actual = self.testable.convert(text=text)
        print(actual.keys())
        # then
        self.assertTrue(len(actual[Word('поделись', 0, 8)]) > 0)
        self.assertTrue(len(actual[Word('со', 9, 11)]) > 0)
        self.assertTrue(len(actual[Word('мнои', 12, 16)]) > 0)
        self.assertTrue(len(actual[Word('.', 16, 17)]) > 0)

    def test_convert_special_letters(self):
        # given
        text = "российский в ро́ссии"
        # when
        actual = self.testable.convert(text=text)
        print(actual.keys())
        # then
        self.assertTrue(len(actual[Word('россиискии', 0, 10)]) > 0)
        self.assertTrue(len(actual[Word('в', 11, 12)]) > 0)
        self.assertTrue(len(actual[Word('россии', 13, 19)]) > 0)


if __name__ == '__main__':
    unittest.main()
