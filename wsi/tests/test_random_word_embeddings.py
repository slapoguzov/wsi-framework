import unittest

from trivial_impl.random_word_embeddings import RandomWordEmbeddings
from wsi import Word


class TestRandomWordEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.testable = RandomWordEmbeddings()

    def test_resolve(self):
        # given
        word = Word("one", 0, 2)
        text = "one two one"
        # when
        actual = self.testable.convert(text=text)
        # then
        self.assertTrue(len(actual[Word('one', 0, 3)]) > 0)
        self.assertTrue(len(actual[Word('two', 4, 7)]) > 0)
        self.assertTrue(len(actual[Word('one', 8, 11)]) > 0)


if __name__ == '__main__':
    unittest.main()
