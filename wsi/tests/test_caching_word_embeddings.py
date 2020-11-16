import os
import unittest

from caching_word_embeddings import CachingWordEmbeddings
from trivial_impl.random_word_embeddings import RandomWordEmbeddings
from wsi import Word


class TestCachingWordEmbeddings(unittest.TestCase):
    def setUp(self):
        self.cache_file = open("test_cache", "a+")

    def tearDown(self):
        self.cache_file.close()
        #os.remove("test_cache")

    def test_1(self):
        # given
        delegate = RandomWordEmbeddings()
        testable = CachingWordEmbeddings(delegate, os.path.abspath(self.cache_file.name))
        text = "Каток выехал на каток."
        # when
        actual = testable.convert(text=text)
        print(actual.keys())
        # then
        self.assertTrue(len(actual[Word('Каток', 0, 5)]) > 0)
        self.assertTrue(len(actual[Word('выехал', 6, 12)]) > 0)
        self.assertTrue(len(actual[Word('на', 13, 15)]) > 0)
        self.assertTrue(len(actual[Word('каток', 16, 21)]) > 0)
        self.assertTrue(len(actual[Word('.', 21, 22)]) > 0)

if __name__ == '__main__':
    unittest.main()
