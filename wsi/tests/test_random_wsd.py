import unittest

from wsi.random_wsi import RandomWsi
from wsi.wsi import Word


class TestRandomWsd(unittest.TestCase):
    def setUp(self) -> None:
        self.testable = RandomWsi()

    def test_resolve(self):
        # given
        word = Word("one", 0, 2)
        text = "one two three"
        # when
        actual = RandomWsi().resolve(word=word, text=text)
        # then
        self.assertEqual(actual.description, text)
        self.assertTrue(actual.id > 0)


if __name__ == '__main__':
    unittest.main()
