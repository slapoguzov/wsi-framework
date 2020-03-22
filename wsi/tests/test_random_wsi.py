import unittest

from trivial_impl.random_wsi import RandomWsi
from wsi import Word


class TestRandomWsi(unittest.TestCase):
    def setUp(self) -> None:
        self.testable = RandomWsi()

    def test_resolve(self):
        # given
        word = Word('one', 0, 2)
        text = "one two three"
        # when
        actual = self.testable.resolve(word=word, text=text)
        # then
        self.assertTrue(actual.id > 0)


if __name__ == '__main__':
    unittest.main()
