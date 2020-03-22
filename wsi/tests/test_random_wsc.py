import unittest

from trivial_impl.random_wsc import RandomWsc
from wsi import Word


class TestRandomWsc(unittest.TestCase):

    def test_resolve(self):
        # given
        usage1 = (Word("one", 0, 2), "one two three")
        usage2 = (Word("w3", 6, 7), "w1 w2 w3")
        self.testable = RandomWsc([usage1, usage2])
        # when
        sense1 = self.testable.get_sense(usage1[0], usage1[1])
        sense2 = self.testable.get_sense(usage2[0], usage2[1])
        # then
        self.assertEqual(sense1.description, usage1[1])
        self.assertTrue(sense1.id > 0)

        self.assertEqual(sense2.description, usage2[1])
        self.assertTrue(sense2.id > 0)


if __name__ == '__main__':
    unittest.main()
