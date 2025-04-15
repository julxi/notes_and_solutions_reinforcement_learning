import unittest
import numpy as np
from windy_grid import WindyGrid


class TestSnapToGrid(unittest.TestCase):

    def test_negative_corner(self):
        # given
        windy_grid = WindyGrid(2, 2, [0, 0], np.array([0, 0]), np.array([0, 0]))

        # when
        result = windy_grid.snap_to_grid(np.array([-1, -1]))

        # then
        self.assertCountEqual(result, np.array([0, 0]))

    def test_negative_corner(self):
        # given
        windy_grid = WindyGrid(2, 3, [0, 0, 0], np.array([0, 0]), np.array([0, 0]))

        # when
        result = windy_grid.snap_to_grid(np.array([2, 3]))

        # then
        self.assertCountEqual(result, np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
