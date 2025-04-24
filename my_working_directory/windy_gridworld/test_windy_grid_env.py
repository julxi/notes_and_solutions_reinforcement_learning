import unittest
import numpy as np
from windy_grid import WindyGrid
from windy_grid_env import WindyGridEnv


class TestSnapToGrid(unittest.TestCase):

    def test_normal_step(self):
        # given
        windy_grid = WindyGrid(2, 2, [0, 0], np.array([0, 0]), np.array([0, 0]))
        windy_grid_env = WindyGridEnv(windy_grid, delta_max=1, wind_var=0)
        windy_grid_env.set_state(np.array([0, 0]))

        # when
        action = np.array([1, 1])
        result = windy_grid_env.take_action(action)

        # then
        self.assertCountEqual(result, np.array([1, 1]))

    def test_step_with_wind(self):
        # given
        windy_grid = WindyGrid(2, 2, [0, 1], np.array([0, 0]), np.array([0, 0]))
        windy_grid_env = WindyGridEnv(windy_grid, wind_var=0)
        windy_grid_env.set_state(np.array([0, 0]))

        # when
        action = np.array([1, 0])
        result = windy_grid_env.take_action(action)

        # then
        self.assertCountEqual(result, np.array([1, 1]))

    def test_step_with_wind_and_snapping(self):
        # given
        windy_grid = WindyGrid(2, 2, [0, 3], np.array([0, 0]), np.array([0, 0]))
        windy_grid_env = WindyGridEnv(windy_grid, wind_var=0)
        windy_grid_env.set_state(np.array([1, 0]))

        # when
        action = np.array([1, 0])
        result = windy_grid_env.take_action(action)

        # then
        self.assertCountEqual(result, np.array([1, 1]))


class TestCalculateActionCode(unittest.TestCase):

    def test_back_and_forth(self):
        # given
        windy_grid = WindyGrid(2, 2, [0, 3], np.array([0, 0]), np.array([0, 0]))
        windy_grid_env = WindyGridEnv(windy_grid, delta_max=3)
        # when + then
        for i in range((2 * windy_grid_env.delta_max + 1) ** 2):
            action = windy_grid_env.calculate_code_to_action(i)
            code = windy_grid_env.calculate_action_to_code(action)
            self.assertEqual(code, i)


class TestProperEnvironment(unittest.TestCase):

    def test_walk(self):
        # given
        windy_grid = WindyGrid(
            width=10,
            height=7,
            wind_speeds=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
            start=np.array([0, 3]),
            end=np.array([7, 3]),
        )

        windy_grid_env = WindyGridEnv(windy_grid, delta_max=1, wind_var=0)

        x,y = windy_grid_env.reset()

        # when
        new_pos = windy_grid_env.take_action(np.array([1, -1]))
        
        # then
        np.testing.assert_array_equal(new_pos, np.array([1, 2]))






if __name__ == "__main__":
    unittest.main()
