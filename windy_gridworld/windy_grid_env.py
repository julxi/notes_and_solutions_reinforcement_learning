import numpy as np
from windy_grid import WindyGrid


class WindyGridEnv:

    def __init__(self, windy_grid, delta_max=1, wind_var=0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.windy_grid = windy_grid
        self.delta_max = delta_max
        self.wind_var = wind_var
        self.state = np.array([0, 0])

        # action codes
        self.action_count = (2 * delta_max + 1) ** 2
        self.actions = self.calculate_action_list()

    def is_terminal(self, state):
        return np.array_equal(state, self.windy_grid.end)

    def reset(self):
        self.state = self.windy_grid.start.copy()
        return self.state

    def set_state(self, state):
        self.state = state

    def calculate_code_to_action(self, a):
        k, l = divmod(a, 2 * self.delta_max + 1)
        return np.array([-self.delta_max + k, -self.delta_max + l])

    def calculate_action_to_code(self, action):
        x, y = action
        return (x + self.delta_max) * (2 * self.delta_max + 1) + (y + self.delta_max)

    def calculate_action_list(self):
        return np.array(
            [self.calculate_code_to_action(i) for i in range(self.action_count)]
        )

    def code_to_action(self, a):
        return self.actions[a]

    def take_action(self, action):

        a = self.calculate_action_to_code(action)
        return self.take_action_code(a)

    def take_action_code(self, a):
        """
        1. snapped move
        2. snapped add wind
        """
        action = self.code_to_action(a)
        after_move = self.windy_grid.snap_to_grid(self.state + action)

        wind_strength = self.windy_grid.wind_speeds[after_move[0]]
        if self.wind_var != 0:
            wind_strength += np.random.randint(-self.wind_var, self.wind_var + 1)
        wind = np.array([0, wind_strength])

        self.state = self.windy_grid.snap_to_grid(after_move + wind)
        return self.state
