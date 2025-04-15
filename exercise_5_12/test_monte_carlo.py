from race_track import RaceTrack
from race_track_environment import RaceTrackEnv, State, Action
from monte_carlo import MonteCarloESEveryVisit


import unittest
import numpy as np


class TestRaceTrackEnv(unittest.TestCase):

    def test_start_episode_in_terminal_state(self):
        # given
        track_ascii = """
F
"""
        race_track = RaceTrack(track_ascii)
        env = RaceTrackEnv(race_track, max_velocity=1, max_acceleration=1)
        policy = []
        evaluation = []
        monte_carlo = MonteCarloESEveryVisit(env, policy, evaluation, 1)

        # when
        ep_states, ep_actions = monte_carlo.generate_episode(0, 0)

        # then
        self.assertEqual(ep_states, [0])
        self.assertEqual(ep_actions, [])

    def test_episode_ends_in_terminal_state(self):
        # given
        track_ascii = """
    F
    S
    """
        race_track = RaceTrack(track_ascii)
        env = RaceTrackEnv(race_track, max_velocity=1, max_acceleration=1)
        policy = []
        evaluation = []
        monte_carlo = MonteCarloESEveryVisit(env, policy, evaluation, 10)

        # when
        ep_states, ep_actions = monte_carlo.generate_episode(0, 4)

        # then
        self.assertEqual(ep_states, [0, 9])
        self.assertEqual(ep_actions, [4])

    def test_policy_ends_in_terminal_state(self):
        # given
        track_ascii = """
F
-
S
"""
        race_track = RaceTrack(track_ascii)
        env = RaceTrackEnv(race_track, max_velocity=1, max_acceleration=1)
        policy = env.actions_per_state
        policy[13] = 0
        evaluation = []
        monte_carlo = MonteCarloESEveryVisit(env, policy, evaluation, 10)

        # when
        ep_states, ep_actions = monte_carlo.generate_episode(0, 4)

        # then
        self.assertEqual(ep_states, [0, 13, 17])
        self.assertEqual(ep_actions, [4, 0])

    def test_episode_runs_too_long(self):
        # given
        track_ascii = """
-
S
"""
        race_track = RaceTrack(track_ascii)
        env = RaceTrackEnv(race_track, max_velocity=1, max_acceleration=2)
        policy = env.actions_per_state
        evaluation = []

        state = State((0, 0), (0, 0))
        action = Action((0, 1))
        s0, a0 = env.state_action_to_index(state, action)

        state = State((0, 1), (0, 1))
        action = Action((0, -2))
        s1, a1 = env.state_action_to_index(state, action)
        policy[s1] = a1

        state = State((0, 0), (0, -1))
        action = Action((0, 2))
        s2, a2 = env.state_action_to_index(state, action)
        policy[s2] = a2

        monte_carlo = MonteCarloESEveryVisit(env, policy, evaluation, 10)

        # when
        ep_states, ep_actions = monte_carlo.generate_episode(s0, a0)

        # then
        self.assertEqual(len(ep_states), 11)
        self.assertEqual(len(ep_actions), 10)


if __name__ == "__main__":
    unittest.main()
