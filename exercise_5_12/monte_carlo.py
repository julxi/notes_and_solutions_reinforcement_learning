import numpy as np
from race_track_environment import RaceTrackEnv


class MonteCarloESEveryVisit:
    def __init__(self, env: RaceTrackEnv, policy, evaluation, max_episode_length):
        self.env = env
        self.policy = policy
        self.evaluation = evaluation
        self.max_episode_length = max_episode_length

    def train(self, number_of_episodes):
        num_states, num_actions = self.evaluation.shape
        state_action_visits = np.zeros((num_states, num_actions), dtype=np.uint32)

        for iii in range(number_of_episodes):
            s, a = self.env.get_random_state_action_pair()
            ep_states, ep_actions = self.generate_episode(s, a)

            if len(ep_actions) == self.max_episode_length:
                seen = set()
                for s, a in zip(ep_states[:-1], ep_actions):
                    if (s, a) in seen:
                        continue
                    seen.add((s, a))
                    state_action_visits[s, a] += 1
                    alpha = 1.0 / state_action_visits[s, a]
                    if self.evaluation[s, a] == -np.inf:
                        print("AAAh1")
                    self.evaluation[s, a] += alpha * (
                        -self.max_episode_length - self.evaluation[s, a]
                    )
                    self.policy[s] = np.argmax(self.evaluation[s])
            else:
                ep_returns = list(range(-len(ep_actions), 0))
                for i in range(len(ep_actions)):
                    s = ep_states[i]
                    a = ep_actions[i]
                    state_action_visits[s, a] += 1
                    alpha = 1.0 / state_action_visits[s, a]
                    self.evaluation[s, a] += alpha * (
                        ep_returns[i] - self.evaluation[s, a]
                    )
                    self.policy[s] = np.argmax(self.evaluation[s])

    def generate_episode(self, initial_state, initial_action):
        """
        Generate an episode starting from a given state and action.
        Generates S0, ..., S_T,
        and A0, ..., A_{T-1}
        where T = max_episode_length or S_T is terminal.

        So the length of an episode is the number of actions taken.
        """
        self.env.set_state_index(initial_state)

        states = [initial_state]
        actions = []

        if self.env.is_terminal(initial_state):
            return states, actions

        state = initial_state
        action = initial_action
        for _ in range(self.max_episode_length):
            actions.append(action)
            state = self.env.take_action_index(action)
            states.append(state)

            if self.env.is_terminal(state):
                break

            action = self.policy[state]

        return states, actions
    
    def generate_episode_from_state(self, initial_state):
        action = self.policy[initial_state]

        return self.generate_episode(initial_state, action)
