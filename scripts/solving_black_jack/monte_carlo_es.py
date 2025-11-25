from ..environment.black_jack import State, Action, BlackJack
from .welford import Welford
from collections import defaultdict
import tqdm
import math


class MonteCarloES:
    def __init__(self, env: BlackJack):
        self.env: BlackJack = env
        self._stats_acc: dict[tuple[State, Action], Welford] = {}

    def _reset_stats(self):
        for s in self.env.state_space:
            for a in self.env.action_space:
                self._stats_acc[(s, a)] = Welford()

    def _generate_episode_es(self, π, start_state: State, start_action: Action):
        """Generate an episode starting from start_state and start_action following policy π."""
        states = [start_state]  # S_0,...,S_T
        rewards = [self.env.set_state(start_state)]  # R_0,...,R_T
        actions = [start_action]  # A_0,...,A_{T-1}

        # take start action
        state, reward, terminated = self.env.step(start_action)
        states.append(state)
        rewards.append(reward)

        # follow policy
        while not terminated:
            action = π[state]
            actions.append(action)
            state, reward, terminated = self.env.step(action)
            states.append(state)
            rewards.append(reward)

        return states, rewards, actions

    def _one_iteration(self, π):  # needs a better name definetly
        for state in self.env.state_space:
            for action in self.env.action_space:
                states, rewards, actions = self._generate_episode_es(π, state, action)
                G = 0.0
                for i in reversed(range(len(actions))):
                    G += rewards[i + 1]
                    s = states[i]
                    a = actions[i]
                    self._stats_acc[(s, a)].update(G)

    def _make_Q_and_var(self):  # also better name :/
        Q_mean = {}
        Q_var_of_mean = {}
        for key, acc in self._stats_acc.items():
            Q_mean[key] = acc.mean
            Q_var_of_mean[key] = acc.variance_of_mean

        return Q_mean, Q_var_of_mean

    def predict(self, π, iterations):
        assert iterations > 1
        self._reset_stats()

        for _ in tqdm.trange(iterations):
            self._one_iteration(π)

        return self._make_Q_and_var()

    def control(
        self,
        iterations,
        min_policy_iterations=1000,
        confidence_gap_threshold=3,
    ):
        self._reset_stats()

        π = {}
        for s in self.env.state_space:
            π[s] = self.env.action_space[0]

        last_policy_change = -1
        policy_changes = []
        min_confidence_gaps = []

        def evaluate_confidence_gaps():
            min_confidence_gap = float("inf")
            action_space = self.env.action_space
            for s in self.env.state_space:
                acc_0 = self._stats_acc[(s, action_space[0])]
                acc_1 = self._stats_acc[(s, action_space[1])]
                diff = abs(acc_0.mean - acc_1.mean)
                std_err = math.sqrt(acc_0.variance_of_mean + acc_1.variance_of_mean)
                gap = diff / std_err if std_err != 0.0 else float("inf")
                if gap < min_confidence_gap:
                    min_confidence_gap = gap
            return min_confidence_gap

        def update_policy():
            changes = []
            for s in self.env.state_space:
                best_action = max(
                    self.env.action_space,
                    key=lambda a: self._stats_acc[(s, a)].mean,
                )
                if π[s] != best_action:
                    π[s] = best_action
                    changes.append((s, best_action))
            return changes

        pbar = tqdm.trange(1, iterations + 1, desc="MC-ES control", dynamic_ncols=True)
        for iteration in pbar:
            changed_states_this_iter = []
            last_policy_change += 1
            self._one_iteration(π)
            if last_policy_change >= min_policy_iterations:
                changed_states_this_iter = update_policy()
                if changed_states_this_iter:
                    policy_changes.append((iteration, changed_states_this_iter))
                    last_policy_change = -1

            if iteration > 3:
                min_confidence_gap = evaluate_confidence_gaps()
            else:
                min_confidence_gap = 0
            min_confidence_gaps.append(min_confidence_gap)

            pbar.set_postfix(
                {
                    "min_gap": f"{min_confidence_gap:.2f}",
                    "stable_for": f"{last_policy_change:05d}",
                }
            )

            if (
                min_confidence_gap > confidence_gap_threshold
                and last_policy_change >= min_policy_iterations
            ):
                print(
                    f"Stopping early at iteration {iteration}; min_confidence_gap={min_confidence_gap:.3f}"
                )
                break

        Q_mean, Q_var = self._make_Q_and_var()
        return π, policy_changes, Q_mean, Q_var


env = BlackJack()
mc_es = MonteCarloES(env)

π, _, _, _ = mc_es.control(
    50_000, min_policy_iterations=1000, confidence_gap_threshold=5
)

from .utils import print_policy

print_policy(π)
