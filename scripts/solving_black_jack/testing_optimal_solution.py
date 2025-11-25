from math import sqrt
import sys
import pickle
from ..environment.black_jack import State, Action, Reward, BlackJack
from collections import Counter, defaultdict
import statistics
import random
from dataclasses import dataclass

π = {}
# sticky background
for dealer_revealed in range(2, 12):
    for player_total in range(12, 22):
        for is_soft in (True, False):
            π[State(player_total, is_soft, dealer_revealed)] = Action.STICK

# soft
for dealer_revealed in range(2, 12):
    for player_total in range(12, 18):
        π[State(player_total, True, dealer_revealed)] = Action.HIT
π[State(18, True, 9)] = Action.HIT
π[State(18, True, 10)] = Action.HIT

# hard
π[State(12, False, 2)] = Action.HIT
π[State(12, False, 3)] = Action.HIT
for dealer_revealed in range(7, 12):
    for player_total in range(12, 17):
        π[State(player_total, False, dealer_revealed)] = Action.HIT
π[State(15, False, 11)] = Action.STICK
π[State(16, False, 11)] = Action.STICK

# optimal solution candidate found by MC-ES
env = BlackJack()


def mc_prediction(env, π, N):
    returns = defaultdict(list)

    for _ in range(N):
        for state in env.state_space:
            for action in env.action_space:
                states, rewards, actions = generate_episode_es(env, π, state, action)
                G = 0
                for i in reversed(range(len(actions))):
                    G += rewards[i + 1]
                    state = states[i]
                    action = actions[i]
                    returns[(state, action)].append(G)

    Q = {}
    Q_var = {}
    for state in env.state_space:
        for action in env.action_space:
            Q[(state, action)] = statistics.mean(returns[(state, action)])
            Q_var[(state, action)] = statistics.variance(
                returns[(state, action)]
            ) / len(returns[(state, action)])

    return Q, Q_var


iterations = 400_000
Q, Q_var = mc_prediction(env, π, iterations)


# check if the same:
changes = []
warnings = []
for state in env.state_space:
    best_action = max(env.action_space, key=lambda a: Q[(state, a)])
    other_action = Action.HIT if best_action == Action.STICK else Action.STICK
    se_diff = sqrt(Q_var[(state, best_action)] + Q_var[(state, other_action)])
    if Q[(state, best_action)] - Q[(state, other_action)] < 1.96 * se_diff:
        warnings.append(f"warning! - Q values to close for state {state}")
    if π[state] != best_action:
        π[state] = best_action
        changes.append((state, best_action))

for change in changes:
    print(change)
print(f"total changes: {len(changes)}")
print(len(warnings))
for warning in warnings:
    print(warning)
