import numpy as np
from windy_grid import WindyGrid
from windy_grid_env import WindyGridEnv


# Algorithm parameters: step size α in (0, 1], small ε > 0
# Initialize Q(s, a), for all s in S^+, a in A(s),
# arbitrarily except that Q(terminal, ·)=0


# Loop for each episode:
#   Initialize S
#   Choose A from S using policy derived from Q (e.g., ε-greedy)
#   Loop for each step of episode:
#       Take action A, observe R, S'
#       Choose A' from S' using policy derived from Q (e.g., ε-greedy)
#       Q(S, A) += α [R + γ Q(S', A') - Q(S,A)]
#       S <- S'; A <- A'
#   until S is terminal
def sarsa_train_for_one_episode(windy_grid_env, q, α=0.1, γ=1, ε=0.1, seed=None):
    """
    make sure that q(s,a) = 0 for terminal s
    """
    rng = np.random.default_rng(seed)

    episode_states = []
    episode_actions = []

    state = windy_grid_env.reset()
    episode_states.append(state)
    a = epsilon_greedy(q, state, windy_grid_env.action_count, ε, rng)
    episode_actions.append(a)
    while windy_grid_env.is_terminal(state) is False:
        state_new = windy_grid_env.take_action_code(a)

        a_new = epsilon_greedy(q, state_new, windy_grid_env.action_count, ε, rng)
        x, y = state
        x_new, y_new = state_new
        # reward for each step is -1
        q[x, y, a] += α * (-1 + γ * q[x_new, y_new, a_new] - q[x, y, a])

        state, a = state_new, a_new
        episode_states.append(state)
        episode_actions.append(a)
    return episode_states, episode_actions


def epsilon_greedy(q, state, action_count, ε, rng):
    x, y = state
    if rng.random() < ε:
        return rng.integers(low=0, high=action_count)
    else:
        return np.argmax(q[x, y])
