import random
import math
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm

import scripts.environment.race_track as rat


def generate_episode_epsilon_greedy(
    env: rat.RaceTrack,
    π: np.ndarray,
    ε: float,
    rng: random.Random,
):
    """
    Generate an episode following `policy_indices` but acting epsilon-greedily.
    Returns (state_indices, action_indices).
    """
    n_actions = len(env.action_space)

    state_indices = []  # S_0, ..., S_T
    action_indices = []  # A_0, ..., A_{T-1}

    state_idx = env.reset_idx()
    state_indices.append(state_idx)

    terminated = False
    while not terminated:

        if rng.random() < ε:
            action_idx = rng.randint(0, n_actions - 1)
        else:
            action_idx = π[state_idx]

        action_indices.append(action_idx)
        state_idx, terminated = env.step_idx(action_idx)
        state_indices.append(state_idx)

    return state_indices, action_indices


def on_policy_first_visit_mc_control_epsilon_soft(
    env: rat.RaceTrack,
    num_episodes: int,
    ε_min: float,
    ε_max: float,
    rng: random.Random,
):
    """
    First-visit on-policy Monte Carlo control with an epsilon-soft policy.

    Returns:
        policy_indices: array mapping state_idx -> greedy action_idx
        Q: action-value estimates (shape = [n_states, n_actions])
        episode_costs: array of episode 'loss' values per episode
    """
    n_states = len(env.state_space)
    n_actions = len(env.action_space)

    # action-value estimates and visit counts
    Q = np.zeros((n_states, n_actions), dtype=float)
    counts = np.zeros((n_states, n_actions), dtype=np.int64)

    # initialise a deterministic policy
    π = np.zeros((n_states), dtype=np.int64)  # 0 is always an action index
    for s_idx in range(n_states):
        π[s_idx] = rng.randint(0, n_actions - 1)

    # diagnostic
    loss = np.zeros((num_episodes), dtype=np.int32)
    reward = env.REWARD

    # ε-decay
    ε_decay_rate = np.log(ε_min / ε_max)
    for episode in tqdm(range(num_episodes)):
        # epsilon schedule
        t = episode / (num_episodes - 1)
        ε = ε_max * math.e ** (math.log(ε_min / ε_max) * t)

        # create episode
        state_indices, action_indices = generate_episode_epsilon_greedy(env, π, ε, rng)
        T = len(action_indices)
        seen = set()

        # forward pass first-visit
        for i in range(T):
            s = state_indices[i]
            a = action_indices[i]
            sa = (s, a)
            if sa in seen:
                continue
            seen.add(sa)

            G = reward * (T - i)
            counts[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / counts[s, a]
            π[s] = np.argmax(Q[s])

        loss[episode] = -reward * T

    return π, Q, loss


def train_and_export(env, num_episodes, ε_max, ε_min, seed, config):
    rng = random.Random(seed)

    # precompute startpoints of moving averages (and fail early)
    window = config["window"]
    n_samples = config["number_of_loss_samples"]

    starts_linear = np.linspace(0, num_episodes, n_samples, endpoint=False, dtype=int)
    starts_log = np.logspace(
        0, np.log10(num_episodes - 1), n_samples, endpoint=False, dtype=int
    )

    starts_total = np.unique(np.concatenate([starts_linear, starts_log]))

    if starts_total[-1] + window >= num_episodes:
        raise ValueError(
            f"... window {window} + last start {starts_total[-1]} >= num_episodes {num_episodes}"
        )

    # training
    env.rng = rng
    π_indexed, Q_indexed, loss = on_policy_first_visit_mc_control_epsilon_soft(
        env,
        num_episodes,
        ε_min,
        ε_max,
        rng,
    )

    # map indices back to human-friendly objects
    π = {
        env.state_space[s_idx]: env.action_space[π_indexed[s_idx]]
        for s_idx in range(len(env.state_space))
    }
    Q = {
        (env.state_space[s_idx], env.action_space[a_idx]): Q_indexed[s_idx, a_idx]
        for s_idx in range(len(env.state_space))
        for a_idx in range(len(env.action_space))
    }

    # moving average of 'loss'
    loss_means = []
    for start in starts_total:
        end = start + window
        window_loss = loss[start:end]
        avg = np.mean(window_loss)
        loss_means.append(avg)
    loss_means = np.array(loss_means)

    # prepare results and write to disk
    result = {
        "seed": seed,
        "env": env,
        "policy": π,
        "evaluation": Q,
        "num_episodes": num_episodes,
        "epsilon_max": ε_max,
        "epsilon_min": ε_min,
        "loss_means": loss_means,
        "loss_means_starts": starts_total,
        "loss_means_window": window,
    }

    track_name = config["track_name"]
    out_path = Path(f"results/race_track_{track_name}.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fo:
        pickle.dump(result, fo, protocol=pickle.HIGHEST_PROTOCOL)
