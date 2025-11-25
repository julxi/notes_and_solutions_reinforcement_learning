import random
import math
from collections import Counter, defaultdict
import environment.black_jack as black_jack
import pickle
from pathlib import Path


def generate_episode_es(env, π, start_state, start_action):
    """Generate an episode starting from start_state and start_action following policy π."""
    states = [start_state]  # S_0, ..., S_T
    rewards = [env.set_state(start_state)]  # R_0, ..., R_T
    actions = [start_action]  # A_0, ..., A_{T-1}

    # take start action
    state, reward, terminated = env.step(start_action)
    states.append(state)
    rewards.append(reward)

    # follow policy
    while not terminated:
        action = π[state]
        actions.append(action)

        state, reward, terminated = env.step(action)

        states.append(state)
        rewards.append(reward)

    return states, rewards, actions


def mc_es(
    env,
    iterations,
    min_policy_iterations=1000,
    confidence_gap_threshold=3,
    report_every=10_000,
    seed=None,
):
    """Monte Carlo control with Exploring Starts.

    Returns:
        π: The learned policy.
        q: The learned action-value function.
    """
    if seed is not None:
        random.seed(seed)

    action_space = env.action_space
    state_space = env.state_space

    π = {}
    q = {}
    state_action_count = Counter()
    m2 = defaultdict(float)  # Welford's M2 accumulator

    # randomized policy
    for s in state_space:
        π[s] = random.choice(action_space)
        for a in action_space:
            q[(s, a)] = 0.0

    def variance(state_action):
        count = state_action_count[state_action]
        if count < 2:
            return float("inf")
        return m2[state_action] / (count - 1)

    def stderr(state_action):
        v = variance(state_action)
        return math.sqrt(v / state_action_count[state_action])

    last_policy_change = -1
    policy_changes = []
    min_confidence_gaps = []

    def evaluate_confidence_gaps():
        min_confidence_gap = float("inf")
        for state in state_space:
            hit = (state, black_jack.Action.HIT)
            stick = (state, black_jack.Action.STICK)
            diff = abs(q[hit] - q[stick])
            uncert = stderr(hit) + stderr(stick)
            gap = diff / uncert if uncert != 0.0 else float("inf")
            if gap < min_confidence_gap:
                min_confidence_gap = gap
        return min_confidence_gap

    def update_policy():
        changes = []
        for state in state_space:
            best_action = max(action_space, key=lambda a: q[(state, a)])
            if π[state] != best_action:
                π[state] = best_action
                changes.append((state, best_action))
        return changes

    for iteration in range(1, iterations + 1):
        changed_states_this_iter = []
        last_policy_change += 1

        for start_state in env.state_space:
            for start_action in env.action_space:
                states, rewards, actions = generate_episode_es(
                    env, π, start_state, start_action
                )

                G = 0
                for i in reversed(range(len(actions))):
                    G += rewards[i + 1]
                    state_action = (states[i], actions[i])

                    # update incremental mean and Welford's M2 for variance
                    state_action_count[state_action] += 1
                    n = state_action_count[state_action]
                    old_mean = q[state_action]
                    delta = G - old_mean
                    new_mean = old_mean + delta / n
                    q[state_action] = new_mean
                    m2[state_action] += delta * (G - new_mean)

        if last_policy_change >= min_policy_iterations:
            changed_states_this_iter = update_policy()
            if changed_states_this_iter:
                policy_changes.append((iteration, changed_states_this_iter))
                if len(changed_states_this_iter) > 1:
                    print(
                        f"iteration={iteration} changed policy for {len(changed_states_this_iter)} states"
                    )
                else:
                    print(
                        f"iteration={iteration} changed policy for {changed_states_this_iter[0][0]} -> {changed_states_this_iter[0][1]}"
                    )
                last_policy_change = -1

        min_confidence_gap = evaluate_confidence_gaps()
        min_confidence_gaps.append(min_confidence_gap)

        if iteration % report_every == 0 or iteration == 1:
            print(f"iteration={iteration} min_confidence_gap={min_confidence_gap}")

        if min_confidence_gap > confidence_gap_threshold:
            print(
                f"Stopping early at iteration {iteration}; min_confidence_gap={min_confidence_gap}"
            )
            break

    return π, q, policy_changes, min_confidence_gaps


def save_pickle_simple(path: str, obj, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    Save obj to <path>.pkl using pickle. No temp files, minimal.
    Example: save_pickle_simple("results/blackjack_run1", package)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    print(f"Saved pickle -> {p}")


if __name__ == "__main__":
    seed = 0
    env = black_jack.BlackJack()

    # params
    iterations = 20_000_000
    min_policy_iterations = 10_000
    confidence_gap_threshold = 5
    report_every = 17_000

    π, q, policy_changes, min_confidence_gaps = mc_es(
        env,
        iterations,
        report_every=report_every,
        min_policy_iterations=min_policy_iterations,
        confidence_gap_threshold=confidence_gap_threshold,
    )

    package = {
        "pi": π,
        "q": q,
        "policy_changes": policy_changes,
        "min_confidence_gaps": min_confidence_gaps,
        "metadata": {
            "seed": seed,
            "confidence_gap_threshold": confidence_gap_threshold,
        },
    }

    save_pickle_simple(
        f"blackjack_{iterations}_{confidence_gap_threshold}.pkl", package
    )
