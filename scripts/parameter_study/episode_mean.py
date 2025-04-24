import numba as nb
import numpy as np


@nb.njit
def _sample_from_probs(p):
    """Return an index according to a discrete distribution p (∑p = 1)."""
    r = np.random.random()
    c = 0.0
    for i in range(p.size):
        c += p[i]
        if r < c:
            return i
    raise AssertionError("probabilities do not sum to 1 within FP tolerance")


@nb.njit
def episode_mean(
    agent_type,
    num_arms,
    steps,
    keep,
    bandit_action_mu,
    bandit_action_sd,
    drift_mu,
    drift_sd,
    rng_seed,
    eps=0.1,
    alpha=0.1,
):
    """
    Run one drifting k-armed bandit episode and return the mean reward over the
    last *keep* steps.

    agent_type
      0 = sample-average ε-greedy
      1 = constant-α   ε-greedy
      2 = gradient ascent,     sample-average baseline
      3 = gradient ascent,     no baseline
    """
    np.random.seed(rng_seed)
    if agent_type not in (0, 1, 2, 3):
        raise ValueError("agent_type must be 0, 1, 2, or 3")

    # --- bandit state ---------------------------------------------------------
    q = np.full(num_arms, bandit_action_mu, dtype=np.float64)

    # --- agent state ----------------------------------------------------------
    if agent_type in (0, 1):  # ε-greedy variants
        q_est = np.zeros(num_arms, dtype=np.float64)
        if agent_type == 0:
            counts = np.zeros(num_arms, dtype=np.int64)
    else:  # gradient-bandit variants
        prefs = np.zeros(num_arms, dtype=np.float64)
        avg_r = 0.0  # only used when a baseline is wanted

    # --- running mean for evaluation -----------------------------------------
    total = 0.0
    cutoff = steps - keep + 1

    # === main loop ============================================================
    for t in range(1, steps + 1):

        # choose an action -----------------------------------------------------
        if agent_type in (2, 3):  # soft-max over preferences
            probs = np.exp(prefs - prefs.max())
            probs /= probs.sum()
            a = _sample_from_probs(probs)
        else:  # ε-greedy
            if np.random.random() < eps:
                a = np.random.randint(num_arms)
            else:
                a = np.argmax(q_est)

        # pull the arm ---------------------------------------------------------
        r = q[a] + bandit_action_sd * np.random.randn()

        # accumulate reward for the tail of the episode ------------------------
        if t >= cutoff:
            total += r

        # update agent ---------------------------------------------------------
        if agent_type == 0:  # sample-average ε-greedy
            counts[a] += 1
            q_est[a] += (r - q_est[a]) / counts[a]

        elif agent_type == 1:  # constant-α ε-greedy
            q_est[a] += alpha * (r - q_est[a])

        else:  # gradient ascent
            # work out the advantage term (r - baseline)
            if agent_type == 2:  # with sample-average baseline
                avg_r += (r - avg_r) / t
                advantage = r - avg_r
            else:  # no baseline
                advantage = r

            # preference update
            for j in range(num_arms):
                if j == a:
                    prefs[j] += alpha * advantage * (1.0 - probs[j])
                else:
                    prefs[j] -= alpha * advantage * probs[j]

        # non-stationary drift -------------------------------------------------
        q += drift_mu + drift_sd * np.random.randn(num_arms)

    return total / keep
