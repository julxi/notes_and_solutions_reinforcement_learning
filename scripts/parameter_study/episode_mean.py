# This module is vibe coded

import numpy as np
from typing import Dict, List
from tqdm import trange


def _vectorized_categorical_sample(probs, rng):
    """
    probs: (..., n_arms) where last axis sums to 1
    returns: indices of shape (...) sampled according to probs
    """
    shape = probs.shape[:-1]
    u = rng.random(size=shape + (1,))  # (..., 1)
    cums = np.cumsum(probs, axis=-1)  # (..., n_arms)
    return np.argmax(u <= cums, axis=-1)


def mean_over_last_episodes(
    groups: Dict[str, List[Dict]],
    n_episodes: int = 200_000,
    keep_last: int = 100_000,
    n_runs: int = 10,
    bandit_q_mu: float = 0.0,
    bandit_q_sd: float = 1.0,
    reward_sd: float = 1.0,
    value_drift: bool = True,
    drift_mu: float = 0.0,
    drift_sd: float = 0.01,
    n_arms: int = 10,
    rng_seed=None,
):
    """
    Vectorised simulator across runs and grouped agents.

    - `groups` is a dict mapping group name to a list of agent spec dicts.
      Valid group names (supported here):
        'eps_sample'  : sample-average epsilon-greedy agents. spec keys: 'eps'
        'eps_const'   : constant-alpha epsilon-greedy agents. spec keys: 'eps', 'alpha'
        'grad_sample' : gradient-bandit with sample-average baseline. spec keys: 'alpha'
        'grad_const'  : gradient-bandit with constant-alpha baseline. spec keys: 'alpha', optional 'baseline_alpha'
        'optimistic'  : optimistic greedy (constant-alpha). spec keys: 'alpha', 'q_init'
        'ucb_sample'  : UCB with sample-average Q update. spec keys: 'c'
        'ucb_const'   : UCB with constant-alpha Q update. spec keys: 'c', 'alpha'

    Returns:
      dict mapping group_name -> np.ndarray (per-agent mean reward over last keep_last episodes).
    """
    rng = np.random.default_rng(rng_seed)

    # --- bandit true action values: drawn per-run, per-arm --------------------
    q = rng.normal(loc=bandit_q_mu, scale=bandit_q_sd, size=(n_runs, n_arms))

    # --- prepare per-group state arrays -------------------------------------
    group_state = {}
    for gname, specs in groups.items():
        n_g = len(specs)
        if n_g == 0:
            continue

        if gname in ("eps_sample", "eps_const"):
            q_est = np.zeros((n_runs, n_g, n_arms), dtype=np.float64)
            counts = (
                np.zeros((n_runs, n_g, n_arms), dtype=np.int64)
                if gname == "eps_sample"
                else None
            )
            eps_arr = np.array([s.get("eps", 0.1) for s in specs], dtype=np.float64)[
                None, :
            ]  # (1, n_g)
            alpha_arr = np.array(
                [s.get("alpha", 0.1) for s in specs], dtype=np.float64
            )[
                None, :
            ]  # (1, n_g)
            total = np.zeros(n_g, dtype=np.float64)
            group_state[gname] = {
                "q_est": q_est,
                "counts": counts,
                "eps": eps_arr,
                "alpha": alpha_arr,
                "total": total,
            }

        elif gname in ("grad_sample", "grad_const"):
            # both use an average baseline; grad_sample updates baseline with 1/t,
            # grad_const updates baseline with a constant baseline_alpha (default alpha).
            prefs = np.zeros((n_runs, n_g, n_arms), dtype=np.float64)
            avg_r = np.zeros((n_runs, n_g), dtype=np.float64)
            alpha_arr = np.array(
                [s.get("alpha", 0.1) for s in specs], dtype=np.float64
            )[
                None, :
            ]  # (1, n_g)
            if gname == "grad_const":
                baseline_alpha_arr = np.array(
                    [s.get("baseline_alpha", s.get("alpha", 0.1)) for s in specs],
                    dtype=np.float64,
                )[
                    None, :
                ]  # (1, n_g)
            else:
                baseline_alpha_arr = None
            total = np.zeros(n_g, dtype=np.float64)
            group_state[gname] = {
                "prefs": prefs,
                "avg_r": avg_r,
                "alpha": alpha_arr,
                "baseline_alpha": baseline_alpha_arr,
                "total": total,
            }

        elif gname == "optimistic":
            # optimistic greedy: constant-alpha updates, initial q_est = q_init
            alpha_arr = np.array(
                [s.get("alpha", 0.1) for s in specs], dtype=np.float64
            )[
                None, :
            ]  # (1, n_g)
            q_init_arr = np.array(
                [s.get("q_init", 5.0) for s in specs], dtype=np.float64
            )  # shape (n_g,)
            # --- Robust initialisation: broadcast q_init into (n_runs, n_g, n_arms)
            q_est = np.zeros((n_runs, n_g, n_arms), dtype=np.float64)
            q_est[:] = q_init_arr[None, :, None]
            total = np.zeros(n_g, dtype=np.float64)
            group_state[gname] = {"q_est": q_est, "alpha": alpha_arr, "total": total}

        elif gname in ("ucb_sample", "ucb_const"):
            # both need counts for UCB bonus; q_est differs in update rule
            q_est = np.zeros((n_runs, n_g, n_arms), dtype=np.float64)
            counts = np.zeros((n_runs, n_g, n_arms), dtype=np.int64)  # for bonus
            c_arr = np.array([s.get("c", 1.0) for s in specs], dtype=np.float64)[
                None, :
            ]  # (1, n_g)
            alpha_arr = np.array(
                [s.get("alpha", 0.1) for s in specs], dtype=np.float64
            )[
                None, :
            ]  # (1, n_g)
            total = np.zeros(n_g, dtype=np.float64)
            group_state[gname] = {
                "q_est": q_est,
                "counts": counts,
                "c": c_arr,
                "alpha": alpha_arr,
                "total": total,
            }

        else:
            raise ValueError(f"Unknown group name: {gname}")

    # --- main loop over episodes --------------------------------------------
    cutoff = n_episodes - keep_last + 1
    for t in trange(1, n_episodes + 1):
        # sample observed rewards for every run × arm (true q plus observation noise)
        r_all = q + reward_sd * rng.normal(size=(n_runs, n_arms))

        # handle epsilon-greedy groups
        for gname in ("eps_sample", "eps_const"):
            st = group_state.get(gname)
            if st is None:
                continue
            q_est = st["q_est"]  # (n_runs, n_g, n_arms)
            counts = st["counts"]
            eps = st["eps"]  # (1, n_g)
            alpha = st["alpha"]  # (1, n_g)
            n_g = q_est.shape[1]

            # explore mask
            u = rng.random(size=(n_runs, n_g))
            explore = u < eps  # (n_runs, n_g)

            exploit_choice = np.argmax(q_est, axis=-1)  # (n_runs, n_g)
            explore_choice = rng.integers(0, n_arms, size=(n_runs, n_g))
            actions = np.where(explore, explore_choice, exploit_choice)  # (n_runs, n_g)

            # rewards for chosen actions
            r_sel = r_all[np.arange(n_runs)[:, None], actions]  # (n_runs, n_g)

            # update estimates
            rows = np.arange(n_runs)[:, None]
            cols = np.arange(n_g)[None, :]

            if gname == "eps_sample":
                counts[rows, cols, actions] += 1
                denom = counts[rows, cols, actions].astype(np.float64)
                q_sel = q_est[rows, cols, actions]
                q_est[rows, cols, actions] = q_sel + (r_sel - q_sel) / denom
            else:  # eps_const
                q_sel = q_est[rows, cols, actions]
                q_est[rows, cols, actions] = q_sel + alpha * (r_sel - q_sel)

            # evaluation accumulation (per-group)
            if t >= cutoff:
                st["total"] += r_sel.mean(axis=0)  # mean across runs -> (n_g,)

        # handle gradient groups (sample-average baseline and constant-alpha baseline)
        for gname in ("grad_sample", "grad_const"):
            st = group_state.get(gname)
            if st is None:
                continue
            prefs = st["prefs"]  # (n_runs, n_g, n_arms)
            avg_r = st["avg_r"]  # (n_runs, n_g)
            alpha = st["alpha"]  # (1, n_g)
            baseline_alpha = st.get("baseline_alpha")  # (1, n_g) or None
            n_g = prefs.shape[1]

            maxp = prefs.max(axis=-1, keepdims=True)
            exps = np.exp(prefs - maxp)
            probs = exps / exps.sum(axis=-1, keepdims=True)  # (n_runs, n_g, n_arms)

            actions = _vectorized_categorical_sample(probs, rng)  # (n_runs, n_g)
            r_sel = r_all[np.arange(n_runs)[:, None], actions]  # (n_runs, n_g)

            # baseline update and advantage
            if gname == "grad_sample":
                # sample-average baseline
                avg_r += (r_sel - avg_r) / float(t)
                advantage = r_sel - avg_r
            else:
                # constant-α baseline (baseline_alpha defaults to alpha if not provided)
                b_alpha = baseline_alpha if baseline_alpha is not None else alpha
                avg_r += b_alpha * (r_sel - avg_r)
                advantage = r_sel - avg_r

            # one-hot chosen arms
            one_hot = np.zeros_like(probs)
            rows = np.arange(n_runs)[:, None]
            cols = np.arange(n_g)[None, :]
            one_hot[rows, cols, actions] = 1.0

            adv3 = advantage[:, :, None]  # (n_runs, n_g, 1)
            prefs += alpha[:, :, None] * adv3 * (one_hot - probs)

            if t >= cutoff:
                st["total"] += r_sel.mean(axis=0)

        # handle optimistic greedy (constant-alpha, optimistic initial q_est)
        st = group_state.get("optimistic")
        if st is not None:
            q_est = st["q_est"]  # (n_runs, n_g, n_arms) already initialised to q_init
            alpha = st["alpha"]  # (1, n_g)
            n_g = q_est.shape[1]

            # purely greedy (no epsilon) because optimism supplies exploration
            actions = np.argmax(q_est, axis=-1)  # (n_runs, n_g)
            r_sel = r_all[np.arange(n_runs)[:, None], actions]  # (n_runs, n_g)

            # update constant-alpha
            rows = np.arange(n_runs)[:, None]
            cols = np.arange(n_g)[None, :]
            q_sel = q_est[rows, cols, actions]
            q_est[rows, cols, actions] = q_sel + alpha * (r_sel - q_sel)

            if t >= cutoff:
                st["total"] += r_sel.mean(axis=0)

        # handle UCB groups
        for gname in ("ucb_sample", "ucb_const"):
            st = group_state.get(gname)
            if st is None:
                continue
            q_est = st["q_est"]  # (n_runs, n_g, n_arms)
            counts = st["counts"]  # (n_runs, n_g, n_arms)
            c = st["c"]  # (1, n_g)
            alpha = st["alpha"]  # (1, n_g) may be unused in sample version
            n_g = q_est.shape[1]

            # compute bonus: c * sqrt( ln(t) / N )
            # avoid division by zero: for N==0 set bonus = very large to ensure selection
            ln_t = np.log(float(t))
            # convert counts to float and compute bonus where counts>0
            cnt_float = counts.astype(np.float64)
            # safe bonus: where counts>0 -> c * sqrt(ln_t / cnt), else large
            with_counts = cnt_float > 0.0
            bonus = np.empty_like(q_est)
            # broadcast c over arms: c shape (1,n_g) -> (n_runs,n_g)
            c_broadcast = c  # (1, n_g)
            # compute sqrt term only where counts>0
            # use broadcasting: cnt_float has shape (n_runs,n_g,n_arms)
            bonus_val = c_broadcast[:, :, None] * np.sqrt(
                ln_t / np.where(with_counts, cnt_float, 1.0)
            )
            # set bonus for with_counts False to a large value
            LARGE = 1e9
            bonus = np.where(with_counts, bonus_val, LARGE)

            # action selection: argmax over (q_est + bonus)
            combined = q_est + bonus
            actions = np.argmax(combined, axis=-1)  # (n_runs, n_g)

            # observe rewards
            r_sel = r_all[np.arange(n_runs)[:, None], actions]  # (n_runs, n_g)

            # update counts then Q estimates
            rows = np.arange(n_runs)[:, None]
            cols = np.arange(n_g)[None, :]

            counts[rows, cols, actions] += 1

            if gname == "ucb_sample":
                # sample-average update
                denom = counts[rows, cols, actions].astype(np.float64)
                q_sel = q_est[rows, cols, actions]
                q_est[rows, cols, actions] = q_sel + (r_sel - q_sel) / denom
            else:
                # constant-alpha update uses provided alpha
                q_sel = q_est[rows, cols, actions]
                q_est[rows, cols, actions] = q_sel + alpha * (r_sel - q_sel)

            if t >= cutoff:
                st["total"] += r_sel.mean(axis=0)

        # --- accumulate for eps/grad groups already done; drift done next -----
        # non-stationary drift for true q values
        if value_drift:
            q += drift_mu + drift_sd * rng.normal(size=(n_runs, n_arms))

    # --- prepare output: dict of grouped means -------------------------------
    result = {}
    for gname, st in group_state.items():
        result[gname] = st["total"] / float(keep_last)

    return result
