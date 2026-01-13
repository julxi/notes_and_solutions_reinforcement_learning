from episode_mean import mean_over_last_episodes
import numpy as np
import pickle
import os

optimistic = [{"q_init": 2**k, "alpha": 0.1} for k in range(-8, 4)]
eps_sample = [{"eps": 2**k} for k in range(-8, 0)]
ucb_sample = [{"c": 2**k} for k in range(-8, 4)]
grad_sample = [{"alpha": 2**k} for k in range(-8, 4)]

groups = {
    "optimistic": optimistic,
    "eps_sample": eps_sample,
    "ucb_sample": ucb_sample,
    "grad_sample": grad_sample,
}

args = {
    "n_runs": 100_000,
    "n_episodes": 1_000,
    "keep_last": 1_000,
    "groups": groups,
    "bandit_q_mu": 0,
    "bandit_q_sd": 1,
    "reward_sd": 1,
    "value_drift": False,
}

res = mean_over_last_episodes(**args)

with open("results/parameter_study_stationary.pkl", "wb") as f:
    pickle.dump({"args": args, "res": res}, f)
