from episode_mean import mean_over_last_episodes
import pickle


eps_sample = [{"eps": 2**k} for k in range(-14, 0)]
eps_const = [{"eps": 2**k, "alpha": 0.1} for k in range(-14, 0)]
ucb_sample = [{"c": 2**k} for k in range(-2, 10)]
ucb_const = [{"c": 2**k, "alpha": 0.1} for k in range(-2, 10)]
grad_sample = [{"alpha": 2**k} for k in range(-16, 4)]
grad_const = [{"alpha": 2**k, "baseline_alpha": 0.1} for k in range(-16, 4)]

groups = {
    "eps_sample": eps_sample,
    "eps_const": eps_const,
    "ucb_sample": ucb_sample,
    "ucb_const": ucb_const,
    "grad_sample": grad_sample,
    "grad_const": grad_const,
}

args = {
    "groups": groups,
    "n_episodes": 200_000,
    "keep_last": 100_000,
    "n_runs": 1_000,
    "bandit_q_mu": 0,
    "bandit_q_sd": 0,
    "reward_sd": 1,
    "value_drift": True,
    "drift_mu": 0,
    "drift_sd": 0.01,
    "n_arms": 10,
    "rng_seed": 0,
}

res = mean_over_last_episodes(**args)

with open("results/parameter_study_nonstationary.pkl", "wb") as f:
    pickle.dump({"args": args, "res": res}, f)
