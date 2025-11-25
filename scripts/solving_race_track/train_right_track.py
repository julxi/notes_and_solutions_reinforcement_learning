import random

from .tracks import TRACK_RIGHT
from .mc_control import train_and_export
import scripts.environment.race_track as rat

if __name__ == "__main__":
    # config
    seed = 0

    ε_max = 0.25
    ε_min = 0.025

    n_episodes = 1_000_000_000
    window = 10_000
    n_loss_samples = 2000

    # main
    env = rat.RaceTrack(TRACK_RIGHT, only_positive_velocity=False)
    config = {
        "track_name": "right",
        "window": window,
        "number_of_loss_samples": n_loss_samples,
    }
    print(
        f"=== Training on {config["track_name"]} track (ε from {ε_max} to {ε_min}) ==="
    )
    train_and_export(env, n_episodes, ε_max, ε_min, seed, config)
