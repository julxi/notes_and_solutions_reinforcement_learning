import numpy as np
import matplotlib.pyplot as plt

from windy_grid import WindyGrid
from windy_grid_env import WindyGridEnv
from sarsa import sarsa_train_for_one_episode


windy_grid = WindyGrid(
    width=10,
    height=7,
    wind_speeds=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    start=np.array([0, 3]),
    end=np.array([7, 3]),
)

windy_grid_env = WindyGridEnv(windy_grid, delta_max=1, wind_var=0)

q = np.zeros((windy_grid.width, windy_grid.height, windy_grid_env.action_count))

# first training
num_episodes = 600
episode_lengths = []

for episode in range(num_episodes):
    states, _ = sarsa_train_for_one_episode(windy_grid_env, q, α=0.1, γ=1, ε=0.1)
    episode_lengths.append(len(states))

# Plotting
data = episode_lengths[-100:]


plt.plot(data, linewidth=1, label=f"after {num_episodes} episodes")
plt.xlabel("Episode")
plt.ylabel("Episode length")
plt.title("SARSA: Episode Length with Smoothing")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
#plt.show()


# exploitation
states, actions = sarsa_train_for_one_episode(windy_grid_env, q, α=0.1, γ=1, ε=0)
print(f"states: {states}")
print(f"actions: {actions}")
print(f"len: {len(states)}")
windy_grid.print_episode(states)




