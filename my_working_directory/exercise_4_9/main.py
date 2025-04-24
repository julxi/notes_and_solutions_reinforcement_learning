import matplotlib.pyplot as plt
import numpy as np
import random




def init():
    values[:] = np.random.randint(2, size=max_capital + 1)
    values[0] = 0
    values[max_capital] = 1
    policy[:] = [random.choice([1, i]) for i in range(max_capital + 1)]
    policy[0] = 0
    policy[max_capital] = 0
    policy[:] = policy.astype(int)


def policy_evaluation():
    while True:
        Δ = 0
        for s in range(1, max_capital):
            v_old = values[s]
            policy_action = policy[s]
            state_win = min(s + policy_action, max_capital)
            state_lose = max(s - policy_action, 0)
            values[s] = prob_lose * values[state_lose] + prob_win * values[state_win]
            Δ = max(Δ, abs(v_old - values[s]))
        if Δ < θ:
            break


def policy_improvement():
    policy_stable = True
    for s in range(1, max_capital):
        old_action = policy[s]
        policy[s] = 1 + np.argmax(
            [
                prob_lose * values[max(s - a, 0)]
                + prob_win * values[min(s + a, max_capital)]
                for a in range(1, s + 1)
            ]
        )
        if old_action != policy[s]:
            policy_stable = False
    return policy_stable

def policy_iteration():
    iterations = 0
    while True:
        iterations += 1
        policy_evaluation()
        if policy_improvement():
            break
    return iterations



seed = 7
random.seed(seed)
np.random.seed(seed)

θ = 1e-8

max_capital = 100
prob_win = 0.45
prob_lose = 1 - prob_win

# states: 0 to 100 (101 states)
# 0, and max_capital are (dummy) terminal states
values = np.zeros(max_capital + 1)
policy = np.zeros(max_capital + 1, dtype=int)



init()

#policy[:] = [1]
#policy[:] = [i for i in range(max_capital + 1)]
#policy[0] = 0
#policy[max_capital] = 0

iterations = policy_iteration()
print("iterations:", iterations)

# Create the plot
fig, ax1 = plt.subplots()

# Plot policy (stakes) on the primary y-axis
ax1.set_xlabel("Capital")
ax1.set_ylabel("Stake", color='tab:blue')
ax1.plot(range(max_capital + 1), policy, marker='o', color='tab:blue', label="Stake (Final Policy)")
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for the value function
ax2 = ax1.twinx()
ax2.set_ylabel("Value Function", color='tab:red')
ax2.plot(range(max_capital + 1), values, marker='s', linestyle='--', color='tab:red', label="Value Function")
ax2.tick_params(axis='y', labelcolor='tab:red')

# Title and layout adjustments
fig.suptitle("Capital vs. Stake & Value Function")
fig.tight_layout()
plt.show()