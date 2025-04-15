import numpy as np
import matplotlib.pyplot as plt

class ArmedBandit:
    def __init__(
        self,
        size=10,
        start_value=1.0,
        value_std=1.0,
        value_walk_dist=1,
    ):
        """
        Initialise a multi-armed bandit problem.
        """
        self.size = size
        self.start_value = float(start_value)
        self.value_std = value_std
        self.value_walk_dist = value_walk_dist
        self.q_values = np.full(
            size, self.start_value, dtype=np.float64
        )  # Explicitly set dtype

    def pull_arm(self, arm):
        """
        Pull an arm and get a stochastic reward.
        """
        if arm < 0 or arm >= self.size:
            raise ValueError("Invalid arm index")

        reward = np.random.normal(self.q_values[arm], self.value_std)
        return reward

    def walk_rewards(self):
        """
        Perform a random walk update on the true action-values.
        """
        signs = np.random.choice([-1, 1], size=self.size)
        self.q_values += signs * self.value_walk_dist  # Now it correctly updates floats


class SampleAverageAgent:
    def __init__(self, size=10, optimism=5, curiosity=0.1):
        self.size = size
        self.curiosity = curiosity
        self.estimates = np.full(self.size, optimism)
        self.action_counts = np.zeros(self.size)

    def choose_action(self):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.curiosity:
            return np.random.randint(self.size)
        else:
            return self.preferred_action()

    def preferred_action(self):
        """
        Return the index of the best action according to current estimates.
        """
        return np.argmax(self.estimates)

    def update_estimates(self, action, reward):
        """
        Update the action-value estimates.
        """
        self.action_counts[action] += 1
        step_size = 1.0 / self.action_counts[action]
        self.estimates[action] += step_size * (reward - self.estimates[action])


class ConstantStepAgent:
    def __init__(self, step_size, size=10, optimism=5, curiosity=0.1):
        self.step_size = step_size
        self.size = size
        self.curiosity = curiosity
        self.estimates = np.full(self.size, optimism)

    def choose_action(self):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.curiosity:
            return np.random.randint(self.size)
        else:
            return self.preferred_action()

    def preferred_action(self):
        """
        Return the index of the best action according to current estimates.
        """
        return np.argmax(self.estimates)

    def update_estimates(self, action, reward):
        """
        Update the action-value estimates.
        """
        self.estimates[action] += self.step_size * (reward - self.estimates[action])


# Set up bandits and agents
steps = 50_000
bandit = ArmedBandit()

saa = SampleAverageAgent()
saa_rewards_average = []
saa_total_rewards = 0
saa_preferred_actions = []

csa = ConstantStepAgent(step_size=0.1)
csa_rewards_average = []
csa_total_rewards = 0
csa_preferred_actions = []

for i in range(1, steps + 1):
    # Sample Average Agent
    action = saa.choose_action()
    reward = bandit.pull_arm(action)
    saa_total_rewards += reward
    saa_rewards_average.append(saa_total_rewards / i)
    saa_preferred_actions.append(saa.preferred_action())
    saa.update_estimates(action, reward)

    # Constant Step Agent
    action = csa.choose_action()
    reward = bandit.pull_arm(action)
    csa_total_rewards += reward
    csa_rewards_average.append(csa_total_rewards / i)
    csa_preferred_actions.append(csa.preferred_action())
    csa.update_estimates(action, reward)

    bandit.walk_rewards()

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot average rewards on ax1
ax1.plot(saa_rewards_average, label="Sample Average Agent")
ax1.plot(csa_rewards_average, label="Constant Step Agent")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Average Reward")
ax1.legend(loc="upper left")
ax1.grid(True)

# Create a second y-axis for preferred actions
ax2 = ax1.twinx()
ax2.plot(saa_preferred_actions, linestyle="--", label="SAA Preferred Action")
ax2.plot(csa_preferred_actions, linestyle="--", label="CSA Preferred Action")
ax2.set_ylabel("Preferred Action (Arm Index)")

# We can combine legends from both axes if needed
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.title("Average Reward and Preferred Action Over Time")
plt.show()
