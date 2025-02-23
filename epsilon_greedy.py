"""
Epsilon greedy algorithm
-------------------------
Explore a random arm with probability epsilon, otherwise exploit the best arm so far

Implemented for two arms with Gaussian distirbutions:
    - Arm 1: N(0,1)
    - Arm 2: N(-delta,1)
    
Epsilon will be min(1, c/t) at each time step t. The mean reward of each arm is updated only
for exploration steps.
"""

import numpy as np
import csv
import os


class EpsilonGreedy:

    def __init__(self, delta, horizon, c):
        self.delta = delta
        self.horizon = horizon
        self.c = c
        if os.path.exists(f"logs/eg_log_{self.delta}.txt"):
            os.remove(f"logs/eg_log_{self.delta}.txt")

    def run_simulation(self, num_simulations):
        """
        Simulate multiple runs of the algorithm and return the mean reward and standard error
        """
        rewards = np.zeros(num_simulations)

        for sim in range(num_simulations):
            # initial means for [arm0, arm1]
            means = [0, 0]

            arm0_total = 0
            arm0_count = 0
            arm1_total = 0
            arm1_count = 0

            total_reward = 0

            for t in range(1, self.horizon + 1):
                epsilon = min(1, self.c / t)

                if np.random.rand() < epsilon:
                    # pick a random arm (exploration)
                    arm = np.random.choice([0, 1])
                    if arm == 0:
                        reward = np.random.normal(0, 1)
                        arm0_count += 1
                        arm0_total += reward
                        means[0] = arm0_total / arm0_count
                        total_reward += reward
                    else:
                        reward = np.random.normal(-self.delta, 1)
                        arm1_count += 1
                        arm1_total += reward
                        means[1] = arm1_total / arm1_count
                        total_reward += reward

                else:
                    # pick the arm with highest mean reward so far and pull
                    arm = np.argmax(means)

                    if arm == 0:
                        reward = np.random.normal(0, 1)
                        total_reward += reward
                    else:
                        reward = np.random.normal(-self.delta, 1)
                        total_reward += reward

            rewards[sim] = total_reward

            # Save results to file after each simulation
            with open(f"logs/eg_log_{self.delta}.txt", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([sim + 1, total_reward])

        mean_reward = np.mean(rewards)
        standard_error = np.std(rewards) / np.sqrt(num_simulations)

        return mean_reward, standard_error

    def optimal_reward(self):
        """
        Returns the optimal reward. For this class it is zero since we have two Guassian arms and
        the optimal arm has mean 0
        """
        return 0
