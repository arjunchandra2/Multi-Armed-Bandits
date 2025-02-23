"""
Explore Then Commit Algorithm
-----------------------------
Explore all arms for a fixed number of rounds, then commit to the best arm.

Implemented for two arms with Gaussian distirbutions:
    - Arm 1: N(0,1)
    - Arm 2: N(-delta,1)

Exploration phase will be m = ceil(n^(2/3))
"""

import numpy as np
import csv
import os


class ExploreThenCommit:

    def __init__(self, delta, horizon):
        self.delta = delta
        self.horizon = horizon
        self.m = int(np.ceil(horizon ** (2 / 3)))
        if os.path.exists(f"logs/etc_log_{self.delta}.txt"):
            os.remove(f"logs/etc_log_{self.delta}.txt")

        # this is the theoretical optimal value for m from the textbook
        # self.m = max(1, int(4/(delta**2) * np.log((horizon * delta**2)/(4))))

    def run_simulation(self, num_simulations):
        """
        Simulate multiple runs of the algorithm and return the mean reward and standard error
        """
        rewards = np.zeros(num_simulations)

        for sim in range(num_simulations):

            # explore phase: pull both arms m times
            arm1_explore = np.random.normal(0, 1, self.m)
            arm2_explore = np.random.normal(-self.delta, 1, self.m)

            # commit phase: pull the best arm for the horizon - 2m remaining rounds
            if np.mean(arm1_explore) > np.mean(arm2_explore):
                commit_rewards = np.random.normal(0, 1, self.horizon - 2 * self.m)
            else:
                commit_rewards = np.random.normal(
                    -self.delta, 1, self.horizon - 2 * self.m
                )

            total_reward = np.sum(
                np.concatenate((arm1_explore, arm2_explore, commit_rewards))
            )

            rewards[sim] = total_reward

            # Save results to file after each simulation
            with open(f"logs/etc_log_{self.delta}.txt", "a", newline="") as file:
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
