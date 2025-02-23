import numpy as np
import csv
import os


class SuccessiveElimination:

    def __init__(self, delta, horizon):
        self.delta = delta
        self.horizon = horizon
        if os.path.exists(f"logs/se_log_{self.delta}.txt"):
            os.remove(f"logs/se_log_{self.delta}.txt")

    def run_simulation(self, num_simulations):
        """
        Run multiple simulations for the Successive Elimination algorithm.
        """
        rewards = np.zeros(num_simulations)

        for sim in range(num_simulations):
            t = 0

            arm1_num_pulls = 0
            arm2_num_pulls = 0
            arm1_total_reward = 0
            arm2_total_reward = 0

            arm1_mean = 0
            arm2_mean = 0

            arm1_bonus = 0
            arm2_bonus = 0

            total_reward = 0

            while t < self.horizon:
                # pull active arms (both are active)
                arm1 = np.random.normal(0, 1)
                arm1_num_pulls += 1
                arm1_total_reward += arm1
                total_reward += arm1

                arm2 = np.random.normal(-self.delta, 1)
                arm2_num_pulls += 1
                arm2_total_reward += arm2
                total_reward += arm2

                t += 2

                # update means and bonus terms
                arm1_mean = arm1_total_reward / arm1_num_pulls
                arm2_mean = arm2_total_reward / arm2_num_pulls

                arm1_bonus = np.sqrt(2 * np.log(self.horizon) / arm1_num_pulls)
                arm2_bonus = np.sqrt(2 * np.log(self.horizon) / arm2_num_pulls)

                # check if we can deactivate one of the arms, then we will only pull from the other arm
                if arm1_mean + arm1_bonus < arm2_mean - arm2_bonus:
                    #pull remaining rounds from arm2
                    total_reward += np.sum(np.random.normal(-self.delta, 1, (self.horizon - t)))
                    break
                elif arm2_mean + arm2_bonus < arm1_mean - arm1_bonus:
                    #pull remaining rounds from arm1
                    total_reward += np.sum(np.random.normal(0, 1, (self.horizon - t)))
                    break

            rewards[sim] = total_reward

            # Save results to file after each simulation
            with open(f"logs/se_log_{self.delta}.txt", "a", newline="") as file:
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
