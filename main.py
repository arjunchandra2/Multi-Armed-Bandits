"""
main.py - driver code to run simulations for the bandit algorithms
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from explore_then_commit import ExploreThenCommit
from epsilon_greedy import EpsilonGreedy
from successive_elimination import SuccessiveElimination


def run_simulations(horizon, num_simulations, gap_parameters, c):
    """
    - Run multiple simulations for each MAB algorithm.
    - Plots regret vs gap parameter.
    """

    etc_regrets = []
    etc_standard_errors = []
    eg_regrets = []
    eg_standard_errors = []
    se_regrets = []
    se_standard_errors = []

    for delta in gap_parameters:
        # Explore-Then-Commit
        ETC = ExploreThenCommit(delta, horizon)
        expected_reward, std_error = ETC.run_simulation(num_simulations)
        regret = ETC.optimal_reward() - expected_reward
        etc_regrets.append(regret)
        etc_standard_errors.append(std_error)

        # Epsilon-Greedy
        EG = EpsilonGreedy(delta, horizon, c)
        expected_reward, std_error = EG.run_simulation(num_simulations)
        regret = EG.optimal_reward() - expected_reward
        eg_regrets.append(regret)
        eg_standard_errors.append(std_error)

        # Successive Elimination
        SE = SuccessiveElimination(delta, horizon)
        expected_reward, std_error = SE.run_simulation(num_simulations)
        regret = SE.optimal_reward() - expected_reward
        se_regrets.append(regret)
        se_standard_errors.append(std_error)

    # Compute theoretical regret bounds
    etc_regrets_theoretical = gap_parameters * ETC.m + gap_parameters * (
        horizon - ETC.m
    ) * np.exp(-ETC.m * np.power(gap_parameters, 2) / 4)
    eg_regrets_theoretical = c * gap_parameters + gap_parameters * horizon / c
    se_regrets_theoretical = np.sqrt(2 * horizon * np.log(horizon)) * np.ones_like(
        gap_parameters
    )

    # Plotting
    plt.errorbar(
        gap_parameters,
        etc_regrets,
        yerr=etc_standard_errors,
        fmt="-o",
        label="ETC",
        color="tab:red",
    )
    plt.errorbar(
        gap_parameters,
        eg_regrets,
        yerr=eg_standard_errors,
        fmt="-o",
        label=f"EG: c={c}",
        color="tab:green",
    )
    plt.errorbar(
        gap_parameters,
        se_regrets,
        yerr=se_standard_errors,
        fmt="-o",
        label="SE",
        color="tab:blue",
    )
    plt.plot(
        gap_parameters,
        etc_regrets_theoretical,
        label="ETC Bound",
        color="tab:red",
        linestyle="--",
    )
    plt.plot(
        gap_parameters,
        eg_regrets_theoretical,
        label="EG Bound",
        color="tab:green",
        linestyle="--",
    )
    plt.plot(
        gap_parameters,
        se_regrets_theoretical,
        label="SE Bound",
        color="tab:blue",
        linestyle="--",
    )
    plt.xlabel("Gap Parameter")
    plt.ylabel("Regret")
    plt.title("Regret vs Gap Parameter for MAB Algorithms")
    plt.legend(prop={"size": 8})
    plt.show()


def main():
    # Set parameters for the simulation here
    horizon = 1000
    num_simulations = 10**2
    gap_parameters = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    seed = 592
    c = 50

    os.makedirs("logs", exist_ok=True)
    np.random.seed(seed)
    run_simulations(horizon, num_simulations, gap_parameters, c)


main()
